#pragma GCC diagnostic ignored "-Wpedantic"

#include "ggml-amx.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include <algorithm>
#include <type_traits>

#if defined(__gnu_linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define ALWAYS_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif

#if defined(__AMX_INT8__)

namespace {

#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
#define VNNI_BLK 4

#define AMX_BLK_SIZE 32

#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

// parallel routines
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) { return (x + y - 1) / y; }

template <typename T>
void balance211(T n, T nth, T ith, T& n_start, T& n_end) {
  T& n_my = n_end;
  if (nth <= 1 || n == 0) {
    n_start = 0;
    n_my = n;
  } else {
    T n1 = div_up(n, nth);
    T n2 = n1 - 1;
    T T1 = n - n2 * nth;
    n_my = ith < T1 ? n1 : n2;
    n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
  }
  n_end += n_start;
}

// parallel with openmp, if openmp is not enabled, go sequential
template <typename func_t>
inline void parallel_for(int n, const func_t& f) {
#if defined(_OPENMP)
#pragma omp parallel
{
  int num_threads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  int tbegin, tend;
  balance211(n, num_threads, tid, tbegin, tend);
  f(tbegin, tend);
}
#else
  f(0, n);
#endif
}

// when openmp is enabled, parallel with openmp.
// when openmp is disabled, parallel with pthread.
template <typename func_t>
inline void parallel_for(int nth, int ith, int n, const func_t& f) {
#if defined(_OPENMP)
  // forbid nested parallelism of pthread and openmp
  GGML_ASSERT(nth == 1 && ith == 0);
  parallel_for(n, f);
#else
  int tbegin, tend;
  balance211(n, nth, ith, tbegin, tend);
  f(tbegin, tend);
#endif
}

// Forced unrolling
template <int n>
struct Unroll {
  template <typename Func, typename... Args>
  ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    Unroll<n - 1>{}(f, args...);
    f(std::integral_constant<int, n - 1>{}, args...);
  }
};

template <>
struct Unroll<1> {
  template <typename Func, typename... Args>
  ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    f(std::integral_constant<int, 0>{}, args...);
  }
};

// quantize A from float to `vec_dot_type`
template <typename T>
inline void from_float(const float * x, char * vy, int64_t k);

template <>
inline void from_float<block_q8_0>(const float * x, char * vy, int64_t k) {
  quantize_row_q8_0(x, vy, k);
}

template <>
inline void from_float<block_q8_1>(const float * x, char * vy, int64_t k) {
  quantize_row_q8_1(x, vy, k);
}

inline float FP16_TO_FP32(ggml_half val) {
  __m256i v = _mm256_setr_epi16(val, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  __m512 o = _mm512_cvtph_ps(v);
  return _mm512_cvtss_f32(o);
}

inline __m512 FP16_TO_FP32_VEC(ggml_half val) {
  __m256i v = _mm256_set1_epi16(val);
  return _mm512_cvtph_ps(v);
}

// type traits
template <typename T> struct PackedTypes {};
template <> struct PackedTypes<block_q4_0> { using type = int8_t; };
template <> struct PackedTypes<block_q4_1> { using type = uint8_t; };
template <> struct PackedTypes<block_q8_0> { using type = int8_t; };
template <typename T> using packed_B_type = typename PackedTypes<T>::type;

template <typename T>
struct do_compensate : std::integral_constant<
    bool, std::is_same<T, block_q8_0>::value> {};

template <typename T>
struct do_unpack : std::integral_constant<bool,
    std::is_same<T, block_q4_0>::value ||
    std::is_same<T, block_q4_1>::value> {};

#define GGML_DISPATCH_QTYPES(QT, ...)                                          \
  [&] {                                                                        \
    switch (QT) {                                                              \
      case GGML_TYPE_Q4_0: {                                                   \
        using type = block_q4_0;                                               \
        using vec_dot_type = block_q8_0;                                       \
        constexpr int blck_size = QK4_0;                                       \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case GGML_TYPE_Q4_1: {                                                   \
        using type = block_q4_1;                                               \
        using vec_dot_type = block_q8_1;                                       \
        constexpr int blck_size = QK4_1;                                       \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case GGML_TYPE_Q8_0: {                                                   \
        using type = block_q8_0;                                               \
        using vec_dot_type = block_q8_0;                                       \
        constexpr int blck_size = QK8_0;                                       \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      default:                                                                 \
        fprintf(stderr, "Unsupported quantized data type\n");                  \
    }                                                                          \
  }()

#define GGML_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...)                             \
  [&] {                                                                        \
    if (BOOL_V) {                                                              \
      constexpr bool BOOL_NAME = true;                                         \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr bool BOOL_NAME = false;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

// define amx tile config data structure
struct tile_config_t{
  uint8_t palette_id = 0;
  uint8_t start_row = 0;
  uint8_t reserved_0[14] = {0};
  uint16_t colsb[16] = {0};
  uint8_t rows[16] = {0};
};

// Notes: amx tile config
//
// Typically, TMUL calculates A and B of size 16 x 64 containing INT8 values,
// and accumulate the result to a 16 x 16 matrix C containing INT32 values,
//
// As many GGUF quantized types as `block_size` of 32, so a 16-16-32 config is used
// instead of the normally used 16-16-64 config.
//
//   Block A: {16, 32}, dtype = int8_t
//   Block B: {16, 32}, dtype = uint8_t/int8_t
//   Block C: {16, 16}, dtype = int32_t
//
// Block B needs to be prepacked to vnni format before feeding into  TMUL:
//   packed_B: from {n, k} to {k/vnni_blk, n, vnni_blck}, viewed in 2d, we get {8, 64}
//
// Therefore, we get tileconfig:
//             A    B    C
//    rows    16    8   16
//    colsb   32   64   16
//
// For tile distribution, follow a 2-2-4 pattern, e.g. A used TMM2-TMM3, B used TMM0-TMM1,
// C used TMM4-TMM7:
//            B TMM0  B TMM1
//    A TMM2  C TMM4  C TMM6
//    A TMM3  C TMM5  C TMM7
//
// Each `amx` kernel handles 4 blocks at a time: 2MB * 2NB, when m < 2 * BLOCK_M, unpack A
// will be needed.
//
// Here another commonly used pattern 1-3-3 is skipped, as it is mostly used when m <=16;
// and the sinlge batch gemm (m=1) has a special fast path with `avx512-vnni`.
//
// ref: https://www.intel.com/content/www/us/en/developer/articles/code-sample/
//   advanced-matrix-extensions-intrinsics-functions.html
//

#define TC_CONFIG_TILE(i, r, cb) tc.rows[i] = r; tc.colsb[i] = cb
void ggml_tile_config_init(void) {
  // TODO: try remove _tile_storeconfig ??
  static thread_local tile_config_t tc;
  tile_config_t current_tc;
  _tile_storeconfig(&current_tc);

  // load only when config changes
  if (tc.palette_id == 0 || (memcmp(&current_tc.colsb, &tc.colsb, sizeof(uint16_t) * 8) != 0 &&
                             memcmp(&current_tc.rows, &tc.rows, sizeof(uint8_t) * 8) != 0)) {
    tc.palette_id = 1;
    tc.start_row = 0;
    TC_CONFIG_TILE(TMM0, 8, 64);
    TC_CONFIG_TILE(TMM1, 8, 64);
    TC_CONFIG_TILE(TMM2, 16, 32);
    TC_CONFIG_TILE(TMM3, 16, 32);
    TC_CONFIG_TILE(TMM4, 16, 64);
    TC_CONFIG_TILE(TMM5, 16, 64);
    TC_CONFIG_TILE(TMM6, 16, 64);
    TC_CONFIG_TILE(TMM7, 16, 64);
    _tile_loadconfig(&tc);
    //printf("### ggml_tile_config_init finished!\n");
  }
}

// we need an extra 16 * 4B (TILE_N * int32_t) for each NB/KB block for compensation.
// See the notes `s8s8 igemm compensation in avx512-vnni` for detail.
template <typename TB>
int get_tile_size() {
  int tile_size = TILE_N * sizeof(TB);
  if (do_compensate<TB>::value) {
    tile_size += TILE_N * sizeof(int32_t);
  }
  return tile_size;
}

template <typename TB, int BLOCK_K>
int get_row_size(int K) {
  int KB = K / BLOCK_K;
  int row_size = KB * sizeof(TB);
  if (do_compensate<TB>::value) {
    row_size += KB * sizeof(int32_t);
  }
  return row_size;
}

// load A from memory to array when nrows can not fill in whole tile
void unpack_A(int8_t * RESTRICT tile, const block_q8_0 * RESTRICT A, int lda, int nr) {
  assert(nr != TILE_M);
  for (int m = 0; m < nr; ++m) {
    const __m256i v = _mm256_loadu_si256((const __m256i *)(A[m * lda].qs));
    _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
  }
}

void unpack_A(int8_t * RESTRICT tile, const block_q8_1 * RESTRICT A, int lda, int nr) {
  assert(nr != TILE_M);
  for (int m = 0; m < nr; ++m) {
    const __m256i v = _mm256_loadu_si256((const __m256i *)(A[m * lda].qs));
    _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
  }
}

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
inline __m256i bytes_from_nibbles_32(const uint8_t * rsi) {
  const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
  const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
  const __m256i lowMask = _mm256_set1_epi8( 0xF );
  return _mm256_and_si256(lowMask, bytes);
}

inline __m512i packNibbles(__m512i r0, __m512i r1) {
  return _mm512_or_si512(r0, _mm512_slli_epi16(r1, 4));
}

#define SHUFFLE_EPI32(a, b, mask) \
    _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), mask))
inline void transpose_8x8_32bit(__m256i * v, __m256i * v1) {
  // unpacking and 32-bit elements
  v1[0] = _mm256_unpacklo_epi32(v[0], v[1]);
  v1[1] = _mm256_unpackhi_epi32(v[0], v[1]);
  v1[2] = _mm256_unpacklo_epi32(v[2], v[3]);
  v1[3] = _mm256_unpackhi_epi32(v[2], v[3]);
  v1[4] = _mm256_unpacklo_epi32(v[4], v[5]);
  v1[5] = _mm256_unpackhi_epi32(v[4], v[5]);
  v1[6] = _mm256_unpacklo_epi32(v[6], v[7]);
  v1[7] = _mm256_unpackhi_epi32(v[6], v[7]);

  // shuffling the 32-bit elements
  v[0] = SHUFFLE_EPI32(v1[0], v1[2], 0x44);
  v[1] = SHUFFLE_EPI32(v1[0], v1[2], 0xee);
  v[2] = SHUFFLE_EPI32(v1[4], v1[6], 0x44);
  v[3] = SHUFFLE_EPI32(v1[4], v1[6], 0xee);
  v[4] = SHUFFLE_EPI32(v1[1], v1[3], 0x44);
  v[5] = SHUFFLE_EPI32(v1[1], v1[3], 0xee);
  v[6] = SHUFFLE_EPI32(v1[5], v1[7], 0x44);
  v[7] = SHUFFLE_EPI32(v1[5], v1[7], 0xee);

  // shuffling 128-bit elements
  v1[0] = _mm256_permute2f128_si256(v[2], v[0], 0x02);
  v1[1] = _mm256_permute2f128_si256(v[3], v[1], 0x02);
  v1[2] = _mm256_permute2f128_si256(v[6], v[4], 0x02);
  v1[3] = _mm256_permute2f128_si256(v[7], v[5], 0x02);
  v1[4] = _mm256_permute2f128_si256(v[2], v[0], 0x13);
  v1[5] = _mm256_permute2f128_si256(v[3], v[1], 0x13);
  v1[6] = _mm256_permute2f128_si256(v[6], v[4], 0x13);
  v1[7] = _mm256_permute2f128_si256(v[7], v[5], 0x13);
}

template <typename TB>
inline void pack_qs(void * RESTRICT packed_B, const TB * RESTRICT B, int KB) {
  int8_t tmp[8 * 64];
  __m256i v[8], v2[8];
  for (int n = 0; n < 8; ++n) {
    v[n] = bytes_from_nibbles_32(B[n * KB].qs);
  }
  transpose_8x8_32bit(v, v2);
  for (int n = 0; n < 8; ++n) {
    _mm256_storeu_si256((__m256i *)(tmp + n * 64), v2[n]);
  }
  for (int n = 0; n < 8; ++n) {
    v[n] = bytes_from_nibbles_32(B[(n + 8) * KB].qs);
  }
  transpose_8x8_32bit(v, v2);
  for (int n = 0; n < 8; ++n) {
    _mm256_storeu_si256((__m256i *)(tmp + n * 64 + 32), v2[n]);
  }

  // pack again with 128 to fully utilize vector length
  for (int n = 0; n < 8; n += 2) {
    __m512i r0 = _mm512_loadu_si512((const __m512i *)(tmp + n * 64));
    __m512i r1 = _mm512_loadu_si512((const __m512i *)(tmp + n * 64 + 64));
    __m512i r1r0 = packNibbles(r0, r1);
    _mm512_storeu_si512((__m512i *)((char *)packed_B + n * 32), r1r0);
  }
}

template <>
inline void pack_qs<block_q8_0>(void * RESTRICT packed_B, const block_q8_0 * RESTRICT B, int KB) {
  __m256i v[8], v2[8];
  for (int n = 0; n < 8; ++n) {
    v[n] = _mm256_loadu_si256((const __m256i *)(B[n * KB].qs));
  }
  transpose_8x8_32bit(v, v2);
  for (int n = 0; n < 8; ++n) {
    _mm256_storeu_si256((__m256i *)((char *)packed_B + n * 64), v2[n]);
  }
  for (int n = 0; n < 8; ++n) {
    v[n] = _mm256_loadu_si256((const __m256i *)(B[(n + 8) * KB].qs));
  }
  transpose_8x8_32bit(v, v2);
  for (int n = 0; n < 8; ++n) {
    _mm256_storeu_si256((__m256i *)((char *)packed_B + n * 64 + 32), v2[n]);
  }
}

// pack B to vnni formats in 4bits or 8 bits
void pack_B(void * RESTRICT packed_B, const block_q4_0 * RESTRICT B, int KB) {
  pack_qs(packed_B, B, KB);
  ggml_half * d0 = reinterpret_cast<ggml_half *>((char *)packed_B + TILE_N * TILE_K / 2);
  for (int n = 0; n < TILE_N; ++n) {
    d0[n] = B[n * KB].d;
  }
}

void pack_B(void * RESTRICT packed_B, const block_q4_1 * RESTRICT B, int KB) {
  pack_qs(packed_B, B, KB);
  ggml_half * d0 = reinterpret_cast<ggml_half *>((char *)packed_B + TILE_N * TILE_K / 2);
  ggml_half * m0 = d0 + TILE_N;
  for (int n = 0; n < TILE_N; ++n) {
    d0[n] = B[n * KB].d;
    m0[n] = B[n * KB].m;
  }
}

inline void s8s8_compensation(void * RESTRICT packed_B) {
  // packed_B layout:
  //   quants {TILE_N, TILEK}  int8_t
  //   d0     {TILE_N}      ggml_half
  //   comp   {TILE_N}        int32_t
  const int offset = TILE_N * TILE_K + TILE_N * sizeof(ggml_half);
  __m512i vcomp = _mm512_setzero_si512();
  const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));
  for (int k = 0; k < 8; ++k) {
    __m512i vb = _mm512_loadu_si512((const __m512i *)((const char *)packed_B + k * 64));
    vcomp = _mm512_dpbusd_epi32(vcomp, off, vb);
  }
  _mm512_storeu_si512((__m512i *)((char *)(packed_B) + offset), vcomp);
}

void pack_B(void * RESTRICT packed_B, const block_q8_0 * RESTRICT B, int KB) {
  pack_qs(packed_B, B, KB);
  ggml_half * d0 = reinterpret_cast<ggml_half *>((char *)packed_B + TILE_N * TILE_K);
  for (int n = 0; n < TILE_N; ++n) {
    d0[n] = B[n * KB].d;
  }
  s8s8_compensation(packed_B);
}

template<typename TB, typename packed_B_t = packed_B_type<TB>>
void unpack_B(packed_B_t * RESTRICT tile, const void * RESTRICT packed_B) {
  GGML_UNUSED(tile);
  GGML_UNUSED(packed_B);
};

template <>
void unpack_B<block_q4_0>(int8_t * RESTRICT tile, const void * RESTRICT packed_B) {
  const __m512i off = _mm512_set1_epi8(8);
  for (int n = 0; n < 8; n += 2) {
    __m512i bytes = _mm512_loadu_si512((const __m512i *)((const char *)packed_B + n * 32));
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    const __m512i r0 = _mm512_sub_epi8(_mm512_and_si512(bytes, lowMask), off);
    const __m512i r1 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask), off);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
  }
}

template <>
void unpack_B<block_q4_1>(uint8_t * RESTRICT tile, const void * RESTRICT packed_B) {
  for (int n = 0; n < 8; n += 2) {
    __m512i bytes = _mm512_loadu_si512((const __m512i *)((const char *)packed_B + n * 32));
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    const __m512i r0 = _mm512_and_si512(bytes, lowMask);
    const __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
  }
}

template <typename TA, typename TB, bool is_acc>
struct acc_C {};

template <bool is_acc>
struct acc_C<block_q8_0, block_q4_0, is_acc> {
  static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_0 * A, int lda, const void * packed_B, int nr) {
    const int offset = TILE_N * TILE_K / 2;
    const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset)));

    for (int m = 0; m < nr; ++m) {
      const __m512 vd1 = _mm512_set1_ps(GGML_FP16_TO_FP32(A[m * lda].d));
      const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

      __m512 vsum;
      if (is_acc) {
        vsum = _mm512_loadu_ps(C + m * ldc);
      } else {
        vsum = _mm512_set1_ps(0.f);
      }
      vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
      _mm512_storeu_ps(C + m * ldc, vsum);
    }
  }
};

static void print_512(const __m512 _x)
{
    __attribute__((aligned(32)))
    float a[16];
    _mm512_storeu_ps(a, _x);

    for (int i = 0; i < 16; i++)
    {
        printf("%.5f ", a[i]);
    }
    printf("\n");
}

static void print_512i(const __m512i _x)
{
    __attribute__((aligned(32)))
    int32_t a[16];
    _mm512_store_si512(a, _x);

    for (int i = 0; i < 16; i++)
    {
        printf("%4d ", a[i]);
    }
    printf("\n");
}

template <bool is_acc>
struct acc_C<block_q8_1, block_q4_1, is_acc> {
  static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_1 * A, int lda, const void * packed_B, int nr) {
    const int offset = TILE_N * TILE_K / 2;
    const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset)));
    const __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset + TILE_N * sizeof(ggml_half))));

    for (int m = 0; m < nr; ++m) {
      const __m512 vd1 = _mm512_set1_ps(GGML_FP16_TO_FP32(A[m * lda].d));
      const __m512 vs1 = _mm512_set1_ps(GGML_FP16_TO_FP32(A[m * lda].s));
      const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

      __m512 vsum;
      if (is_acc) {
        vsum = _mm512_loadu_ps(C + m * ldc);
      } else {
        vsum = _mm512_set1_ps(0.f);
      }
      vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
      vsum = _mm512_fmadd_ps(vm0, vs1, vsum);
      _mm512_storeu_ps(C + m * ldc, vsum);
    }
  }
};

template <bool is_acc>
struct acc_C<block_q8_0, block_q8_0, is_acc> {
  static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_0 * A, int lda, const void * packed_B, int nr) {
    const int offset = TILE_N * TILE_K;
    const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset)));

    for (int m = 0; m < nr; ++m) {
      const __m512 vd1 = _mm512_set1_ps(GGML_FP16_TO_FP32(A[m * lda].d));
      const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

      __m512 vsum;
      if (is_acc) {
        vsum = _mm512_loadu_ps(C + m * ldc);
      } else {
        vsum = _mm512_set1_ps(0.f);
      }
      vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
      _mm512_storeu_ps(C + m * ldc, vsum);
    }
  }
};

// re-organize in the format {NB, KB, TILE_SIZE}:
#define PACKED_INDEX(n, k, KB, tile_size) (n * KB + k) * tile_size

template<typename TB, int BLOCK_K>
void convert_B_packed_format(void * RESTRICT packed_B, const TB * RESTRICT B, int N, int K) {
  const int NB = N / TILE_N;
  const int KB = K / BLOCK_K;
  const int TILE_SIZE = get_tile_size<TB>();

  //printf("### convert_B_packed_format: NB = %d, KB = %d, tile_size = %d\n", NB, KB, TILE_SIZE);
  // parallel on NB should be enough
  parallel_for(NB, [&](int begin, int end) {
    for (int n = begin; n < end; ++n) {
      for (int k = 0; k < KB; ++k) {
        int n0 = n * TILE_N;
        pack_B((char *)packed_B + PACKED_INDEX(n, k, KB, TILE_SIZE), &B[n0 * KB + k], KB);
      }
    }
  });
}

// TODO: remove `BLOCK_K` ?
template <typename TA, typename TB, typename TC, int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni {};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_0, block_q4_0, float, BLOCK_M, BLOCK_N, BLOCK_K> {
  static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {
    //printf("### tinygemm_kernel_vnni - q4_0, BLOCK_M = %d, BLOCK_N = %d, BLOCK_K = %d\n", BLOCK_M, BLOCK_N, BLOCK_K);

    constexpr int COLS = BLOCK_N / 16;
    const int TILE_SIZE = TILE_N * sizeof(block_q4_0);

    const block_q8_0 * RESTRICT A = static_cast<const block_q8_0 *>(_A);
    const char * RESTRICT B = static_cast<const char *>(_B);

    __m512i va[8];
    __m512 vc[COLS];
    __m512 vd1;

    // sum of offsets, shared across COLS
    //
    // avx512-vnni does not have `_mm512_dpbssd_epi32`,
    // need to transfrom ss to us:
    //   a * (b - 8) is equavilent to b * a - 8 * a
    //   s    u   u                   u   s   u   s
    //
    __m512i vcomp;

    const __m512i off = _mm512_set1_epi8(8);
    const __m512i lowMask = _mm512_set1_epi8(0xF);

    auto loadc = [&](int col) {
      vc[col] = _mm512_setzero_ps();
    };
    Unroll<COLS>{}(loadc);

    auto compute = [&](int col, int i) {
      // load a and compute compensation
      if (col == 0) {
        const int32_t * a_ptr = reinterpret_cast<const int32_t *>(A[0 * KB + i].qs);
        vcomp = _mm512_setzero_si512();
        for (int k = 0; k < 8; ++k) {
          va[k] = _mm512_set1_epi32(a_ptr[k]);
          vcomp = _mm512_dpbusd_epi32(vcomp, off, va[k]);
        }
        vd1 = _mm512_set1_ps(GGML_FP16_TO_FP32(A[0 * KB + i].d));
      }

      // load b
       __m512i vsum = _mm512_setzero_si512();
      const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
      for (int k = 0; k < 8; k += 2) {
        __m512i bytes = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 32));
        __m512i vb0 = _mm512_and_si512(bytes, lowMask);
        vsum = _mm512_dpbusd_epi32(vsum, vb0, va[k + 0]);
        __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
        vsum = _mm512_dpbusd_epi32(vsum, vb1, va[k + 1]);
      }
      const int offset = TILE_N * TILE_K / 2;
      const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset)));
      vsum = _mm512_sub_epi32(vsum, vcomp);

      vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
    };

    for (int i = 0; i < KB; ++i) {
      Unroll<COLS>{}(compute, i);
    }

    //store to C
    auto storec = [&](int col) {
      _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
    };
    Unroll<COLS>{}(storec);
  }
};

template <int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_1, block_q4_1, float, 1, BLOCK_N, BLOCK_K> {
  static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {
    //printf("### tinygemm_kernel_vnni - q4_1, BLOCK_M = %d, BLOCK_N = %d, BLOCK_K = %d\n", 1, BLOCK_N, BLOCK_K);

    constexpr int COLS = BLOCK_N / 16;
    const int TILE_SIZE = TILE_N * sizeof(block_q4_1);

    const block_q8_1 * RESTRICT A = static_cast<const block_q8_1 *>(_A);
    const char * RESTRICT B = static_cast<const char *>(_B);

    __m512i va[8];
    __m512i vb[8];
    __m512 vc[COLS];
    __m512 vd1, vs1;

    const __m512i lowMask = _mm512_set1_epi8(0xF);

    auto loadc = [&](int col) {
      vc[col] = _mm512_setzero_ps();
    };
    Unroll<COLS>{}(loadc);

    auto compute = [&](int col, int i) {
      // load a
      if (col == 0) {
        const int32_t * a_ptr = reinterpret_cast<const int32_t *>(A[0 * KB + i].qs);
        for (int k = 0; k < 8; ++k) {
          va[k] = _mm512_set1_epi32(a_ptr[k]);
        }
        vd1 = _mm512_set1_ps(GGML_FP16_TO_FP32(A[0 * KB + i].d));
        vs1 = _mm512_set1_ps(GGML_FP16_TO_FP32(A[0 * KB + i].s));
      }

      // load b
      const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
      for (int k = 0; k < 8; k += 2) {
        __m512i bytes = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 32));
        vb[k + 0] = _mm512_and_si512(bytes, lowMask);
        vb[k + 1] = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
      }
      const int offset = TILE_N * TILE_K / 2;
      const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset)));
      const __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset + TILE_N * sizeof(ggml_half))));

       __m512i vsum = _mm512_setzero_si512();
      for (int k = 0; k < 8; ++k) {
        vsum = _mm512_dpbusd_epi32(vsum, vb[k], va[k]);
      }

      vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
      vc[col] = _mm512_fmadd_ps(vm0, vs1, vc[col]);
    };

    for (int i = 0; i < KB; ++i) {
      Unroll<COLS>{}(compute, i);
    }

    //store to C
    auto storec = [&](int col) {
      _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
    };
    Unroll<COLS>{}(storec);
  }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_0, block_q8_0, float, BLOCK_M, BLOCK_N, BLOCK_K> {
  static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {
    //printf("### tinygemm_kernel_vnni - q8_0, BLOCK_M = %d, BLOCK_N = %d, BLOCK_K = %d\n", BLOCK_M, BLOCK_N, BLOCK_K);

    constexpr int COLS = BLOCK_N / 16;
    const int TILE_SIZE = TILE_N * sizeof(block_q8_0) + TILE_N * sizeof(int32_t);

    const block_q8_0 * RESTRICT A = static_cast<const block_q8_0 *>(_A);
    const char * RESTRICT B = static_cast<const char *>(_B);

    __m512i va[8];
    __m512i vb[8];
    __m512 vc[COLS];
    __m512 vd1;

    // Notes: s8s8 igemm compensation in avx512-vnni
    // change s8s8 to u8s8 with compensate
    //   a * b = (a + 128) * b - 128 * b
    //   s   s       u       s    u    s
    //
    // (128 * b is pre-computed when packing B to vnni formats)
    //
    const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));

    auto loadc = [&](int col) {
      vc[col] = _mm512_setzero_ps();
    };
    Unroll<COLS>{}(loadc);

    auto compute = [&](int col, int i) {
      // load a and add offset 128
      if (col == 0) {
        const int32_t * a_ptr = reinterpret_cast<const int32_t *>(A[0 * KB + i].qs);
        for (int k = 0; k < 8; ++k) {
          va[k] = _mm512_set1_epi32(a_ptr[k]);
          va[k] = _mm512_add_epi8(va[k], off);
        }
        vd1 = _mm512_set1_ps(GGML_FP16_TO_FP32(A[0 * KB + i].d));
      }

      // load b
      const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
      for (int k = 0; k < 8; ++k) {
        vb[k] = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 64));
      }
      const int offset = TILE_N * TILE_K;
      const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset)));
      const int offset2 = TILE_N * TILE_K + TILE_N * sizeof(ggml_half);
      const __m512i vcomp = _mm512_loadu_si512((const __m512i *)(b_ptr + offset2));

      __m512i vsum = _mm512_setzero_si512();
      for (int k = 0; k < 8; ++k) {
        vsum = _mm512_dpbusd_epi32(vsum, va[k], vb[k]);
      }
      vsum = _mm512_sub_epi32(vsum, vcomp);

      vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
    };

    for (int i = 0; i < KB; ++i) {
      Unroll<COLS>{}(compute, i);
    }

    //store to C
    auto storec = [&](int col) {
      _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
    };
    Unroll<COLS>{}(storec);
  }
};

#define LAUNCH_TINYGEMM_KERNEL_VNNI(NB_SIZE)                                         \
    tinygemm_kernel_vnni<vec_dot_type, type, float, 1, NB_SIZE, blck_size>::apply(   \
        KB, (const char *)wdata + 0 * row_size_A,                                    \
        (const char *)src0->extra + PACKED_INDEX(nb * kTilesN, 0, KB, TILE_SIZE),    \
        (float *) dst->data + 0 * N + nb_start, ldc)

// TODO: remove `BLOCK_K` ?
template <typename TA, typename TB, typename TC, int BLOCK_K>
void tinygemm_kernel_amx(int M, int N, int KB, const void * RESTRICT _A, const void * RESTRICT _B, TC * RESTRICT C, int ldc) {
  using packed_B_t = packed_B_type<TB>;
  const int TILE_SIZE = get_tile_size<TB>();
  const bool need_unpack = do_unpack<TB>::value;

  GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
  const TA * RESTRICT A = static_cast<const TA *>(_A);
  const char * RESTRICT B = static_cast<const char *>(_B);

  const int m0 = std::min(M, TILE_M);
  const int m1 = std::max(M - TILE_M, 0);
  const int lda = KB * sizeof(TA);
  //const int ldb = KB * sizeof(TB);

  static thread_local packed_B_t Tile0[TILE_N * TILE_K];
  static thread_local packed_B_t Tile1[TILE_N * TILE_K];
  static thread_local int8_t Tile23[TILE_M * TILE_K];

  static thread_local int32_t TileC0[TILE_M * TILE_N * 4];
  static thread_local int32_t TileC1[TILE_M * TILE_N * 4];

  // double buffering C to interleave avx512 and amx
  int32_t * C_cur = TileC0;
  int32_t * C_pre = TileC1;

  #define Tile4(base) base
  #define Tile5(base) base + TILE_M * TILE_N
  #define Tile6(base) base + 2 * TILE_M * TILE_N
  #define Tile7(base) base + 3 * TILE_M * TILE_N

  if (M == 2 * TILE_M) {
    // i = 0
    if (need_unpack) {
      unpack_B<TB>(Tile0, B + PACKED_INDEX(0, 0, KB, TILE_SIZE));
      _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
    } else {
      _tile_loadd(TMM0, B + PACKED_INDEX(0, 0, KB, TILE_SIZE), TILE_N * VNNI_BLK);
    }
    _tile_zero(TMM4);
    _tile_loadd(TMM2, A[0].qs, lda);
    _tile_dpbssd(TMM4, TMM2, TMM0);
    _tile_stored(TMM4, Tile4(C_pre), TILE_N * sizeof(int32_t));

    _tile_zero(TMM5);
    _tile_loadd(TMM3, A[TILE_M * KB + 0].qs, lda);
    _tile_dpbssd(TMM5, TMM3, TMM0);
    _tile_stored(TMM5, Tile5(C_pre), TILE_N * sizeof(int32_t));

    if (need_unpack) {
      unpack_B<TB>(Tile1, B + PACKED_INDEX(1, 0, KB, TILE_SIZE));
      _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
    } else {
      _tile_loadd(TMM1, B + PACKED_INDEX(1, 0, KB, TILE_SIZE), TILE_N * VNNI_BLK);
    }
    _tile_zero(TMM6);
    _tile_dpbssd(TMM6, TMM2, TMM1);
    _tile_stored(TMM6, Tile6(C_pre), TILE_N * sizeof(int32_t));

    _tile_zero(TMM7);
    _tile_dpbssd(TMM7, TMM3, TMM1);
    _tile_stored(TMM7, Tile7(C_pre), TILE_N * sizeof(int32_t));

    for (int i = 1; i < KB; ++i) {
      // index of previous iter
      const int ii = i - 1;
      GGML_DISPATCH_BOOL(ii > 0, is_acc, [&] {
        if (need_unpack) {
          unpack_B<TB>(Tile0, B + PACKED_INDEX(0, i, KB, TILE_SIZE));
          _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
        } else {
          _tile_loadd(TMM0, B + PACKED_INDEX(0, i, KB, TILE_SIZE), TILE_N * VNNI_BLK);
        }
        _tile_zero(TMM4);
        _tile_loadd(TMM2, A[i].qs, lda);
        acc_C<TA, TB, is_acc>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M);

        _tile_dpbssd(TMM4, TMM2, TMM0);
        _tile_stored(TMM4, Tile4(C_cur), TILE_N * sizeof(int32_t));

        _tile_zero(TMM5);
        _tile_loadd(TMM3, A[TILE_M * KB + i].qs, lda);
        acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M);

        _tile_dpbssd(TMM5, TMM3, TMM0);
        _tile_stored(TMM5, Tile5(C_cur), TILE_N * sizeof(int32_t));

        if (need_unpack) {
          unpack_B<TB>(Tile1, B + PACKED_INDEX(1, i, KB, TILE_SIZE));
          _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
        } else {
          _tile_loadd(TMM1, B + PACKED_INDEX(1, i, KB, TILE_SIZE), TILE_N * VNNI_BLK);
        }
        _tile_zero(TMM6);
        acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M);

        _tile_dpbssd(TMM6, TMM2, TMM1);
        _tile_stored(TMM6, Tile6(C_cur), TILE_N * sizeof(int32_t));

        _tile_zero(TMM7);
        acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M);

        _tile_dpbssd(TMM7, TMM3, TMM1);
        _tile_stored(TMM7, Tile7(C_cur), TILE_N * sizeof(int32_t));

        std::swap(C_cur, C_pre);
      });
    }
    // final accumulation
    {
      int ii = KB - 1;
      acc_C<TA, TB, true>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M);
      acc_C<TA, TB, true>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M);
      acc_C<TA, TB, true>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M);
      acc_C<TA, TB, true>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M);
    }
  } else {
    for (int i = 0; i < KB; ++i) {
      _tile_zero(TMM4);
      _tile_zero(TMM6);
      if (m1 != 0) {
        _tile_zero(TMM5);
        _tile_zero(TMM7);
      }

      if (need_unpack) {
        unpack_B<TB>(Tile0, B + PACKED_INDEX(0, i, KB, TILE_SIZE));
        _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
      } else {
        _tile_loadd(TMM0, B + PACKED_INDEX(0, i, KB, TILE_SIZE), TILE_N * VNNI_BLK);
      }

      if (need_unpack) {
        unpack_B<TB>(Tile1, B + PACKED_INDEX(1, i, KB, TILE_SIZE));
        _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
      } else {
        _tile_loadd(TMM1, B + PACKED_INDEX(1, i, KB, TILE_SIZE), TILE_N * VNNI_BLK);
      }

      if (m0 == TILE_M) {
        _tile_loadd(TMM2, A[i].qs, lda);
      } else {
        unpack_A(Tile23, &A[i], KB, m0);
        _tile_loadd(TMM2, Tile23, TILE_K);
      }

      _tile_dpbssd(TMM4, TMM2, TMM0);
      _tile_dpbssd(TMM6, TMM2, TMM1);

      _tile_stored(TMM4, Tile4(C_cur), TILE_N * sizeof(int32_t));
      _tile_stored(TMM6, Tile6(C_cur), TILE_N * sizeof(int32_t));

      GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
        acc_C<TA, TB, is_acc>::apply(C,          ldc, Tile4(C_cur), &A[i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m0);
        acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_cur), &A[i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m0);
      });
      if (m1 != 0) {
        unpack_A(Tile23, &A[TILE_M * KB + i], KB, m1);
        _tile_loadd(TMM3, Tile23, TILE_K);

        _tile_dpbssd(TMM5, TMM3, TMM0);
        _tile_dpbssd(TMM7, TMM3, TMM1);
        _tile_stored(TMM5, Tile5(C_cur), TILE_N * sizeof(int32_t));
        _tile_stored(TMM7, Tile7(C_cur), TILE_N * sizeof(int32_t));
        GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
          acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Tile5(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m1);
          acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m1);
        });
      }
    }
  }
  return;
}

} // anonymous namespace

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

void print_C(float * dst, int64_t nrows, int64_t ncols) {
  for (int m = 0; m < nrows; ++m) {
    for (int n = 0; n < ncols; ++n) {
      printf(" %.3f", dst[m * ncols + n]);
    }
    printf("\n");
  }
}

bool ggml_amx_init() {
#if defined(__gnu_linux__)
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    fprintf(stderr, "AMX is not ready to be used!\n");
    return false;
  }
  return true;
#elif defined(_WIN32)
  return true;
#endif
}

bool ggml_compute_forward_mul_mat_use_amx(struct ggml_tensor * dst) {
  // load tile config
  ggml_tile_config_init();

  const struct ggml_tensor * src0 = dst->src[0];
  const struct ggml_tensor * src1 = dst->src[1];

  const enum ggml_type type = src0->type;
  const int64_t ne0 = dst->ne[0];

  // amx kernels enables for Q4_0, Q4_1
  bool has_amx_kernels = (type == GGML_TYPE_Q4_0) ||
      (type == GGML_TYPE_Q4_1) ||
      (type == GGML_TYPE_Q8_0);

  // handle only 2d gemm for now
  auto is_contiguous_2d = [](const struct ggml_tensor * t) {
    return ggml_is_contiguous(t) && t->ne[3] == 1 && t->ne[2] == 1;
  };

  return dst->op != GGML_OP_MUL_MAT_ID &&
      is_contiguous_2d(src0) &&
      is_contiguous_2d(src1) &&
      src1->type == GGML_TYPE_F32 &&
      has_amx_kernels &&
      // out features is 32x
      ne0 % (TILE_N * 2) == 0;
}

// NB: mixed dtype gemm with Advanced Matrix Extensions (Intel AMX)
//
// src0: weight in shape of {N, K}, quantized
// src1: input  in shape of {M, K}, float32
// dst:  output in shape of {M, N}, float32
//
// the function performs: dst = src1 @ src0.T
//
void ggml_mul_mat_amx(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
  struct ggml_tensor * src0 = dst->src[0];
  struct ggml_tensor * src1 = dst->src[1];

  const enum ggml_type TYPE = src0->type;

  const int ith = params->ith;
  const int nth = params->nth;

  const int M = dst->ne[1];
  const int N = dst->ne[0];
  const int K = src0->ne[0];
  const int ldc = dst->nb[1] / dst->nb[0];

  // packed A data
  char* wdata = static_cast<char *>(params->wdata);

  if (params->type == GGML_TASK_TYPE_INIT) {
    if (ith != 0) {
      return;
    }

    GGML_DISPATCH_QTYPES(TYPE, [&] {
      const size_t row_size_A = K / blck_size * sizeof(vec_dot_type);
      GGML_ASSERT(params->wsize >= M * row_size_A);

      // quantize mat A
      const float * A_data = static_cast<const float *>(src1->data);
      for (int m = 0; m < M; ++m) {
        from_float<vec_dot_type>(A_data + m * K, wdata + m * row_size_A, K);
      }

      GGML_ASSERT(TILE_K == blck_size);
      // pack mat B to vnni format
      if (src0->extra == nullptr) {
        const size_t row_size_B = get_row_size<type, blck_size>(K);
        src0->extra = aligned_alloc(64, N * row_size_B);
        convert_B_packed_format<type, blck_size>((void *)src0->extra, (const type *)src0->data, N, K);
      }
    });
    return;
  }

  if (params->type == GGML_TASK_TYPE_FINALIZE) {
    return;
  }

  // expect weight to be pre-packed here
  GGML_ASSERT(src0->extra != nullptr);

  if (M == 1) {
    // MB = 1 and handle 4 tiles in each block
    constexpr int kTilesN = 8;
    constexpr int BLOCK_N = TILE_N * kTilesN;
    const int NB = div_up(N, BLOCK_N);

    parallel_for(nth, ith, NB, [&](int begin, int end) {
      GGML_DISPATCH_QTYPES(TYPE, [&] {
        const int KB = K / blck_size;
        const int TILE_SIZE = get_tile_size<type>();
        const int row_size_A = KB * sizeof(vec_dot_type);
        for (int i = begin; i < end; ++i) {
          int nb = i;
          int nb_start = nb * BLOCK_N;
          int nb_size = std::min(BLOCK_N, N - nb_start); // 32, 64, 96

          switch (nb_size) {
            //case 160: LAUNCH_TINYGEMM_KERNEL_VNNI(160); break;
            case 128: LAUNCH_TINYGEMM_KERNEL_VNNI(128); break;
            case 96: LAUNCH_TINYGEMM_KERNEL_VNNI(96); break;
            case 64: LAUNCH_TINYGEMM_KERNEL_VNNI(64); break;
            case 32: LAUNCH_TINYGEMM_KERNEL_VNNI(32); break;
            default: fprintf(stderr, "Unexpected n block size!\n");
          }
        }
      });
    });
    return;
  }

  // handle 4 tiles at a tile
  constexpr int BLOCK_M = TILE_M * 2;
  constexpr int BLOCK_N = TILE_N * 2;
  const int MB = div_up(M, BLOCK_M);
  const int NB = div_up(N, BLOCK_N);

  parallel_for(nth, ith, MB * NB, [&](int begin, int end) {
    GGML_DISPATCH_QTYPES(TYPE, [&] {
      const int KB = K / blck_size;
      const int TILE_SIZE = get_tile_size<type>();
      const int row_size_A = KB * sizeof(vec_dot_type);

      for (int i = begin; i < end; ++i) {
        int mb = i / NB;
        int nb = i % NB;

        int mb_start = mb * BLOCK_M;
        int mb_size = std::min(BLOCK_M, M - mb_start);
        int nb_start = nb * BLOCK_N;
        int nb_size = BLOCK_N;

        tinygemm_kernel_amx<vec_dot_type, type, float, blck_size>(
            mb_size, nb_size, KB,
            (const char *)wdata + mb_start * row_size_A,
            (const char *)src0->extra + PACKED_INDEX(nb * 2, 0, KB, TILE_SIZE),
            (float *) dst->data + mb_start * N + nb_start, ldc);
      }
    });
  });
}

#else // if defined(__AMX_INT8__)

bool ggml_amx_init() {
  fprintf(stderr, "GGML is not compiled with AMX support!\n");
  return false;
}

bool ggml_compute_forward_mul_mat_use_amx(struct ggml_tensor * dst) {
  fprintf(stderr, "GGML is not compiled with AMX support!\n");
  return false;
}

void ggml_mul_mat_amx(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
  fprintf(stderr, "GGML is not compiled with AMX support!\n");
}

#endif // if defined(__AMX_INT8__)
