#pragma GCC diagnostic ignored "-Wpedantic"

#include "ggml-amx.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#if defined(__gnu_linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif


#include <iostream>

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
template <typename T>
static void balance211(T n, T nth, T ith, T& n_start, T& n_end) {
  T& n_my = n_end;
  if (nth <= 1 || n == 0) {
    n_start = 0;
    n_my = n;
  } else {
    T n1 = (n + nth - 1) / nth;
    T n2 = n1 - 1;
    T T1 = n - n2 * nth;
    n_my = ith < T1 ? n1 : n2;
    n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
  }
  n_end += n_start;
}

//
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

template <typename T> struct PackedTypes {};
template <> struct PackedTypes<block_q4_0> { using type = int8_t; };
template <> struct PackedTypes<block_q4_1> { using type = uint8_t; };
template <typename T> using packed_B_type = typename PackedTypes<T>::type;

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

template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) { return (x + y - 1) / y; }

// define tile config data structure
struct tile_config_t{
  uint8_t palette_id = 0;
  uint8_t start_row = 0;
  uint8_t reserved_0[14] = {0};
  uint16_t colsb[16] = {0};
  uint8_t rows[16] = {0};
};

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
    printf("###\nggml_tile_config_init finished!\n");
  }
}

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

static void print_512i(const __m512i _x)
{
    __attribute__((aligned(32)))
    uint8_t a[64];
    _mm512_store_si512(a, _x);

    for (int i = 0; i < 64; i++)
    {
        printf("%3d ", a[i]);
    }
    printf("\n");
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

template<typename TB, typename packed_B_t = packed_B_type<TB>>
void unpack_B(packed_B_t * RESTRICT tile, const void * RESTRICT packed_B);

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

template <typename TA, typename TB>
void acc_C(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const TA * A, int lda, const void * packed_B, int nr, bool is_acc);

template <>
void acc_C<block_q8_0, block_q4_0>(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_0 * A, int lda, const void * packed_B, int nr, bool is_acc) {
  const int offset = TILE_N * TILE_K /2;
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

static void print_512(const __m512 _x)
{
    __attribute__((aligned(32)))
    float a[16];
    _mm512_storeu_ps(a, _x);

    for (int i = 0; i < 16; i++)
    {
        printf("%.3f ", a[i]);
    }
    printf("\n");
}

template <>
void acc_C<block_q8_1, block_q4_1>(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_1 * A, int lda, const void * packed_B, int nr, bool is_acc) {
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

// re-organize in the format {NB, KB, TILE_SIZE}:
#define PACKED_INDEX(n, k, KB, tile_size) (n * KB + k) * tile_size

template<typename TB, int BLCK_SIZE>
void convert_B_packed_format(void * RESTRICT packed_B, const TB * RESTRICT B, int N, int K) {
  const int NB = N / TILE_N;
  const int KB = K / BLCK_SIZE;
  const int TILE_SIZE = TILE_N * sizeof(TB);

  printf("### convert_B_packed_format: NB = %d, KB = %d, tile_size = %d\n", NB, KB, TILE_SIZE);
  for (int n = 0; n < NB; ++n) {
    for (int k = 0; k < KB; ++k) {
      int n0 = n * TILE_N;
      pack_B((char *)packed_B + PACKED_INDEX(n, k, KB, TILE_SIZE), &B[n0 * KB + k], KB);
    }
  }
}

template <typename TA, typename TB, typename TC, int BLCK_SIZE>
void tiny_gemm_kernel(int M, int N, int KB, const void * RESTRICT _A, const void * RESTRICT _B, TC * RESTRICT C, int ldc) {
  using packed_B_t = packed_B_type<TB>;
  const int TILE_SIZE = TILE_N * sizeof(TB);

  GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
  const TA * RESTRICT A = static_cast<const TA *>(_A);
  const char * RESTRICT B = static_cast<const char *>(_B);

  const int m0 = std::min(M, TILE_M);
  const int m1 = std::max(M - TILE_M, 0);
  const int lda = KB * sizeof(TA);
  //const int ldb = KB * sizeof(TB);

  packed_B_t Tile0[TILE_N * TILE_K];
  packed_B_t Tile1[TILE_N * TILE_K];
  int8_t Tile23[TILE_M * TILE_K];

  int32_t Tile4[TILE_M * TILE_N];
  int32_t Tile5[TILE_M * TILE_N];
  int32_t Tile6[TILE_M * TILE_N];
  int32_t Tile7[TILE_M * TILE_N];

  if (M == 2 * TILE_M) {
    for (int i = 0; i < KB; ++i) {
      _tile_zero(TMM4);
      _tile_zero(TMM5);
      _tile_zero(TMM6);
      _tile_zero(TMM7);

      unpack_B<TB>(Tile0, B + PACKED_INDEX(0, i, KB, TILE_SIZE));
      _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);

      unpack_B<TB>(Tile1, B + PACKED_INDEX(1, i, KB, TILE_SIZE));
      _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);

      _tile_loadd(TMM2, A[i].qs, lda);
      _tile_loadd(TMM3, A[TILE_M * KB + i].qs, lda);

      _tile_dpbssd(TMM4, TMM2, TMM0);
      _tile_dpbssd(TMM5, TMM3, TMM0);
      _tile_dpbssd(TMM6, TMM2, TMM1);
      _tile_dpbssd(TMM7, TMM3, TMM1);

      _tile_stored(TMM4, Tile4, TILE_N * sizeof(int32_t));
      _tile_stored(TMM5, Tile5, TILE_N * sizeof(int32_t));
      _tile_stored(TMM6, Tile6, TILE_N * sizeof(int32_t));
      _tile_stored(TMM7, Tile7, TILE_N * sizeof(int32_t));

      const bool is_acc = i > 0;
      acc_C<TA, TB>(C,                         ldc, Tile4, &A[i],               KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), TILE_M, is_acc);
      acc_C<TA, TB>(C + TILE_M * ldc,          ldc, Tile5, &A[TILE_M * KB + i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), TILE_M, is_acc);
      acc_C<TA, TB>(C + TILE_N,                ldc, Tile6, &A[i],               KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), TILE_M, is_acc);
      acc_C<TA, TB>(C + TILE_M * ldc + TILE_N, ldc, Tile7, &A[TILE_M * KB + i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), TILE_M, is_acc);
    }
  } else {
    for (int i = 0; i < KB; ++i) {
      _tile_zero(TMM4);
      _tile_zero(TMM6);
      if (m1 != 0) {
        _tile_zero(TMM5);
        _tile_zero(TMM7);
      }

      unpack_B<TB>(Tile0, B + PACKED_INDEX(0, i, KB, TILE_SIZE));
      _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);

      unpack_B<TB>(Tile1, B + PACKED_INDEX(1, i, KB, TILE_SIZE));
      _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);

      if (m0 == TILE_M) {
        _tile_loadd(TMM2, A[i].qs, lda);
      } else {
        unpack_A(Tile23, &A[i], KB, m0);
        _tile_loadd(TMM2, Tile23, TILE_K);
      }

      _tile_dpbssd(TMM4, TMM2, TMM0);
      _tile_dpbssd(TMM6, TMM2, TMM1);

      _tile_stored(TMM4, Tile4, TILE_N * sizeof(int32_t));
      _tile_stored(TMM6, Tile6, TILE_N * sizeof(int32_t));

      const bool is_acc = i > 0;
      acc_C<TA, TB>(C,          ldc, Tile4, &A[i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m0, is_acc);
      acc_C<TA, TB>(C + TILE_N, ldc, Tile6, &A[i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m0, is_acc);
      if (m1 != 0) {
        unpack_A(Tile23, &A[TILE_M * KB + i], KB, m1);
        _tile_loadd(TMM3, Tile23, TILE_K);

        _tile_dpbssd(TMM5, TMM3, TMM0);
        _tile_dpbssd(TMM7, TMM3, TMM1);
        _tile_stored(TMM5, Tile5, TILE_N * sizeof(int32_t));
        _tile_stored(TMM7, Tile7, TILE_N * sizeof(int32_t));
        acc_C<TA, TB>(C + TILE_M * ldc,          ldc, Tile5, &A[TILE_M * KB + i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m1, is_acc);
        acc_C<TA, TB>(C + TILE_M * ldc + TILE_N, ldc, Tile7, &A[TILE_M * KB + i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m1, is_acc);
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
      (type == GGML_TYPE_Q4_1);

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

  const int ith = params->ith;
  const int nth = params->nth;

  const int M = dst->ne[1];
  const int N = dst->ne[0];
  const int K = src0->ne[0];
  const int ldc = dst->nb[1] / dst->nb[0];
  printf("\n@@@@@@ M = %d, N = %d, K = %d\n", M, N, K);

  const enum ggml_type TYPE = src0->type;

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
      char * packed_A_data = static_cast<char *>(params->wdata);
      for (int m = 0; m < M; ++m) {
        from_float<vec_dot_type>(A_data + m * K, packed_A_data + m * row_size_A, K);
      }

      // pack mat B to vnni format
      GGML_ASSERT(TILE_K == blck_size);
      const size_t row_size_B = K / blck_size * sizeof(type);
      src0->extra = aligned_alloc(64, N * row_size_B);
      convert_B_packed_format<type, blck_size>((void *)src0->extra, (const type *)src0->data, N, K);

    });
    return;
  }

  if (params->type == GGML_TASK_TYPE_FINALIZE) {
    return;
  }

  // handle 4 tiles at a tile
  constexpr int BLOCK_M = TILE_M * 2;
  constexpr int BLOCK_N = TILE_N * 2;
  const int MB = div_up(M, BLOCK_M);
  const int NB = div_up(N, BLOCK_N);

  int begin, end;
  balance211(MB * NB, nth, ith, begin, end);
  printf("@@@ MB = %d, NB = %d, begin = %d, end = %d\n", MB, NB, begin, end);
  GGML_ASSERT(src0->extra != nullptr);

  GGML_DISPATCH_QTYPES(TYPE, [&] {
    const int KB = K / blck_size;
    const int TILE_SIZE = TILE_N * sizeof(type);
    const size_t row_size_A = KB * sizeof(vec_dot_type);

    for (int i = begin; i < end; ++i) {
      int mb = i / NB;
      int nb = i % NB;

      int mb_start = mb * BLOCK_M;
      int mb_size = std::min(BLOCK_M, M - mb_start);
      int nb_start = nb * BLOCK_N;
      int nb_size = BLOCK_N;

      tiny_gemm_kernel<vec_dot_type, type, float, blck_size>(
          mb_size, nb_size, KB,
          (const char *)wdata + mb_start * row_size_A,
          (const char *)src0->extra + PACKED_INDEX(nb * 2, 0, KB, TILE_SIZE),
          (float *) dst->data + mb_start * N + nb_start, ldc);
    }
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
