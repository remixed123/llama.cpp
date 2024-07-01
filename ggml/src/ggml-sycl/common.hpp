//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_COMMON_HPP
#define GGML_SYCL_COMMON_HPP

#include <fstream>
#include <iostream>
#include <regex>

#include "dpct/helper.hpp"
#include "ggml-sycl.h"
#include "presets.hpp"

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

void* ggml_sycl_host_malloc(size_t size);
void ggml_sycl_host_free(void* ptr);

static int g_ggml_sycl_debug = 0;
#define GGML_SYCL_DEBUG(...)        \
  do {                              \
    if (g_ggml_sycl_debug)          \
      fprintf(stderr, __VA_ARGS__); \
  } while (0)

#define CHECK_TRY_ERROR(expr)                                            \
  [&]() {                                                                \
    try {                                                                \
      expr;                                                              \
      return dpct::success;                                              \
    } catch (std::exception const& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::endl;                                            \
      return dpct::default_error;                                        \
    }                                                                    \
  }()

#define __SYCL_ARCH__ DPCT_COMPATIBILITY_TEMP
#define VER_4VEC 610 // todo for hardward optimize.
#define VER_GEN9 700 // todo for hardward optimize.
#define VER_GEN12 1000000 // todo for hardward optimize.
#define VER_GEN13 (VER_GEN12 + 1030) // todo for hardward optimize.

#define GGML_SYCL_MAX_NODES 8192 // TODO: adapt to hardwares

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#if !defined(GGML_SYCL_FORCE_MMQ)
    #define SYCL_USE_XMX
#endif

// max batch size to use MMQ kernels when tensor cores are available
#define MMQ_MAX_BATCH_SIZE 32

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_SYCL_DMMV_X
#define GGML_SYCL_DMMV_X 32
#endif
#ifndef GGML_SYCL_MMV_Y
#define GGML_SYCL_MMV_Y 1
#endif

typedef sycl::queue *queue_ptr;

enum ggml_sycl_backend_gpu_mode {
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

enum ggml_sycl_backend_device_filter {
  SYCL_ALL_DEVICES = 0,
  SYCL_DEVICES_TOP_LEVEL_ZERO,
  SYCL_VISIBLE_DEVICES
};

static_assert(sizeof(sycl::half) == sizeof(ggml_fp16_t), "wrong fp16 size");

static void crash() {
  int* ptr = NULL;
  *ptr = 0;
}

[[noreturn]] static void ggml_sycl_error(
    const char* stmt,
    const char* func,
    const char* file,
    const int line,
    const char* msg) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, msg);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, line);
  GGML_ASSERT(!"SYCL error");
}

#define SYCL_RETURN_ERROR 1

#define SYCL_CHECK(err)                     \
  do {                                      \
    auto err_ = (err);                      \
    if (err_ != 0)                          \
      ggml_sycl_error(                      \
          #err,                             \
          __func__,                         \
          __FILE__,                         \
          __LINE__,                         \
          "Meet error in this line code!"); \
  } while (0)


#if DPCT_COMPAT_RT_VERSION >= 11100
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#else
#define GGML_SYCL_ASSUME(x)
#endif // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
typedef sycl::half dfloat; // dequantize float
typedef sycl::half2 dfloat2;
#else
typedef float dfloat; // dequantize float
typedef sycl::float2 dfloat2;
#endif // GGML_SYCL_F16

#define MMVQ_MAX_BATCH_SIZE  8

static const int8_t kvalues_iq4nl[16]={-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static int g_all_sycl_device_count = -1;
static bool g_ggml_backend_sycl_buffer_type_initialized = false;

static ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

static void* g_scratch_buffer = nullptr;
static size_t g_scratch_size = 0; // disabled by default
static size_t g_scratch_offset = 0;

int get_current_device_id();

[[noreturn]] static inline void bad_arch(const sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::exit(1);

  (void)bad_arch; // suppress unused function warning
}

inline dpct::err0 ggml_sycl_set_device(const int device_id) try {

  int current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d, current_device_id=%d\n", device_id, current_device_id);
  if (device_id == current_device_id) {
    return 0;
  }

  return CHECK_TRY_ERROR(dpct::select_device(device_id));

} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

class sycl_device_mgr {
  public:
    std::vector<int> device_ids;
    std::vector<sycl::device> devices;
    std::vector<int> max_compute_units;
    std::vector<int> work_group_sizes;
    sycl::queue *first_queue;
    std::vector<sycl::queue> _queues;
    std::vector<sycl::context> ctxs;
    std::string device_list = "";

    sycl_device_mgr(ggml_sycl_backend_device_filter device_filter);

    sycl::queue *_create_queue_ptr(sycl::device device); //internal API to hide dpct API.
    void create_context_for_group_gpus();
    sycl::queue *create_queue_for_device(sycl::device &device);
    sycl::queue *create_queue_for_device_id(int device_id);
    int get_device_index(int device_id);
    void create_context_for_devices();
    void init_allow_devices();
    bool is_allowed_device(int device_id);
    void detect_all_sycl_device_list();
    void detect_sycl_visible_device_list();
    void detect_sycl_gpu_list_with_max_cu();
    int get_device_count();
    bool is_ext_oneapi_device(const sycl::device &dev);
};


struct ggml_sycl_device_info {
    int device_count;
    bool oneapi_device_selector_existed = false;
    bool sycl_visible_devices_existed = false;

    struct sycl_device_info {
        int     cc;                 // compute capability
        // int     nsm;                // number of streaming multiprocessors
        // size_t  smpb;               // max. shared memory per block
        bool    vmm;                // virtual memory support
        size_t  total_vram;
    };

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<float, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    sycl_device_mgr *device_mgr = NULL;

    void print_gpu_device_list();
    int work_group_size(int device_id);
    void refresh_device();
    bool is_allowed_device(int device_id);
    const char* devices_list();
    int get_device_id(int device_index);
};

struct ggml_sycl_pool {
    virtual ~ggml_sycl_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>
struct ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() {
        return ptr;
    }

    ggml_sycl_pool_alloc() = default;
    ggml_sycl_pool_alloc(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = delete;
    ggml_sycl_pool_alloc& operator=(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc& operator=(ggml_sycl_pool_alloc &&) = delete;
};

// backend interface

struct ggml_tensor_extra_gpu {
  void* data_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
};

struct ggml_backend_sycl_context {
    int device;
    std::string name;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(struct ggml_sycl_device_info &sycl_device_info, int device_id) :
        device(device_id),
        name(GGML_SYCL_NAME + std::to_string(device)) {
            for (int i=0;i<GGML_SYCL_MAX_STREAMS; i++){
                qptrs[device_id][i] = sycl_device_info.device_mgr->create_queue_for_device_id(device_id);
            }
    }

    queue_ptr stream(int device, int stream) {
        assert(qptrs[device][stream] != nullptr);
        return qptrs[device][stream];
    }

    queue_ptr stream() {
        return stream(device, 0);
    }

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    static std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, int device);

    ggml_sycl_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        return *pools[device];
    }

    ggml_sycl_pool & pool() {
        return pool(device);
    }
};

static inline void exit_with_stack_print() {
    SYCL_CHECK(SYCL_RETURN_ERROR);
}


static inline int get_sycl_env(const char *env_name, int default_val);
static inline bool env_existed(const char *env_name);
void* ggml_sycl_host_malloc(size_t size);
void ggml_sycl_host_free(void* ptr);
static std::vector<int> get_sycl_visible_devices();
void ggml_backend_sycl_print_sycl_devices();
static ggml_sycl_device_info ggml_sycl_init();
ggml_sycl_device_info &ggml_sycl_info();

#endif // GGML_SYCL_COMMON_HPP
