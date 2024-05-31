#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


bool ggml_amx_init(void);

bool ggml_compute_forward_mul_mat_use_amx(struct ggml_tensor * dst);

void ggml_mul_mat_amx(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
