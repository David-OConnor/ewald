// c/spme_vkfft.h
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VkContext VkContext;
typedef struct VkFftPlan VkFftPlan;

void* vk_make_context_default(void); // returns VkContext*
void  vk_destroy_context(void* ctx);

void* vkfft_make_plan_r2c_c2r_many(void* ctx, int32_t nx, int32_t ny, int32_t nz); // returns VkFftPlan*
void  vkfft_destroy_plan(void* plan);

void* vk_alloc_and_upload(void* ctx, const void* host_src, uint64_t nbytes); // returns VkBuffer wrapper*
void* vk_alloc_zeroed(void* ctx, uint64_t nbytes);
void  vk_download(void* ctx, void* dev_buf, void* host_dst, uint64_t nbytes);
void  vk_free(void* ctx, void* dev_buf);

void  vkfft_exec_forward_r2c(void* plan, void* real_in_dev, void* complex_out_dev);
void  vkfft_exec_inverse_c2r(void* plan, void* complex_in_dev, void* real_out_dev);

// Compute shaders for SPME k-space
void  vk_spme_apply_ghat_and_grad(
    void* ctx,
    void* rho_k,
    void* exk, void* eyk, void* ezk,
    void* kx, void* ky, void* kz,
    void* bx, void* by, void* bz,
    int32_t nx, int32_t ny, int32_t nz, float vol, float alpha);

void  vk_spme_scale_real_fields(void* ex, void* ey, void* ez, int32_t nx, int32_t ny, int32_t nz);

double vk_spme_energy_half_spectrum_sum(
    void* ctx,
    void* rho_k,
    void* kx, void* ky, void* kz,
    void* bx, void* by, void* bz,
    int32_t nx, int32_t ny, int32_t nz, float vol, float alpha);

#ifdef __cplusplus
}
#endif
