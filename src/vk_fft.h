// vk_fft.h  (your FFI header)
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Context lifecycle (adopt or create a CUDA stream)
void* vk_make_context_default(void);
void* vk_make_context_from_stream(void* cu_stream); // returns VkContext*
void  vk_destroy_context(void* ctx);

// VkFFT plan lifecycle
void* vkfft_make_plan_r2c_c2r_many(void* ctx, int32_t nx, int32_t ny, int32_t nz);
void  vkfft_destroy_plan(void* plan);

// FFT execs (raw device pointers)
void  vkfft_exec_forward_r2c(void* plan, void* real_in_dev, void* complex_out_dev);
void  vkfft_exec_inverse_c2r(void* plan, void* complex_in_dev, void* real_out_dev);

// K-space pipeline (raw device pointers)
// Keep ctx: we may need its CUstream/device internally.
void  vk_apply_ghat_and_grad(
    void* ctx,
    const void* rho_k,
    void* exk, void* eyk, void* ezk,
    const void* kx, const void* ky, const void* kz,
    const void* bx, const void* by, const void* bz,
    int32_t nx, int32_t ny, int32_t nz, float vol, float alpha);

// Energy reduction on half-spectrum
double vk_energy_half_spectrum_sum(
    void* ctx,
    const void* rho_k,
    const void* kx, const void* ky, const void* kz,
    const void* bx, const void* by, const void* bz,
    int32_t nx, int32_t ny, int32_t nz, float vol, float alpha);


#ifdef __cplusplus
}
#endif
