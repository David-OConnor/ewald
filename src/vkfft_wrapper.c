#include "vkFFT/vkFFT.h"
#include <cuda.h>
#include <cuda_runtime.h>

typedef struct {
    VkFFTApplication app;
    VkFFTConfiguration config;
} VkFFTHandle;

int vkfft_init(VkFFTHandle* handle, int size, int batch) {
    memset(&handle->config, 0, sizeof(VkFFTConfiguration));
    handle->config.FFTdim = 1;
    handle->config.size[0] = size;
    handle->config.device = 0;  // CUDA device 0
    handle->config.isInputFormatted = 1;
    handle->config.isOutputFormatted = 1;
    handle->config.numBatches = batch;
    handle->config.api = VkFFTAPI_CUDA;

    return initializeVkFFT(&handle->app, &handle->config);
}

int vkfft_forward(VkFFTHandle* handle, CUdeviceptr input, CUdeviceptr output) {
    return VkFFTAppend(&handle->app, -1, (void*)&input, (void*)&output, VKFFT_FORWARD);
}

int vkfft_inverse(VkFFTHandle* handle, CUdeviceptr input, CUdeviceptr output) {
    return VkFFTAppend(&handle->app, -1, (void*)&input, (void*)&output, VKFFT_BACKWARD);
}

void vkfft_cleanup(VkFFTHandle* handle) {
    deleteVkFFT(&handle->app);
}



// todo: Above, below, or both?

// Opaque context created once (instance/device/queue/allocator)
typedef struct VkContext { void* instance; void* device; void* queue; void* allocator; } VkContext;

void* vkfft_make_plan_r2c_c2r_many(void* ctx, int nx, int ny, int nz);
void  vkfft_destroy_plan(void* plan);

void* vk_alloc_and_upload(void* ctx, const void* host_src, uint64_t nbytes);
void* vk_alloc_zeroed(void* ctx, uint64_t nbytes);
void  vk_download(void* ctx, void* dev_buf, void* host_dst, uint64_t nbytes);
void  vk_free(void* ctx, void* dev_buf);

void  vkfft_exec_forward_r2c(void* plan, void* real_in_dev, void* complex_out_dev);
void  vkfft_exec_inverse_c2r(void* plan, void* complex_in_dev, void* real_out_dev);

void  vk_spme_apply_ghat_and_grad(void* ctx,
    void* rho_k, void* exk, void* eyk, void* ezk,
    void* kx, void* ky, void* kz,
    void* bx, void* by, void* bz,
    int nx, int ny, int nz, float vol, float alpha);

void  vk_spme_scale_real_fields(void* ex, void* ey, void* ez, int nx, int ny, int nz);

double vk_spme_energy_half_spectrum_sum(void* ctx,
    void* rho_k, void* kx, void* ky, void* kz, void* bx, void* by, void* bz,
    int nx, int ny, int nz, float vol, float alpha);
