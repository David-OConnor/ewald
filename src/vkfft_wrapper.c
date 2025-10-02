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
