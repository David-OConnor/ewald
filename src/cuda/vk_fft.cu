// vk_fft.c
#include <stdlib.h>
#include <string.h>
#include <cuda.h> // Driver API

#define VKFFT_BACKEND 1  // CUDA // todo: Probably not required, as set in build system.
#include "vkFFT.h"  // VKFFT
#include "vk_fft.h"  // Our header; just the function signatures from this module.


typedef struct VkFftPlan {
    VkFFTApplication app;
    VkFFTConfiguration cfg;
    uint64_t Nx;
    uint64_t Ny;
    uint64_t Nz;
    CUdevice dev;
    CUcontext ctx;
    CUstream stream;
} VkFftPlan;


void* make_plan(int32_t nx, int32_t ny, int32_t nz, void* stream) {
    VkFftPlan* plan = (VkFftPlan*)calloc(1, sizeof(VkFftPlan));
    if (!plan) {
        return NULL;
    }

    // CUDA device configuration -------------
    if (cuCtxGetCurrent(&plan->ctx) != CUDA_SUCCESS) {
        free(plan);
        return NULL;
    }
    if (cuCtxGetDevice(&plan->dev) != CUDA_SUCCESS) {
        free(plan);
        return NULL;
    }

    plan->stream = (CUstream)stream;

    plan->Nx = (uint64_t)nx;
    plan->Ny = (uint64_t)ny;
    plan->Nz = (uint64_t)nz;

    VkFFTConfiguration* cfg = &plan->cfg;
    memset(cfg, 0, sizeof(*cfg));

    cfg->device = &plan->dev;
    cfg->stream = &plan->stream;

    cfg->num_streams = 1;
    // FFT configuration ----------

    // Configure for Z fast (contiguous), X slow (strided)
    cfg->FFTdim  = 3;
    cfg->size[0] = (uint64_t)nz;
    cfg->size[1] = (uint64_t)ny;
    cfg->size[2] = (uint64_t)nx;

    cfg->isInputFormatted  = 1;
    cfg->isOutputFormatted = 1;

//     cfg->inputBufferStride[0] = cfg->size[0];
//     cfg->inputBufferStride[1] = cfg->inputBufferStride[0] * cfg->size[1];
//     cfg->inputBufferStride[2] = cfg->inputBufferStride[1] * cfg->size[2];

    cfg->inputBufferStride[0] = 1;                          // z step
    cfg->inputBufferStride[1] = cfg->size[0];               // y step = nz
    cfg->inputBufferStride[2] = cfg->size[0] * cfg->size[1]; // x step = nz*ny

    cfg->bufferStride[0] = (uint64_t)(cfg->size[0] / 2) + 1;
    cfg->bufferStride[1] = cfg->bufferStride[0] * cfg->size[1];
    cfg->bufferStride[2] = cfg->bufferStride[1] * cfg->size[2];

    cfg->performR2C = 1;
    cfg->normalize = 0;
    cfg->numberBatches = 1;

    VkFFTResult res = initializeVkFFT(&plan->app, *cfg);
    if (res != VKFFT_SUCCESS) {
        free(plan);
        return NULL;
    }

    return plan;
}

void destroy_plan(void* plan_) {
    VkFftPlan* plan = (VkFftPlan*)plan_;
    if (!plan) {
        return;
    }

    deleteVkFFT(&plan->app);
    free(plan);
}

void exec_forward(void* plan_, void* real_in, void* complex_out) {
    VkFftPlan* plan = (VkFftPlan*)plan_;

    CUdeviceptr in = (CUdeviceptr)real_in;
    CUdeviceptr out = (CUdeviceptr)complex_out;

    VkFFTLaunchParams lp;
    memset(&lp, 0, sizeof(lp));

    lp.buffer = (void**)&in;
    lp.outputBuffer = (void**)&out;

    VkFFTResult res = VkFFTAppend(&plan->app, -1, &lp);
}

void exec_inverse(void* plan_, void* complex_in, void* real_out) {
    VkFftPlan* plan = (VkFftPlan*)plan_;

    CUdeviceptr in = (CUdeviceptr)complex_in;
    CUdeviceptr out = (CUdeviceptr)real_out;

    VkFFTLaunchParams lp;
    memset(&lp, 0, sizeof(lp));

    lp.buffer = (void**)&in;
    lp.outputBuffer = (void**)&out;

    VkFFTResult res = VkFFTAppend(&plan->app, 1, &lp);
}