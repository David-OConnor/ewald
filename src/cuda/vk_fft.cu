// vk_fft.c
#include <stdlib.h>
#include <string.h>
#include <cuda.h>      // Driver API

#define VKFFT_BACKEND 1  // CUDA // todo: Probably not required, as set in build system.
#include "vkFFT.h"     // third-party library header (VkFFTApplication, etc.)
#include "vk_fft.h"    // your FFI header (prototypes above)


typedef struct {
    CUdevice  dev;
    CUcontext ctx;
    CUstream  stream;
    int owns_stream; // 0 = adopted, 1 = created
} VkContext;

typedef struct {
    VkFFTApplication app_r2c;
    VkFFTApplication app_c2r;
    VkFFTConfiguration cfg_r2c;
    VkFFTConfiguration cfg_c2r;
    uint64_t Nx, Ny, Nz;
} VkFftPlan;


void* vk_make_context_from_stream(void* cu_stream_void) {
    VkContext* c = (VkContext*)calloc(1, sizeof(VkContext));
    c->stream = (CUstream)cu_stream_void;
    c->owns_stream = 0;

    cuInit(0);
    CUcontext cur = NULL;
    cuCtxGetCurrent(&cur);
    if (cur == NULL) {
        CUdevice dev0; cuDeviceGet(&dev0, 0);
        cuDevicePrimaryCtxRetain(&cur, dev0);
        cuCtxSetCurrent(cur);
    }
    c->ctx = cur;
    cuCtxGetDevice(&c->dev);
    return c;
}

void* vk_make_context_default(void) {
    VkContext* c = (VkContext*)calloc(1, sizeof(VkContext));
    cuInit(0);
    cuDeviceGet(&c->dev, 0);
    CUcontext primary = NULL;
    cuDevicePrimaryCtxRetain(&primary, c->dev);
    cuCtxSetCurrent(primary);
    c->ctx = primary;
    cuStreamCreate(&c->stream, CU_STREAM_DEFAULT);
    c->owns_stream = 1;
    return c;
}

void vk_destroy_context(void* ctx_) {
    VkContext* c = (VkContext*)ctx_;
    if (!c) return;
    if (c->owns_stream) cuStreamDestroy(c->stream);
    if (c->owns_stream) cuDevicePrimaryCtxRelease(c->dev);
    free(c);
}

// Set up a plan for a 3D FFT using real-to-complex, and complex-to-real transforms.
void* make_plan(void* ctx_, int32_t nx, int32_t ny, int32_t nz) {
 VkContext* c = (VkContext*)ctx_;
    VkFftPlan* p = (VkFftPlan*)calloc(1, sizeof(VkFftPlan));
    p->Nx = (uint64_t)nx;
    p->Ny = (uint64_t)ny;
    p->Nz = (uint64_t)nz;

    // -------------------------------
    // Forward (R2C) configuration
    // -------------------------------
    memset(&p->cfg_r2c, 0, sizeof(p->cfg_r2c));

    p->cfg_r2c.FFTdim = 3;
    p->cfg_r2c.size[0] = p->Nx;
    p->cfg_r2c.size[1] = p->Ny;
    p->cfg_r2c.size[2] = p->Nz;

    p->cfg_r2c.device = &c->dev;
    p->cfg_r2c.stream = &c->stream;
    p->cfg_r2c.num_streams = 1;

    p->cfg_r2c.performR2C = 1;
    p->cfg_r2c.normalize  = 0;
    p->cfg_r2c.isInputFormatted  = 0;
    p->cfg_r2c.isOutputFormatted = 0;
    p->cfg_r2c.numberBatches = 1;
    p->cfg_r2c.printMemoryLayout = 1;

    // -------------------------------
    // Inverse (C2R) configuration
    // -------------------------------
    memset(&p->cfg_c2r, 0, sizeof(p->cfg_c2r));
    p->cfg_c2r.FFTdim = 4;
    p->cfg_c2r.size[0] = 3; // batch axis: Ex,Ey,Ez
    p->cfg_c2r.size[1] = p->Nx;
    p->cfg_c2r.size[2] = p->Ny;
    p->cfg_c2r.size[3] = p->Nz;

    p->cfg_c2r.omitDimension[0] = 1; // do not FFT over batch
    p->cfg_c2r.performR2C = 0;
    p->cfg_c2r.normalize = 0; // cuFFT inverse is unnormalized
    p->cfg_c2r.device = &c->dev;
    p->cfg_c2r.stream = &c->stream;
    p->cfg_c2r.num_streams = 1;
    p->cfg_c2r.numberBatches = 3;

    // -------------------------------
    // Initialize VkFFT plans
    // -------------------------------
    if (initializeVkFFT(&p->app_r2c, p->cfg_r2c) != VKFFT_SUCCESS) {
        free(p);
        return NULL;
    }

    if (initializeVkFFT(&p->app_c2r, p->cfg_c2r) != VKFFT_SUCCESS) {
        deleteVkFFT(&p->app_r2c);
        free(p);
        return NULL;
    }

    return p;
}

void destroy_plan(void* plan_) {
    VkFftPlan* p = (VkFftPlan*)plan_;
    if (!p) return;
    deleteVkFFT(&p->app_r2c);
    deleteVkFFT(&p->app_c2r);
    free(p);
}

void exec_forward(void* plan_, void* real_in, void* complex_out) {
    VkFftPlan* p = (VkFftPlan*)plan_;

    CUdeviceptr in  = (CUdeviceptr)real_in;
    CUdeviceptr out = (CUdeviceptr)complex_out;

    VkFFTLaunchParams lp; memset(&lp, 0, sizeof(lp));

    lp.buffer       = (void**)&in;
    lp.outputBuffer = (void**)&out;

    VkFFTAppend(&p->app_r2c, -1, &lp); // forward
}

void exec_inverse(void* plan_, void* complex_in, void* real_out) {
    VkFftPlan* p = (VkFftPlan*)plan_;

    CUdeviceptr in  = (CUdeviceptr)complex_in;
    CUdeviceptr out = (CUdeviceptr)real_out;

    VkFFTLaunchParams lp; memset(&lp, 0, sizeof(lp));

    lp.buffer = (void**)&in;
    lp.outputBuffer = (void**)&out;

    // 1 here sets the FFT to inverse
    VkFFTAppend(&p->app_c2r, 1, &lp); // inverse
}
