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
void* make_plan(void* ctx_, int32_t nx, int32_t ny, int32_t nz, void* cu_stream) {
    VkContext* c = (VkContext*)ctx_;
    VkFftPlan* p = (VkFftPlan*)calloc(1, sizeof(VkFftPlan));

    p->stream = cu_stream;

    VkFFTConfiguration& c = p->cfg;
    // Driver/stream hooks (CUDA backend).
    c.device      = &p->cu_dev;       // CUdevice*
    c.stream      = &p->stream;       // cudaStream_t*
    c.num_streams = 1;

    // 3D transform. Map cuFFT (x,y,z) -> VkFFT (D,H,W) with W fastest.
    // So set W= nz, H= ny, D= nx to make Z the contiguous dimension like cuFFT.
    c.FFTdim   = 3;
    c.size[0]  = (uint64_t)nz;  // W (fastest)  == cuFFT's Z
    c.size[1]  = (uint64_t)ny;  // H            == cuFFT's Y
    c.size[2]  = (uint64_t)nx;  // D (slowest)  == cuFFT's X

    // Real<->Complex transforms; un-normalized inverse (matches cuFFT defaults).
    c.performR2C = 1; // enables R2C/C2R
    c.normalize  = 0; // cuFFT is un-normalized on inverse

    // Single batch, single-precision by default (matches your CUFFT_R2C/C2R FP32).
    c.numberBatches = 1;

    // Initialize once; you’ll pass buffers at launch time via VkFFTLaunchParams.
    VkFFTResult res = initializeVkFFT(&p->app, &p->cfg);
    if (res != VKFFT_SUCCESS) {
        free(p);
        return nullptr;
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
