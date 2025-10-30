// vk_fft.c
#include <stdlib.h>
#include <string.h>
#include <cuda.h>      // Driver API

#define VKFFT_BACKEND 1  // CUDA // todo: Probably not required, as set in build system.
#include "vkFFT.h"     // third-party library header (VkFFTApplication, etc.)
#include "vk_fft.h"    // your FFI header (prototypes above)


// typedef struct {
//     CUdevice  dev;
//     CUcontext ctx;
//     CUstream  stream;
//     int owns_stream; // 0 = adopted, 1 = created
// } VkContext;
//
// typedef struct {
//     VkFFTApplication app_r2c;
//     VkFFTApplication app_c2r;
//     VkFFTConfiguration cfg_r2c;
//     VkFFTConfiguration cfg_c2r;
//     uint64_t Nx, Ny, Nz;
// } VkFftPlan;


typedef struct VkContext {
    CUdevice  dev;
    CUcontext ctx;
    CUstream  stream;
    int       owns_stream; // 0 = adopted, 1 = we created it
} VkContext;

// Our plan: single app, single cfg, plus cached CUDA handles.
typedef struct VkFftPlan {
    VkFFTApplication  app;
    VkFFTConfiguration cfg;
    CUdevice           cu_dev;
    CUcontext          cu_ctx;
    cudaStream_t       stream;
} VkFftPlan;



void* vk_make_context_from_stream(void* cu_stream_void) {
    VkContext* c = (VkContext*)calloc(1, sizeof(VkContext));
    if (!c) return NULL;

    c->stream = (CUstream)cu_stream_void;
    c->owns_stream = 0;

    cuInit(0);

    CUcontext cur = NULL;
    cuCtxGetCurrent(&cur);
    if (cur == NULL) {
        CUdevice dev0;
        cuDeviceGet(&dev0, 0);
        cuDevicePrimaryCtxRetain(&cur, dev0);
        cuCtxSetCurrent(cur);
    }

    c->ctx = cur;
    cuCtxGetDevice(&c->dev);
    return c;
}

void* vk_make_context_default(void) {
    VkContext* c = (VkContext*)calloc(1, sizeof(VkContext));
    if (!c) return NULL;

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

void* make_plan(void* ctx_, int32_t nx, int32_t ny, int32_t nz, void* cu_stream)
{
    VkContext* g = (VkContext*)ctx_;

    VkFftPlan* p = (VkFftPlan*)calloc(1, sizeof(VkFftPlan));
    if (!p) return NULL;

    p->cu_dev  = g->dev;
    p->cu_ctx  = g->ctx;
    p->stream  = cu_stream ? (cudaStream_t)cu_stream
                           : (cudaStream_t)g->stream;

    VkFFTConfiguration* cfg = &p->cfg;
    memset(cfg, 0, sizeof(*cfg));

    cfg->device      = &p->cu_dev;
    cfg->stream      = &p->stream;
    cfg->num_streams = 1;

    cfg->FFTdim  = 3;
    cfg->size[0] = (uint64_t)nz;
    cfg->size[1] = (uint64_t)ny;
    cfg->size[2] = (uint64_t)nx;

    cfg->performR2C    = 1;
    cfg->normalize     = 0;
    cfg->numberBatches = 1;

    VkFFTResult res = initializeVkFFT(&p->app, *cfg);
    if (res != VKFFT_SUCCESS) {
        free(p);
        return NULL;
    }

    return p;
}

void destroy_plan(void* plan_) {
    VkFftPlan* p = (VkFftPlan*)plan_;
    if (!p) return;
    deleteVkFFT(&p->app);
    free(p);
}

void exec_forward(void* plan_, void* real_in, void* complex_out) {
    VkFftPlan* p = (VkFftPlan*)plan_;

    CUdeviceptr in  = (CUdeviceptr)real_in;
    CUdeviceptr out = (CUdeviceptr)complex_out;

    VkFFTLaunchParams lp;
    memset(&lp, 0, sizeof(lp));

    lp.buffer       = (void**)&in;
    lp.outputBuffer = (void**)&out;
    // no lp.stream: VkFFTLaunchParams in your headers doesn’t have it; stream is in cfg.

    VkFFTAppend(&p->app, -1, &lp); // forward
}

void exec_inverse(void* plan_, void* complex_in, void* real_out) {
    VkFftPlan* p = (VkFftPlan*)plan_;

    CUdeviceptr in  = (CUdeviceptr)complex_in;
    CUdeviceptr out = (CUdeviceptr)real_out;

    VkFFTLaunchParams lp;
    memset(&lp, 0, sizeof(lp));

    lp.buffer       = (void**)&in;
    lp.outputBuffer = (void**)&out;

    VkFFTAppend(&p->app, 1, &lp); // inverse
}