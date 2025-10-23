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

    // 1) ASSIGN THESE FIRST
    p->Nx = (uint64_t)nx;
    p->Ny = (uint64_t)ny;
    p->Nz = (uint64_t)nz;

    uint64_t Nz  = p->Nz;
    uint64_t Ny  = p->Ny;
    uint64_t Nx  = p->Nx;
    uint64_t nzc = Nz/2 + 1;

    size_t s_r = sizeof(float);      // real
    size_t s_c = sizeof(float2);     // complex (interleaved)

    // -------------------------------
    // Forward (R2C) configuration
    // -------------------------------
    memset(&p->cfg_r2c, 0, sizeof(p->cfg_r2c));
    p->cfg_r2c.FFTdim = 3;

    // 2) TELL VkFFT THE LOGICAL SIZES (Z-fast: [Nz, Ny, Nx])
    p->cfg_r2c.size[0] = Nz;   // fast (W)  <- Z
    p->cfg_r2c.size[1] = Ny;   //           <- Y
    p->cfg_r2c.size[2] = Nx;   //           <- X

    // Real input strides (BYTES), Z-fast: [1, Nz, Ny*Nz]
    p->cfg_r2c.inputBufferStride[0] = (Ny * Nz) * s_r;  // step 1 in X = jump an entire YZ plane
    p->cfg_r2c.inputBufferStride[1] = (Nz)      * s_r;  // step 1 in Y = jump one Z-line
    p->cfg_r2c.inputBufferStride[2] = (1)       * s_r;  // step 1 in Z = move to next element

    // Complex output [Nx, Ny, nzc] with Z' contiguous:
    p->cfg_r2c.outputBufferStride[0] = (Ny * nzc) * s_c; // step 1 in X
    p->cfg_r2c.outputBufferStride[1] = (nzc)      * s_c; // step 1 in Y
    p->cfg_r2c.outputBufferStride[2] = (1)        * s_c; // step 1 in Z'

    p->cfg_r2c.disableMergeSequencesR2C = 1;

    p->cfg_r2c.performR2C = 1;
    p->cfg_r2c.normalize  = 0;
    p->cfg_r2c.numberBatches = 1;

    p->cfg_r2c.device = &c->dev;
    p->cfg_r2c.stream = &c->stream;
    p->cfg_r2c.num_streams = 1;

    // -------------------------------
    // Inverse (C2R) configuration
    // -------------------------------
    memset(&p->cfg_c2r, 0, sizeof(p->cfg_c2r));
    p->cfg_c2r.FFTdim = 3;

    // 2) LOGICAL SIZES AGAIN (Z-fast)
    p->cfg_c2r.size[0] = Nz;   // fast (W)  <- Z
    p->cfg_c2r.size[1] = Ny;   //           <- Y
    p->cfg_c2r.size[2] = Nx;   //           <- X

    p->cfg_c2r.inputBufferStride[0] = 1      * s_c;          // Z' step
    p->cfg_c2r.inputBufferStride[1] = nzc    * s_c;          // Y step
    p->cfg_c2r.inputBufferStride[2] = Ny*nzc * s_c;          // X step

    // Real output full strides (BYTES), Z-fast
    p->cfg_c2r.outputBufferStride[0] = 1      * s_r;         // Z step
    p->cfg_c2r.outputBufferStride[1] = Nz     * s_r;         // Y step
    p->cfg_c2r.outputBufferStride[2] = Ny*Nz  * s_r;         // X step

    p->cfg_c2r.performR2C = 0;    // C2R
    p->cfg_c2r.normalize  = 0;    // match cuFFT (unnormalized)
    p->cfg_c2r.numberBatches = 1; // one field per call

    p->cfg_c2r.device = &c->dev;
    p->cfg_c2r.stream = &c->stream;
    p->cfg_c2r.num_streams = 1;

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
