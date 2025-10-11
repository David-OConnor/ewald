#include <stdlib.h>
#include <string.h>
#include <cuda.h>            // Driver API only
#include "vkFFT.h"

// ---- tiny CUDA ctx/stream + plan (Driver API only) ----
typedef struct {
    CUdevice  dev;
    CUcontext ctx;
    CUstream  stream;
} VkContext;

typedef struct {
    VkFFTApplication app_r2c;
    VkFFTApplication app_c2r;
    VkFFTConfiguration cfg_r2c;
    VkFFTConfiguration cfg_c2r;
    uint64_t Nx, Ny, Nz;
} VkFftPlan;

typedef struct { CUdeviceptr ptr; size_t size; } VkBuf;

// ---- utils ----
static VkContext* make_default_ctx(void) {
    VkContext* c = (VkContext*)calloc(1, sizeof(VkContext));
    cuInit(0);
    cuDeviceGet(&c->dev, 0);

    // Use primary context to avoid cuCtxCreate_v4 signature churn.
    CUcontext primary = NULL;
    cuDevicePrimaryCtxRetain(&primary, c->dev);
    cuCtxSetCurrent(primary);
    c->ctx = primary;

    cuStreamCreate(&c->stream, CU_STREAM_DEFAULT);
    return c;
}
static void destroy_ctx(VkContext* c) {
    if (!c) return;
    cuStreamDestroy(c->stream);
    // Release the primary context we retained.
    cuDevicePrimaryCtxRelease(c->dev);
    free(c);
}
static VkBuf* alloc_host_visible(VkContext* c, size_t nbytes, int zero) {
    (void)c;
    VkBuf* b = (VkBuf*)calloc(1, sizeof(VkBuf));
    cuMemAlloc(&b->ptr, nbytes);
    b->size = nbytes;
    if (zero && nbytes) cuMemsetD8(b->ptr, 0, nbytes);
    return b;
}
static void free_buf(VkContext* c, VkBuf* b) {
    (void)c; if (!b) return; if (b->ptr) cuMemFree(b->ptr); free(b);
}
static void upload(VkContext* c, VkBuf* b, const void* src, size_t n) {
    (void)c; if (n) cuMemcpyHtoD(b->ptr, src, n);
}
static void download(VkContext* c, VkBuf* b, void* dst, size_t n) {
    (void)c; if (n) cuMemcpyDtoH(dst, b->ptr, n);
}

// ---- FFI exports ----
void* vk_make_context_default(void) { return make_default_ctx(); }
void  vk_destroy_context(void* ctx) { destroy_ctx((VkContext*)ctx); }

void* vk_alloc_and_upload(void* ctx, const void* host_src, uint64_t nbytes) {
    VkContext* c = (VkContext*)ctx;
    VkBuf* b = alloc_host_visible(c, (size_t)nbytes, 0);
    upload(c, b, host_src, (size_t)nbytes);
    return b;
}
void* vk_alloc_zeroed(void* ctx, uint64_t nbytes) {
    VkContext* c = (VkContext*)ctx;
    return alloc_host_visible(c, (size_t)nbytes, 1);
}
void  vk_download(void* ctx, void* dev_buf, void* host_dst, uint64_t nbytes) {
    VkContext* c = (VkContext*)ctx;
    download(c, (VkBuf*)dev_buf, host_dst, (size_t)nbytes);
}
void  vk_free(void* ctx, void* dev_buf) {
    VkContext* c = (VkContext*)ctx;
    free_buf(c, (VkBuf*)dev_buf);
}

void* vkfft_make_plan_r2c_c2r_many(void* ctx, int32_t nx, int32_t ny, int32_t nz) {
    VkContext* c = (VkContext*)ctx;
    VkFftPlan* p = (VkFftPlan*)calloc(1, sizeof(VkFftPlan));
    p->Nx = (uint64_t)nx; p->Ny = (uint64_t)ny; p->Nz = (uint64_t)nz;

    memset(&p->cfg_r2c, 0, sizeof(p->cfg_r2c));
    p->cfg_r2c.FFTdim = 3;
    p->cfg_r2c.size[0] = p->Nx; p->cfg_r2c.size[1] = p->Ny; p->cfg_r2c.size[2] = p->Nz;
    p->cfg_r2c.device      = &c->dev;
    p->cfg_r2c.stream      = &c->stream;
    p->cfg_r2c.num_streams = 1;
    p->cfg_r2c.performR2C  = 1;

    memset(&p->cfg_c2r, 0, sizeof(p->cfg_c2r));
    p->cfg_c2r.FFTdim = 3;
    p->cfg_c2r.size[0] = p->Nx; p->cfg_c2r.size[1] = p->Ny; p->cfg_c2r.size[2] = p->Nz;
    p->cfg_c2r.device      = &c->dev;
    p->cfg_c2r.stream      = &c->stream;
    p->cfg_c2r.num_streams = 1;
    p->cfg_c2r.performR2C  = 0;

    if (initializeVkFFT(&p->app_r2c, p->cfg_r2c) != VKFFT_SUCCESS) { free(p); return NULL; }
    if (initializeVkFFT(&p->app_c2r, p->cfg_c2r) != VKFFT_SUCCESS) { deleteVkFFT(&p->app_r2c); free(p); return NULL; }
    return p;
}

void  vkfft_destroy_plan(void* plan_) {
    VkFftPlan* p = (VkFftPlan*)plan_;
    if (!p) return;
    deleteVkFFT(&p->app_r2c);
    deleteVkFFT(&p->app_c2r);
    free(p);
}

void  vkfft_exec_forward_r2c(void* plan_, void* real_in_dev, void* complex_out_dev) {
    VkFftPlan* p = (VkFftPlan*)plan_;
    VkBuf* rin = (VkBuf*)real_in_dev;
    VkBuf* kout = (VkBuf*)complex_out_dev;

    VkFFTLaunchParams lp;
    memset(&lp, 0, sizeof(lp));
    lp.buffer       = (void**)&rin->ptr;   // sizes not required
    lp.outputBuffer = (void**)&kout->ptr;

    VkFFTAppend(&p->app_r2c, -1, &lp); // forward
}

void  vkfft_exec_inverse_c2r(void* plan_, void* complex_in_dev, void* real_out_dev) {
    VkFftPlan* p = (VkFftPlan*)plan_;
    VkBuf* kin = (VkBuf*)complex_in_dev;
    VkBuf* rout = (VkBuf*)real_out_dev;

    VkFFTLaunchParams lp;
    memset(&lp, 0, sizeof(lp));
    lp.buffer       = (void**)&kin->ptr;
    lp.outputBuffer = (void**)&rout->ptr;

    VkFFTAppend(&p->app_c2r, 1, &lp); // inverse
}
