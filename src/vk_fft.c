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
    VkFFTApplication   app_r2c;
    VkFFTApplication   app_c2r;
    VkFFTConfiguration cfg_r2c;
    VkFFTConfiguration cfg_c2r;
    uint64_t Nx, Ny, Nz;
} VkFftPlan;

#ifdef __cplusplus
extern "C" {
#endif

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

void* make_plan_r2c_c2r_many(void* ctx_, int32_t nx, int32_t ny, int32_t nz) {
    VkContext* c = (VkContext*)ctx_;
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

void destroy_plan(void* plan_) {
    VkFftPlan* p = (VkFftPlan*)plan_;
    if (!p) return;
    deleteVkFFT(&p->app_r2c);
    deleteVkFFT(&p->app_c2r);
    free(p);
}

void exec_forward_r2c(void* plan_, void* real_in_dev, void* complex_out_dev) {
    VkFftPlan* p = (VkFftPlan*)plan_;
    CUdeviceptr in  = (CUdeviceptr)real_in_dev;
    CUdeviceptr out = (CUdeviceptr)complex_out_dev;
    VkFFTLaunchParams lp; memset(&lp, 0, sizeof(lp));
    lp.buffer       = (void**)&in;
    lp.outputBuffer = (void**)&out;
    VkFFTAppend(&p->app_r2c, -1, &lp); // forward
}

void exec_inverse_c2r(void* plan_, void* complex_in_dev, void* real_out_dev) {
    VkFftPlan* p = (VkFftPlan*)plan_;
    CUdeviceptr in  = (CUdeviceptr)complex_in_dev;
    CUdeviceptr out = (CUdeviceptr)real_out_dev;
    VkFFTLaunchParams lp; memset(&lp, 0, sizeof(lp));
    lp.buffer       = (void**)&in;
    lp.outputBuffer = (void**)&out;
    VkFFTAppend(&p->app_c2r, 1, &lp); // inverse
}

// to replace the individual one?
void exec_inverse_ExEyEz_c2r(void* plan_, void* exk, void* eyk, void* ezk,
                              void* ex,  void* ey,  void* ez)
{
    VkFftPlan* p = (VkFftPlan*)plan_;

    CUdeviceptr in0 = (CUdeviceptr)exk;
    CUdeviceptr in1 = (CUdeviceptr)eyk;
    CUdeviceptr in2 = (CUdeviceptr)ezk;

    CUdeviceptr out0 = (CUdeviceptr)ex;
    CUdeviceptr out1 = (CUdeviceptr)ey;
    CUdeviceptr out2 = (CUdeviceptr)ez;

    void* in_arr[3]  = { &in0,  &in1,  &in2  };
    void* out_arr[3] = { &out0, &out1, &out2 };

    VkFFTLaunchParams lp; memset(&lp, 0, sizeof(lp));
    lp.buffer        = (void**)in_arr;
    lp.outputBuffer  = (void**)out_arr;
    lp.numberBuffers = 3;

    VkFFTAppend(&p->app_c2r, 1, &lp);
}


/* ----- NEW: define the two vk_* k-space symbols so Rust can link ----- */

void vk_apply_ghat_and_grad(
    void* ctx_,
    const void* rho_k,
    void* exk, void* eyk, void* ezk,
    const void* kx, const void* ky, const void* kz,
    const void* bx, const void* by, const void* bz,
    int32_t nx, int32_t ny, int32_t nz, float vol, float alpha)
{
    (void)vol; (void)alpha;
    VkContext* c = (VkContext*)ctx_;
    // You can call a CUDA kernel here (in a .cu TU) using c->stream,
    // or temporarily memset outputs to zero as a stub:
    cuMemsetD8((CUdeviceptr)exk, 0, (size_t)nx*ny*(nz/2+1)*2*sizeof(float));
    cuMemsetD8((CUdeviceptr)eyk, 0, (size_t)nx*ny*(nz/2+1)*2*sizeof(float));
    cuMemsetD8((CUdeviceptr)ezk, 0, (size_t)nx*ny*(nz/2+1)*2*sizeof(float));
    (void)rho_k; (void)kx; (void)ky; (void)kz; (void)bx; (void)by; (void)bz;
    (void)c;
}

double vk_energy_half_spectrum_sum(
    void* ctx_,
    const void* rho_k,
    const void* kx, const void* ky, const void* kz,
    const void* bx, const void* by, const void* bz,
    int32_t nx, int32_t ny, int32_t nz, float vol, float alpha)
{
    (void)ctx_; (void)rho_k; (void)kx; (void)ky; (void)kz; (void)bx; (void)by; (void)bz;
    (void)nx; (void)ny; (void)nz; (void)vol; (void)alpha;
    // Replace with a CUDA reduction kernel; stub returns 0 to link:
    return 0.0;
}

#ifdef __cplusplus
} // extern "C"
#endif
