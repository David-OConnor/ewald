// Maybe this is required to use the CUDA backend of vkfft?

// spme_vkfft_cuda.c (compiled with -DVKFFT_BACKEND=2 and linked with CUDA)
#include "vkFFT.h"
#include <cuda.h>     // or cuda_runtime_api.h if you prefer the runtime API

typedef struct {
  CUcontext cu_ctx;
  CUdevice  cu_dev;
  CUstream  cu_stream;
} CudaCtx;

typedef struct {
  VkFFTApplication app_r2c;
  VkFFTApplication app_c2r;
  VkFFTConfiguration cfg_r2c;
  VkFFTConfiguration cfg_c2r;
} CudaPlan;

void* vk_make_context_default(void) {
  CudaCtx* c = calloc(1, sizeof(CudaCtx));
  cuInit(0);
  cuDeviceGet(&c->cu_dev, 0);
  cuCtxCreate(&c->cu_ctx, 0, c->cu_dev);
  cuStreamCreate(&c->cu_stream, CU_STREAM_DEFAULT);
  return c;
}

void  vk_destroy_context(void* ctx_) {
  CudaCtx* c = (CudaCtx*)ctx_;
  if (!c) return;
  cuStreamDestroy(c->cu_stream);
  cuCtxDestroy(c->cu_ctx);
  free(c);
}

void* vkfft_make_plan_r2c_c2r_many(void* ctx_, int nx, int ny, int nz) {
  CudaCtx* c = (CudaCtx*)ctx_;
  CudaPlan* p = calloc(1, sizeof(CudaPlan));

  // R2C
  memset(&p->cfg_r2c, 0, sizeof(p->cfg_r2c));
  p->cfg_r2c.api = VKFFT_BACKEND_CUDA;     // <—— important
  p->cfg_r2c.FFTdim = 3;
  p->cfg_r2c.size[0] = (uint64_t)nx;
  p->cfg_r2c.size[1] = (uint64_t)ny;
  p->cfg_r2c.size[2] = (uint64_t)nz;
  p->cfg_r2c.performR2C = 1;
  p->cfg_r2c.cuda_stream = c->cu_stream;   // VkFFT expects a stream/queue

  if (initializeVkFFT(&p->app_r2c, &p->cfg_r2c) != VKFFT_SUCCESS) { free(p); return NULL; }

  // C2R
  memset(&p->cfg_c2r, 0, sizeof(p->cfg_c2r));
  p->cfg_c2r.api = VKFFT_BACKEND_CUDA;
  p->cfg_c2r.FFTdim = 3;
  p->cfg_c2r.size[0] = (uint64_t)nx;
  p->cfg_c2r.size[1] = (uint64_t)ny;
  p->cfg_c2r.size[2] = (uint64_t)nz;
  p->cfg_c2r.performR2C = 0;
  p->cfg_c2r.cuda_stream = c->cu_stream;

  if (initializeVkFFT(&p->app_c2r, &p->cfg_c2r) != VKFFT_SUCCESS) {
    deleteVkFFT(&p->app_r2c); free(p); return NULL;
  }
  return p;
}

void vkfft_destroy_plan(void* plan_) {
  CudaPlan* p = (CudaPlan*)plan_;
  if (!p) return;
  deleteVkFFT(&p->app_r2c);
  deleteVkFFT(&p->app_c2r);
  free(p);
}

void vkfft_exec_forward_r2c(void* plan_, void* real_in_dev, void* complex_out_dev) {
  CudaPlan* p = (CudaPlan*)plan_;
  VkFFTLaunchParams lp = {0};
  lp.buffer = &real_in_dev;     // raw CUdeviceptr casted to void*
  lp.outputBuffer = &complex_out_dev;
  performVulkanFFT(&p->app_r2c, &lp, -1, 1);
}

void vkfft_exec_inverse_c2r(void* plan_, void* complex_in_dev, void* real_out_dev) {
  CudaPlan* p = (CudaPlan*)plan_;
  VkFFTLaunchParams lp = {0};
  lp.buffer = &complex_in_dev;
  lp.outputBuffer = &real_out_dev;
  performVulkanFFT(&p->app_c2r, &lp, 1, 1);
}


void* vk_alloc_and_upload(void* ctx_, const void* src, uint64_t nbytes) {
  (void)ctx_;
  CUdeviceptr d; cuMemAlloc(&d, nbytes);
  cuMemcpyHtoD(d, src, nbytes);
  return (void*)(uintptr_t)d;
}
void* vk_alloc_zeroed(void* ctx_, uint64_t nbytes) {
  (void)ctx_;
  CUdeviceptr d; cuMemAlloc(&d, nbytes);
  cuMemsetD8(d, 0, nbytes);
  return (void*)(uintptr_t)d;
}
void  vk_download(void* ctx_, void* dev, void* dst, uint64_t nbytes) {
  (void)ctx_;
  CUdeviceptr d = (CUdeviceptr)(uintptr_t)dev;
  cuMemcpyDtoH(dst, d, nbytes);
}
void  vk_free(void* ctx_, void* dev) {
  (void)ctx_;
  CUdeviceptr d = (CUdeviceptr)(uintptr_t)dev;
  cuMemFree(d);
}