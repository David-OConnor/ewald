// Perform FFTs using cuFFT. [docs](https://docs.nvidia.com/cuda/cufft/)

// We use the scatter functionality here for both cuFFT and
// VkFFT. The rest of this is for cuFFT only.

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cstdio>
#include <cstdint>


// A minimal CUFFT error checker.
#ifndef CUFFT_CHECK
#define CUFFT_CHECK(call)                                                   \
  do {                                                                      \
    cufftResult _e = (call);                                                \
    if (_e != CUFFT_SUCCESS) {                                              \
      printf("CUFFT error %d at %s:%d\n", (int)_e, __FILE__, __LINE__);     \
    }                                                                       \
  } while (0)
#endif

struct PlanWrap {
    cufftHandle plan_r2c;
    cufftHandle plan_c2r;
    cudaStream_t stream;
};

// https://docs.nvidia.com/cuda/cufft/#cufftplan3d
extern "C"
void* make_plan(int nx, int ny, int nz, void* cu_stream) {
    auto* w = new PlanWrap();

    w->stream = reinterpret_cast<cudaStream_t>(cu_stream);

    // With Plan3D, Z is the fastest-changing dimension (contiguous); x is the slowest.
    CUFFT_CHECK(cufftPlan3d(&w->plan_r2c, nx, ny, nz, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan3d(&w->plan_c2r, nx, ny, nz, CUFFT_C2R));

    CUFFT_CHECK(cufftSetStream(w->plan_r2c, w->stream));
    CUFFT_CHECK(cufftSetStream(w->plan_c2r, w->stream));

    return w;
}


extern "C"
void destroy_plan(void* plan) {
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    if (!w) return;

    cufftDestroy(w->plan_r2c);
    cufftDestroy(w->plan_c2r);

    delete w;
}

// https://docs.nvidia.com/cuda/cufft/#cufftexecr2c-and-cufftexecd2z
// Performs a forward real-to-copmlex FFT of rho. Note: This is more efficient
// than complex-to-complex.
extern "C"
void exec_forward(void* plan, float* rho_real, cufftComplex* rho) {
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    if (!w) return;

    CUFFT_CHECK(cufftExecR2C(w->plan_r2c, rho_real, rho));
}

extern "C"
void exec_inverse(
    void* plan,
    cufftComplex* exk,
    cufftComplex* eyk,
    cufftComplex* ezk,
    float* ex,
    float* ey,
    float* ez
){
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    if (!w) return;

    CUFFT_CHECK(cufftExecC2R(w->plan_c2r, exk, ex));
    CUFFT_CHECK(cufftExecC2R(w->plan_c2r, eyk, ey));
    CUFFT_CHECK(cufftExecC2R(w->plan_c2r, ezk, ez));
}