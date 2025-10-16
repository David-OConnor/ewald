// Code here is only used by the cuFFT pipeline.

// We use the scatter functionality here for both cuFFT and
// VkFFT. The rest of this is for cuFFT only.

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cstdio>
#include <cstdint>

#include "shared.cu"


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
    cufftHandle plan_c2r_many; // batch=3 for exk, eyk, ezk
    size_t n_real;  // nx*ny*nz
    size_t n_cmplx; // nx*ny*(nz/2+1) if using z as transform axis
    int nx, ny, nz;
    cudaStream_t stream;
};

extern "C"
void* make_plan_r2c_c2r_many(int nx, int ny, int nz, void* cu_stream) {
    auto* w = new PlanWrap();
    w->nx = nx; w->ny = ny; w->nz = nz;
    w->n_real  = size_t(nx)*ny*nz;
    w->n_cmplx = size_t(nx)*ny*(nz/2 + 1);
    w->stream = reinterpret_cast<cudaStream_t>(cu_stream);

    CUFFT_CHECK(cufftPlan3d(&w->plan_r2c, nx, ny, nz, CUFFT_R2C));
    CUFFT_CHECK(cufftSetStream(w->plan_r2c, w->stream));

    // PlanMany for 3 fields back to real grids
    int n[3] = {nx, ny, nz};
    int inembed[3]  = {nx, ny, nz/2 + 1};
    int onembed[3]  = {nx, ny, nz};
    int istride = 1, ostride = 1;
    int idist = nx*ny*(nz/2 + 1);
    int odist = nx*ny*nz;
    int batch = 3;

    CUFFT_CHECK(cufftPlanMany(&w->plan_c2r_many, 3, n,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2R, batch));
    CUFFT_CHECK(cufftSetStream(w->plan_c2r_many, w->stream));
    return w;
}


extern "C"
void destroy_plan_r2c_c2r_many(void* plan) {
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    if (!w) return;
    cufftDestroy(w->plan_r2c);
    cufftDestroy(w->plan_c2r_many);
    delete w;
}

extern "C"
void exec_forward_r2c(void* plan, float* rho_real, cufftComplex* rho_k) {
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    if (!w) return;
    CUFFT_CHECK(cufftExecR2C(w->plan_r2c, rho_real, rho_k));
}

extern "C"
void exec_inverse_ExEyEz_c2r(
    void* plan,
    cufftComplex* exk, /* base of [exk|eyk|ezk] */
    cufftComplex* /*eyk*/,
    cufftComplex* /*ezk*/,
    float* ex /* base of [ex|ey|ez] */,
    float* /*ey*/,
    float* /*ez*/
){
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    if (!w) return;

    // exk must point to 3*n_cmplx contiguous cufftComplex
    // ex  must point to 3*n_real   contiguous float
    CUFFT_CHECK(cufftExecC2R(w->plan_c2r_many, exk, ex));
}

