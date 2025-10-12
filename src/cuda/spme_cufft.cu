// Code here is only used by the cuFFT pipeline.

// We use the scatter functionality here for both cuFFT and
// VkFFT. The rest of this is for cuFFT only.

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cstdio>
#include <cstdint>

#include "spme_shared.cu"


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

extern "C" __global__
void apply_ghat_and_grad(
    const cufftComplex* __restrict__ rho,
    cufftComplex* __restrict__ exk,
    cufftComplex* __restrict__ eyk,
    cufftComplex* __restrict__ ezk,
    const float* __restrict__ kx,
    const float* __restrict__ ky,
    const float* __restrict__ kz,
    const float* __restrict__ bx,
    const float* __restrict__ by,
    const float* __restrict__ bz,
    int nx, int ny, int nz, float vol, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cmplx = nx*ny*(nz/2 + 1);
    if (idx >= n_cmplx) return;

    int nxny = nx*ny;
    int iz = idx / nxny;          // 0 .. nz/2
    int rem = idx - iz*nxny;
    int iy = rem / nx;            // 0 .. ny-1
    int ix = rem - iy*nx;         // 0 .. nx-1

    float kxv = kx[ix], kyv = ky[iy], kzv = kz[iz];
    float k2  = fmaf(kxv, kxv, fmaf(kyv, kyv, kzv*kzv));
    if (k2 == 0.f) { exk[idx].x=exk[idx].y=0.f; eyk[idx]=exk[idx]; ezk[idx]=exk[idx]; return; }

    float bmod2 = bx[ix] * by[iy] * bz[iz];
    if (bmod2 <= 1e-10f) { exk[idx].x=exk[idx].y=0.f; eyk[idx]=exk[idx]; ezk[idx]=exk[idx]; return; }

    float ghat = (2.0f*3.14159265358979323846f*2.0f / vol) * __expf(-k2/(4.0f*alpha*alpha)) / (k2*bmod2);

    float a = rho[idx].x * ghat;
    float b = rho[idx].y * ghat;

    exk[idx].x =  kxv * b; exk[idx].y = -kxv * a;
    eyk[idx].x =  kyv * b; eyk[idx].y = -kyv * a;
    ezk[idx].x =  kzv * b; ezk[idx].y = -kzv * a;

    // after computing a,b and setting exk/eyk/ezk
    const bool rim_x = (ix==0) || ((nx%2)==0 && ix==(nx/2));
    const bool rim_y = (iy==0) || ((ny%2)==0 && iy==(ny/2));
    const bool rim_z = (iz==0) || ((nz%2)==0 && iz==(nz/2));
    // Imag parts must be zero on self-conjugate rims
    if (rim_x) exk[idx].y = 0.0f;
    if (rim_y) eyk[idx].y = 0.0f;
    if (rim_z) ezk[idx].y = 0.0f;
    // Optional (more conservative): zero the entire component on its Nyquist plane
    // if ((nx%2)==0 && ix==(nx/2)) exk[idx].x = exk[idx].y = 0.0f;
    // if ((ny%2)==0 && iy==(ny/2)) eyk[idx].x = eyk[idx].y = 0.0f;
    // if ((nz%2)==0 && iz==(nz/2)) ezk[idx].x = ezk[idx].y = 0.0f;
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
void apply_ghat_and_grad_launch(
    const void* rho,
    void* exk, void* eyk, void* ezk,
    const void* kx, const void* ky, const void* kz,
    const void* bx, const void* by, const void* bz,
    int nx, int ny, int nz, float vol, float alpha,
    void* cu_stream)
{
    auto s = reinterpret_cast<cudaStream_t>(cu_stream);
    int n = nx*ny*(nz/2 + 1);          // <-- define n here

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    apply_ghat_and_grad<<<blocks, threads, 0, s>>>(
        static_cast<const cufftComplex*>(rho),
        static_cast<cufftComplex*>(exk),
        static_cast<cufftComplex*>(eyk),
        static_cast<cufftComplex*>(ezk),
        static_cast<const float*>(kx),
        static_cast<const float*>(ky),
        static_cast<const float*>(kz),
        static_cast<const float*>(bx),
        static_cast<const float*>(by),
        static_cast<const float*>(bz),
        nx, ny, nz, vol, alpha
    );
}

extern "C"
void exec_inverse_ExEyEz_c2r(void* plan,
                                  cufftComplex* exk, /* base of [exk|eyk|ezk] */
                                  cufftComplex* /*eyk*/,
                                  cufftComplex* /*ezk*/,
                                  float* ex /* base of [ex|ey|ez] */,
                                  float* /*ey*/,
                                  float* /*ez*/)
{
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    if (!w) return;

    // exk must point to 3*n_cmplx contiguous cufftComplex
    // ex  must point to 3*n_real   contiguous float
    CUFFT_CHECK(cufftExecC2R(w->plan_c2r_many, exk, ex));
}

extern "C"
void spme_exec_forward_r2c(void* plan, float* rho_real, cufftComplex* rho_k) {
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    if (!w) return;
    CUFFT_CHECK(cufftExecR2C(w->plan_r2c, rho_real, rho_k));
}

__global__ void energy_half_spectrum(
    const cufftComplex* __restrict__ rho_k,
    const float* __restrict__ kx, const float* __restrict__ ky, const float* __restrict__ kz,
    const float* __restrict__ bx, const float* __restrict__ by, const float* __restrict__ bz,
    int nx, int ny, int nz, float vol, float alpha,
    double* __restrict__ out_partial)
{
    extern __shared__ double ssum[];
    int tid = threadIdx.x;
    double acc = 0.0;

    int nxy = nx*ny;
    int n_cmplx = nxy*(nz/2 + 1);

    // CUFFT normalization for |rho_k|^2
    int N = nx*ny*nz;
    double invN2 = 1.0 / (double(N) * double(N));

    for (int idx = blockIdx.x*blockDim.x + tid; idx < n_cmplx; idx += gridDim.x*blockDim.x) {
        int iz = idx / nxy;
        int rem = idx - iz*nxy;
        int iy = rem / nx;
        int ix = rem - iy*nx;

        float kxv = kx[ix], kyv = ky[iy], kzv = kz[iz];
        float k2  = fmaf(kxv,kxv, fmaf(kyv,kyv, kzv*kzv));
        if (k2 == 0.f) continue;

        float bmod2 = bx[ix]*by[iy]*bz[iz];
        if (bmod2 <= 1e-10f) continue;

        float ghat = (2.0f*3.14159265358979323846f*2.0f / vol) * __expf(-k2/(4.0f*alpha*alpha)) / (k2*bmod2);

        float a = rho_k[idx].x, b = rho_k[idx].y;
        double mag2 = double(a)*a + double(b)*b;

        int twice = (iz==0 || ((nz%2)==0 && iz==(nz/2))) ? 1 : 2;
        acc += 0.5 * double(twice) * double(ghat) * (mag2 * invN2);
    }
    ssum[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x/2; s>0; s>>=1) {
        if (tid < s) ssum[tid] += ssum[tid+s];
        __syncthreads();
    }
    if (tid==0) out_partial[blockIdx.x] = ssum[0];
}

extern "C"
void energy_half_spectrum_launch(
    const void* rho_k,
    const void* kx, const void* ky, const void* kz,
    const void* bx, const void* by, const void* bz,
    int nx, int ny, int nz, float vol, float alpha,
    void* partial_sums,   // device ptr to double[blocks]
    int blocks, int threads, void* cu_stream)
{
    auto s = reinterpret_cast<cudaStream_t>(cu_stream);
    size_t shmem = size_t(threads) * sizeof(double);

    energy_half_spectrum<<<blocks, threads, shmem, s>>>(
        static_cast<const cufftComplex*>(rho_k),
        static_cast<const float*>(kx), static_cast<const float*>(ky), static_cast<const float*>(kz),
        static_cast<const float*>(bx), static_cast<const float*>(by), static_cast<const float*>(bz),
        nx, ny, nz, vol, alpha,
        static_cast<double*>(partial_sums));
}