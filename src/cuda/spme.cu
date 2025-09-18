#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cstdio>


// put this at file scope (not inside a kernel)
__device__ __forceinline__ int wrap_i(int a, int n) { a %= n; return (a < 0) ? a + n : a; }

struct PlanWrap {
    cufftHandle plan;
    size_t n_per_grid;
    cudaStream_t stream;
};


extern "C" __global__
void spme_gather_forces_to_atoms_cplx(
    const float3* pos,
    const cufftComplex* exk,
    const cufftComplex* eyk,
    const cufftComplex* ezk,
    const float* q,
    float3* out_f,
    int n_atoms, int nx, int ny, int nz,
    float lx, float ly, float lz,
    float inv_n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;

    float3 r = pos[i];
    float sx = r.x / lx * nx;
    float sy = r.y / ly * ny;
    float sz = r.z / lz * nz;

    int ix0 = __float2int_rd(sx) - 1;
    int iy0 = __float2int_rd(sy) - 1;
    int iz0 = __float2int_rd(sz) - 1;

    float u = sx - floorf(sx);
    float v = sy - floorf(sy);
    float w = sz - floorf(sz);

    float u2=u*u, u3=u2*u, um=1.f-u; float wx[4] = {(um*um*um)/6.f,(3.f*u3-6.f*u2+4.f)/6.f,(-3.f*u3+3.f*u2+3.f*u+1.f)/6.f,u3/6.f};
    float v2=v*v, v3=v2*v, vm=1.f-v; float wy[4] = {(vm*vm*vm)/6.f,(3.f*v3-6.f*v2+4.f)/6.f,(-3.f*v3+3.f*v2+3.f*v+1.f)/6.f,v3/6.f};
    float w2=w*w, w3=w2*w, wm=1.f-w; float wz[4] = {(wm*wm*wm)/6.f,(3.f*w3-6.f*w2+4.f)/6.f,(-3.f*w3+3.f*w2+3.f*w+1.f)/6.f,w3/6.f};

    float Ex=0.f, Ey=0.f, Ez=0.f;
    for(int a=0;a<4;++a){
        int ix = wrap_i(ix0 + a, nx);
        float wxa = wx[a];
        for(int b=0;b<4;++b){
            int iy = wrap_i(iy0 + b, ny);
            float wxy = wxa * wy[b];
            size_t base = size_t(iy)*nx + ix;
            for(int c=0;c<4;++c){
                int iz = wrap_i(iz0 + c, nz);
                float wfac = wxy * wz[c];
                size_t idx = size_t(iz)*nx*ny + base;
                Ex += wfac * exk[idx].x;
                Ey += wfac * eyk[idx].x;
                Ez += wfac * ezk[idx].x;
            }
        }
    }

    float s = q[i] * inv_n;
    out_f[i] = make_float3(Ex*s, Ey*s, Ez*s);
}

extern "C"
void spme_gather_forces_to_atoms_cplx_launch(
    const void* pos,
    const void* exk, const void* eyk, const void* ezk,
    const void* q,
    void* out_f,
    int n_atoms, int nx, int ny, int nz,
    float lx, float ly, float lz,
    float inv_n,
    void* cu_stream)
{
    auto s = reinterpret_cast<cudaStream_t>(cu_stream);
    int threads = 256;
    int blocks  = (n_atoms + threads - 1) / threads;
    spme_gather_forces_to_atoms_cplx<<<blocks, threads, 0, s>>>(
        static_cast<const float3*>(pos),
        static_cast<const cufftComplex*>(exk),
        static_cast<const cufftComplex*>(eyk),
        static_cast<const cufftComplex*>(ezk),
        static_cast<const float*>(q),
        static_cast<float3*>(out_f),
        n_atoms, nx, ny, nz, lx, ly, lz, inv_n
    );
}

// spme_fft.cu
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>

static __global__ void c2c_to_real3_scale_k(
    const cufftComplex* __restrict__ exk,
    const cufftComplex* __restrict__ eyk,
    const cufftComplex* __restrict__ ezk,
    float* __restrict__ ex,
    float* __restrict__ ey,
    float* __restrict__ ez,
    size_t n, float scale)
{
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) {
        ex[i] = exk[i].x * scale;
        ey[i] = eyk[i].x * scale;
        ez[i] = ezk[i].x * scale;
    }
}

// Your kernel (unchanged) + a tiny launcher wrapper:

extern "C" __global__
void spme_gather_forces_to_atoms(
    const float3* __restrict__ pos,
    const float*  __restrict__ ex,
    const float*  __restrict__ ey,
    const float*  __restrict__ ez,
    const float*  __restrict__ q,
    float3*       __restrict__ out_f,
    int n_atoms, int nx, int ny, int nz,
    float lx, float ly, float lz);


extern "C"
void* spme_make_plan_c2c(int nx, int ny, int nz, void* cu_stream) {
    auto* w = new PlanWrap();
    w->n_per_grid = size_t(nx) * ny * nz;
    w->stream = reinterpret_cast<cudaStream_t>(cu_stream);

    cufftResult r = cufftPlan3d(&w->plan, nx, ny, nz, CUFFT_C2C);
    if (r != CUFFT_SUCCESS) { printf("cufftPlan3d err=%d\n", int(r)); delete w; return nullptr; }

    r = cufftSetStream(w->plan, w->stream);
    if (r != CUFFT_SUCCESS) { printf("cufftSetStream err=%d\n", int(r)); cufftDestroy(w->plan); delete w; return nullptr; }

    return w;
}

extern "C"
void spme_exec_inverse_3_c2c(void* plan, cufftComplex* exk, cufftComplex* eyk, cufftComplex* ezk) {
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    if (!w) return;
    cufftExecC2C(w->plan, exk, exk, CUFFT_INVERSE);
    cufftExecC2C(w->plan, eyk, eyk, CUFFT_INVERSE);
    cufftExecC2C(w->plan, ezk, ezk, CUFFT_INVERSE);
}

extern "C"
void spme_destroy_plan(void* plan) {
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    if (!w) return;
    cufftDestroy(w->plan);
    delete w;
}

// kspace.cu
#include <cufft.h>
extern "C" __global__
void spme_apply_ghat_and_grad(
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
    int nxyz = nx*ny*nz;
    if (idx >= nxyz) return;

    int ix = idx % nx;
    int iy = (idx / nx) % ny;
    int iz = idx / (nx*ny);

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
}

extern "C"
void spme_apply_ghat_and_grad_launch(
    const void* rho,
    void* exk, void* eyk, void* ezk,
    const void* kx, const void* ky, const void* kz,
    const void* bx, const void* by, const void* bz,
    int nx, int ny, int nz, float vol, float alpha,
    void* cu_stream)
{
    auto s = reinterpret_cast<cudaStream_t>(cu_stream);
    int n = nx*ny*nz;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    spme_apply_ghat_and_grad<<<blocks, threads, 0, s>>>(
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
