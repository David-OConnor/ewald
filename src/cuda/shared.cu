// Contains code used by both GPU pipelines. This does not include a cuFFT dependency.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// Note: We use float2 instead of cufftComplex, as it's compatible with vkFFT.

__device__ __forceinline__ int wrap_i(int a, int n) { a %= n; return (a < 0) ? a + n : a; }

__global__ void scale3(float* ex, float* ey, float* ez, size_t n, float s) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) {
        ex[i] *= s; ey[i] *= s; ez[i] *= s;
    }
}

// Kernel for charge spreading
extern "C" __global__
void spread_charges(
    const float3* pos,
    const float*  q,
    float* rho,  // real grid, size nx*ny*nz
    int n_atoms,
    int nx,
    int ny,
    int nz,
    float lx,
    float ly,
    float lz
) {
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

    float qi = q[i];

    for (int a=0; a<4; ++a) {
        int ix = wrap_i(ix0 + a, nx);
        float wxa = wx[a];
        for (int b=0; b<4; ++b) {
            int iy = wrap_i(iy0 + b, ny);
            float wxy = wxa * wy[b];
            size_t base = size_t(iy)*nx + ix;
            for (int c=0; c<4; ++c) {
                int iz = wrap_i(iz0 + c, nz);
                size_t idx = size_t(iz)*nx*ny + base;
                atomicAdd(&rho[idx], qi * wxy * wz[c]);
            }
        }
    }
}


// A kernel. Apply G(k) and gradient to get Exk/Eyk/Ezk
extern "C" __global__
void apply_ghat_and_grad(
    const float2* rho,
    float2* exk,
    float2* eyk,
    float2* ezk,
    const float* kx,
    const float* ky,
    const float* kz,
    const float* bx,
    const float* by,
    const float* bz,
    int nx, 
    int ny,
    int nz,
    float vol,
    float alpha
 ) {
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


// todo: should these have the thread/stride splitting your main short-range kernes have??
extern "C" __global__
void gather_forces_to_atoms(
    const float3* pos,
    const float*  ex,
    const float*  ey,
    const float*  ez,
    const float*  q,
    float3*       out_f,
    int n_atoms,
    int nx,
    int ny,
    int nz,
    float lx,
    float ly,
    float lz
) {
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

    float Exi=0.f, Eyi=0.f, Ezi=0.f;
    for (int a=0;a<4;++a){
        int ix = wrap_i(ix0 + a, nx);
        float wxa = wx[a];
        for (int b=0;b<4;++b){
            int iy = wrap_i(iy0 + b, ny);
            float wxy = wxa * wy[b];
            size_t base = size_t(iy)*nx + ix;
            for (int c=0;c<4;++c){
                int iz = wrap_i(iz0 + c, nz);
                float wfac = wxy * wz[c];
                size_t idx = size_t(iz)*nx*ny + base;
                Exi += wfac * ex[idx];
                Eyi += wfac * ey[idx];
                Ezi += wfac * ez[idx];
            }
        }
    }

    float s = q[i];
    out_f[i] = make_float3(Exi*s, Eyi*s, Ezi*s);
}



// A kernel
__global__ void energy_half_spectrum(
    const float2* rho_k,
    const float* kx,
    const float* ky,
    const float* kz,
    const float* bx,
    const float* by,
    const float* bz,
    int nx,
    int ny,
    int nz,
    float vol,
    float alpha,
    double* out_partial
) {
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
