// Contains device code; GPU kernels. It does not perform FFTs. This does not include a cuFFT, nor vkFFT dependency.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// Note: We use float2 instead of cufftComplex, as it doesn't rely on cuFFT.

const int32_t SPLINE_ORDER = 4;

__device__ __forceinline__
int wrap(int32_t a, int32_t n) {
    a %= n; return (a < 0) ? a + n : a;
}


//  Corresponds directly to a host function.
__device__
void bspline4_weights(float s, int32_t* i0, float w[4]) {
    float sfloor = floorf(s);
    float u = s - sfloor;
    *i0 = (int32_t)sfloor - 1;

    float u2 = u * u;
    float u3 = fmaf(u2, u, 0.0f);

    float w0 = (1.0f - u);
    w0 = (w0 * w0 * w0) * (1.0f / 6.0f);

    float w1 = (3.0f * u3 - 6.0f * u2 + 4.0f) * (1.0f / 6.0f);
    float w2 = (-3.0f * u3 + 3.0f * u2 + 3.0f * u + 1.0f) * (1.0f / 6.0f);
    float w3 = u3 * (1.0f / 6.0f);

    w[0] = w0;
    w[1] = w1;
    w[2] = w2;
    w[3] = w3;
}

// Kernel for charge spreading. Z as the fast/contiguous axis.
// Corresponds directly to a equivalent CPU function.
extern "C" __global__
void spread_charges(
    const float3* pos,
    const float*  q,
    float* rho, // real part
    int32_t n_atoms,
    int32_t nx,
    int32_t ny,
    int32_t nz,
    float lx,
    float ly,
    float lz
) {
    size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t nynz = ny * nz;

    for (size_t i = i0; i < (size_t)n_atoms; i += stride) {
        float3 r = pos[i];

        float sx = r.x / lx * nx;
        float sy = r.y / ly * ny;
        float sz = r.z / lz * nz;

        int32_t ix0, iy0, iz0;
        float wx[SPLINE_ORDER], wy[SPLINE_ORDER], wz[SPLINE_ORDER];

        bspline4_weights(sx, &ix0, wx);
        bspline4_weights(sy, &iy0, wy);
        bspline4_weights(sz, &iz0, wz);

        float qi = q[i];

        for (int32_t a=0; a < SPLINE_ORDER; a++) {
            int32_t ix = wrap(ix0 + a, nx);
            float wxa = wx[a];

            for (int32_t b=0; b < SPLINE_ORDER; b++) {
                int32_t iy = wrap(iy0 + b, ny);
                float wxy = wxa * wy[b];

                // Z-fast base for this (ix,iy) column
                size_t base = (size_t)ix * nynz + (size_t)iy * (size_t)nz;

                for (int32_t c=0; c < SPLINE_ORDER; c++) {
                    int32_t iz = wrap(iz0 + c, nz);
                    size_t idx = base + (size_t)iz; // contiguous over z
                    atomicAdd(&rho[idx], qi * wxy * wz[c]);
                }
            }
        }
    }
}


// A kernel. Apply G(k) and gradient to get Exk/Eyk/Ezk
// todo: Deprecated in favor of the *compute potential* version.
extern "C" __global__
void apply_ghat_and_grad(
    const float2* rho,
    float2* exk,
    float2* eyk,
    float2* ezk,
    //
    const float* kx,
    const float* ky,
    const float* kz,
    //
    const float* bx,
    const float* by,
    const float* bz,
    int32_t nx,
    int32_t ny,
    int32_t nz,
    float vol,
    float alpha,
    int32_t n_real
 ) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t nzc = nz/2 + 1;
    int32_t n_cmplx = nx * ny * nzc;
    if (idx >= n_cmplx) return;

    // Z-fast decoding
    int32_t iz = idx % nzc;                 // 0 .. nz/2
    int32_t iy = (idx / nzc) % ny;          // 0 .. ny-1
    int32_t ix =  idx / (nzc * ny);         // 0 .. nx-1

    float kxv = kx[ix], kyv = ky[iy], kzv = kz[iz];

    float k2  = fmaf(kxv, kxv, fmaf(kyv, kyv, kzv*kzv));
    if (k2 == 0.f) { exk[idx].x=exk[idx].y=0.f; eyk[idx]=exk[idx]; ezk[idx]=exk[idx]; return; }

    float bmod2 = bx[ix] * by[iy] * bz[iz];
    if (bmod2 <= 1e-10f) { exk[idx].x=exk[idx].y=0.f; eyk[idx]=exk[idx]; ezk[idx]=exk[idx]; return; }

    const float TWO_TAU = 12.56637061435917295385f; // 4π

    float ghat = (TWO_TAU / vol) * __expf(-k2 / (4.0f * alpha * alpha)) / (k2 * bmod2);

    // φ(k) = G(k) * ρ(k)
    float phi_k_real = rho[idx].x * ghat;
    float phi_k_im   = rho[idx].y * ghat;

    // E(k) = i * k * φ(k)
    // ex = (-kx * Im φ,  kx * Re φ)
    // ey = (-ky * Im φ,  ky * Re φ)
    // ez = (-kz * Im φ,  kz * Re φ)
    exk[idx].x = -kxv * phi_k_im;
    exk[idx].y =  kxv * phi_k_real;

    eyk[idx].x = -kyv * phi_k_im;
    eyk[idx].y =  kyv * phi_k_real;

    ezk[idx].x = -kzv * phi_k_im;
    ezk[idx].y =  kzv * phi_k_real;


    // Self-conjugate rim handling:
    const bool rim_x = (ix==0) || ((nx%2)==0 && ix==(nx/2));
    const bool rim_y = (iy==0) || ((ny%2)==0 && iy==(ny/2));
    const bool rim_z = (iz==0) || ((nz%2)==0 && iz==(nz/2));


    // For φ(k) real on rims, E(k) should be purely imaginary after i*k multiplication.
    // So zero the REAL parts, not the imaginary ones.
    if (rim_x) exk[idx].x = 0.0f;
    if (rim_y) eyk[idx].x = 0.0f;
    if (rim_z) ezk[idx].x = 0.0f;
}

// A kernel. Apply G(k) to get phi_k, and (optionally) write per-mode energy for reduction.
extern "C" __global__
void apply_ghat_and_compute_potential(
    const float2* rho,
    float2* phi_k,
    //
    const float* kx,
    const float* ky,
    const float* kz,
    //
    const float* bx,
    const float* by,
    const float* bz,
    int32_t nx,
    int32_t ny,
    int32_t nz,
    float vol,
    float alpha,
    //
    double* energy_out // may be null; if non-null, energy_out[idx] = 0.5 * Re(rho * phi_k*)
) {
    int32_t idx = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);

    int32_t nzc = nz / 2 + 1;
    int32_t n_cmplx = nx * ny * nzc;
    if (idx >= n_cmplx) return;

    int32_t iz = idx % nzc;
    int32_t iy = (idx / nzc) % ny;
    int32_t ix = idx / (nzc * ny);

    float kxv = kx[ix];
    float kyv = ky[iy];
    float kzv = kz[iz];

    float k2 = fmaf(kxv, kxv, fmaf(kyv, kyv, kzv * kzv));

    float bmod2 = bx[ix] * by[iy] * bz[iz];

    if (k2 == 0.0f || bmod2 <= 1e-10f) {
        phi_k[idx].x = 0.0f;
        phi_k[idx].y = 0.0f;
        if (energy_out) energy_out[idx] = 0.0;
        return;
    }

    const float TWO_TAU = 12.56637061435917295385f; // 4π

    float ghat = (TWO_TAU / vol) * __expf(-k2 / (4.0f * alpha * alpha)) / (k2 * bmod2);

    float2 rho_v = rho[idx];

    float2 val;
    val.x = rho_v.x * ghat;
    val.y = rho_v.y * ghat;

    // Enforce self-conjugate points to be purely real (numerical cleanup).
    const bool rim_x = (ix == 0) || (((nx & 1) == 0) && (ix == (nx / 2)));
    const bool rim_y = (iy == 0) || (((ny & 1) == 0) && (iy == (ny / 2)));
    const bool rim_z = (iz == 0) || (((nz & 1) == 0) && (iz == (nz / 2)));
    if (rim_x && rim_y && rim_z) {
        val.y = 0.0f;
    }

    phi_k[idx] = val;

    if (energy_out) {
        // 0.5 * (rho.re * val.re + rho.im * val.im)
        energy_out[idx] = 0.5 * ((double)rho_v.x * (double)val.x + (double)rho_v.y * (double)val.y);
    }
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
    int32_t n_atoms,
    int32_t nx,
    int32_t ny,
    int32_t nz,
    float lx,
    float ly,
    float lz
) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;

    float3 r = pos[i];
    float sx = r.x / lx * nx;
    float sy = r.y / ly * ny;
    float sz = r.z / lz * nz;

    int32_t ix0 = __float2int_rd(sx) - 1;
    int32_t iy0 = __float2int_rd(sy) - 1;
    int32_t iz0 = __float2int_rd(sz) - 1;

    float u = sx - floorf(sx);
    float v = sy - floorf(sy);
    float w = sz - floorf(sz);

    float u2=u*u, u3=u2*u, um=1.f-u; float wx[4] = {(um*um*um)/6.f,(3.f*u3-6.f*u2+4.f)/6.f,(-3.f*u3+3.f*u2+3.f*u+1.f)/6.f,u3/6.f};
    float v2=v*v, v3=v2*v, vm=1.f-v; float wy[4] = {(vm*vm*vm)/6.f,(3.f*v3-6.f*v2+4.f)/6.f,(-3.f*v3+3.f*v2+3.f*v+1.f)/6.f,v3/6.f};
    float w2=w*w, w3=w2*w, wm=1.f-w; float wz[4] = {(wm*wm*wm)/6.f,(3.f*w3-6.f*w2+4.f)/6.f,(-3.f*w3+3.f*w2+3.f*w+1.f)/6.f,w3/6.f};

    float Exi=0.f, Eyi=0.f, Ezi=0.f;
        for (int32_t a=0; a<4; a++){
            int32_t ix = wrap(ix0 + a, nx);
            float wxa = wx[a];

            for (int32_t b=0; b<4; b++){
                int32_t iy = wrap(iy0 + b, ny);
                float wxy = wxa * wy[b];

                // Z-fast base for this (ix,iy)
                size_t base = (size_t)ix * (size_t)(ny * nz) + (size_t)iy * (size_t)nz;

                for (int32_t c=0; c<4; c++){
                    int32_t iz = wrap(iz0 + c, nz);
                    float wfac = wxy * wz[c];
                    size_t idx = base + (size_t)iz;   // contiguous over z

                    Exi += wfac * ex[idx];
                    Eyi += wfac * ey[idx];
                    Ezi += wfac * ez[idx];
               }
           }
        }

    float s = q[i];
    out_f[i] = make_float3(Exi*s, Eyi*s, Ezi*s);
}


// todo: Can or should we combine this with ghat and grad as we do on the CPU? Likely
// todo won't make much of a performance impact.
// A kernel
extern "C" __global__
void energy_half_spectrum(
    const float2* rho_k,
    const float* kx,
    const float* ky,
    const float* kz,
    const float* bx,
    const float* by,
    const float* bz,
    int32_t nx,
    int32_t ny,
    int32_t nz,
    float vol,
    float alpha,
    double* out_partial
) {
    __shared__ double ssum[256];
    int32_t tid = threadIdx.x;
    double acc = 0.0;

    int32_t nzc = nz/2 + 1;
    int32_t n_cmplx = nx * ny * nzc;

    for (int32_t idx = blockIdx.x*blockDim.x + tid; idx < n_cmplx; idx += gridDim.x*blockDim.x) {
        int32_t iz = idx % nzc;
        int32_t iy = (idx / nzc) % ny;
        int32_t ix =  idx / (nzc * ny);

        float kxv = kx[ix], kyv = ky[iy], kzv = kz[iz];
        float k2  = fmaf(kxv,kxv, fmaf(kyv,kyv, kzv*kzv));
        if (k2 == 0.f) continue;

        float bmod2 = bx[ix]*by[iy]*bz[iz];
        if (bmod2 <= 1e-10f) continue;

        // 4π/vol * exp(-k²/(4α²)) / (k² * bmod²)
        float ghat = (2.0f*3.14159265358979323846f*2.0f / vol) *
                     __expf(-k2/(4.0f*alpha*alpha)) / (k2*bmod2);

        double a = (double)rho_k[idx].x;
        double b = (double)rho_k[idx].y;
        double phi_re = (double)ghat * a;
        double phi_im = (double)ghat * b;

        // 1/2 * Re{ rho*(k) * phi(k) } = 1/2 * (a*phi_re + b*phi_im)
        acc += 0.5 * (a*phi_re + b*phi_im);
    }

    ssum[tid] = acc;
    __syncthreads();
    for (int32_t s = blockDim.x/2; s>0; s>>=1) {
        if (tid < s) ssum[tid] += ssum[tid+s];
        __syncthreads();
    }
    if (tid==0) out_partial[blockIdx.x] = ssum[0];
}
