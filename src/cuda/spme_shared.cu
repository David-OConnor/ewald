// Contains code used by both GPU pipelines. This does not include a cuFFT dependency.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>


extern "C" __global__
void spme_scatter_rho_4x4x4(
    const float3* __restrict__ pos,
    const float*  __restrict__ q,
    float*        __restrict__ rho,   // real grid, size nx*ny*nz
    int n_atoms, int nx, int ny, int nz,
    float lx, float ly, float lz)
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


// todo: Does this need a __device__ tag?
extern "C"
void spme_scale_ExEyEz_after_c2r(float* ex, float* ey, float* ez,
                                 int nx, int ny, int nz, void* cu_stream) {
    auto s = reinterpret_cast<cudaStream_t>(cu_stream);
    size_t n = size_t(nx)*ny*nz;
    int threads = 256;
    int blocks  = int((n + threads - 1) / threads);
    float invN  = 1.0f / float(n);
    scale3<<<blocks, threads, 0, s>>>(ex, ey, ez, n, invN);
}

// todo: Does this need a __device__ tag?
extern "C"
void spme_gather_forces_to_atoms_launch(
    const void* pos,
    const void* ex, const void* ey, const void* ez,
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
    spme_gather_forces_to_atoms<<<blocks, threads, 0, s>>>(
        static_cast<const float3*>(pos),
        static_cast<const float*>(ex),
        static_cast<const float*>(ey),
        static_cast<const float*>(ez),
        static_cast<const float*>(q),
        static_cast<float3*>(out_f),
        n_atoms, nx, ny, nz, lx, ly, lz
    );
}