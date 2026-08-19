// Perform FFTs using cuFFT. [docs](https://docs.nvidia.com/cuda/cufft/)

// We use the scatter functionality here for both cuFFT and
// VkFFT. The rest of this is for cuFFT only.

// cuFFT is resolved at runtime (dlopen / LoadLibrary) rather than linked against, so that a binary
// built with the `cufft` feature still *starts* on a machine with no CUDA installation. A load-time
// dependency on libcufft.so.12 / cufft64_12.dll would make the OS loader refuse to run the
// executable there, defeating the CPU fallback. `gpu_fft_available()` reports whether the lookup
// succeeded; callers use it to decide between the GPU and CPU paths.

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <dlfcn.h>
  #include <unistd.h>
#endif


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


// The handful of cuFFT entry points we use, looked up by name.
typedef cufftResult (*Fn_cufftPlan3d)(cufftHandle*, int, int, int, cufftType);
typedef cufftResult (*Fn_cufftPlanMany)(cufftHandle*, int, int*, int*, int, int,
                                        int*, int, int, cufftType, int);
typedef cufftResult (*Fn_cufftSetStream)(cufftHandle, cudaStream_t);
typedef cufftResult (*Fn_cufftDestroy)(cufftHandle);
typedef cufftResult (*Fn_cufftExecR2C)(cufftHandle, cufftReal*, cufftComplex*);
typedef cufftResult (*Fn_cufftExecC2R)(cufftHandle, cufftComplex*, cufftReal*);

struct CufftApi {
    Fn_cufftPlan3d    plan3d;
    Fn_cufftPlanMany  plan_many;
    Fn_cufftSetStream set_stream;
    Fn_cufftDestroy   destroy;
    Fn_cufftExecR2C   exec_r2c;
    Fn_cufftExecC2R   exec_c2r;
    bool loaded;
};

static CufftApi g_cufft = {};
static bool g_cufft_attempted = false;

// The SONAME/filename varies with the CUDA major version; try newest first. On Windows the default
// search order starts with the directory the executable lives in, so a DLL shipped alongside the
// application is found without an install step.
#ifdef _WIN32
static const char* CUFFT_LIB_NAMES[] = {
    "cufft64_12.dll", "cufft64_11.dll", "cufft64_10.dll",
};
#else
static const char* CUFFT_LIB_NAMES[] = {
    "libcufft.so.12", "libcufft.so.11", "libcufft.so.10", "libcufft.so",
};
#endif

#ifdef _WIN32
typedef HMODULE LibHandle;
static LibHandle lib_open(const char* name) { return LoadLibraryA(name); }
static void* lib_sym(LibHandle h, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(h, name));
}
#else
typedef void* LibHandle;
static LibHandle lib_open(const char* name) { return dlopen(name, RTLD_LAZY | RTLD_LOCAL); }
static void* lib_sym(LibHandle h, const char* name) { return dlsym(h, name); }

// Linux has no equivalent of Windows' "look next to the .exe" search rule, and we would rather not
// require that the application was linked with an `$ORIGIN` rpath. Build an absolute path into the
// executable's own directory so a bundled copy of the library is picked up as well.
static bool exe_relative_path(const char* lib_name, char* out, size_t out_len) {
    char exe_path[4096];
    ssize_t n = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (n <= 0) return false;
    exe_path[n] = '\0';

    char* slash = strrchr(exe_path, '/');
    if (!slash) return false;
    *slash = '\0';

    if (strlen(exe_path) + strlen(lib_name) + 2 > out_len) return false;
    snprintf(out, out_len, "%s/%s", exe_path, lib_name);
    return true;
}
#endif

static LibHandle open_cufft_lib() {
    const size_t count = sizeof(CUFFT_LIB_NAMES) / sizeof(CUFFT_LIB_NAMES[0]);

    for (size_t i = 0; i < count; i++) {
        LibHandle h = lib_open(CUFFT_LIB_NAMES[i]);
        if (h) return h;
    }

#ifndef _WIN32
    for (size_t i = 0; i < count; i++) {
        char path[4096];
        if (!exe_relative_path(CUFFT_LIB_NAMES[i], path, sizeof(path))) continue;
        LibHandle h = lib_open(path);
        if (h) return h;
    }
#endif

    return nullptr;
}

// Resolves the cuFFT entry points on first use. Returns null if cuFFT is not present.
static const CufftApi* cufft_api() {
    if (!g_cufft_attempted) {
        g_cufft_attempted = true;

        LibHandle h = open_cufft_lib();
        if (h) {
            g_cufft.plan3d     = (Fn_cufftPlan3d)   lib_sym(h, "cufftPlan3d");
            g_cufft.plan_many  = (Fn_cufftPlanMany) lib_sym(h, "cufftPlanMany");
            g_cufft.set_stream = (Fn_cufftSetStream)lib_sym(h, "cufftSetStream");
            g_cufft.destroy    = (Fn_cufftDestroy)  lib_sym(h, "cufftDestroy");
            g_cufft.exec_r2c   = (Fn_cufftExecR2C)  lib_sym(h, "cufftExecR2C");
            g_cufft.exec_c2r   = (Fn_cufftExecC2R)  lib_sym(h, "cufftExecC2R");

            g_cufft.loaded = g_cufft.plan3d && g_cufft.plan_many && g_cufft.set_stream &&
                             g_cufft.destroy && g_cufft.exec_r2c && g_cufft.exec_c2r;

            if (!g_cufft.loaded) {
                printf("cuFFT was found, but is missing expected symbols; using the CPU FFT.\n");
            }
        }
        // We deliberately never unload: the resolved pointers stay valid for the process lifetime.
    }

    return g_cufft.loaded ? &g_cufft : nullptr;
}

// Whether the cuFFT library could be loaded. 1 = yes. Callers use this to decide whether the GPU
// path is usable at all, before committing to it.
extern "C"
int gpu_fft_available() {
    return cufft_api() ? 1 : 0;
}

struct PlanWrap {
    cufftHandle plan_r2c;
    cufftHandle plan_c2r;
    cudaStream_t stream;
};

// https://docs.nvidia.com/cuda/cufft/#cufftplan3d
extern "C"
void* make_plan(int nx, int ny, int nz, void* cu_stream) {
    const CufftApi* api = cufft_api();
    if (!api) {
        printf("cuFFT is unavailable; cannot create a GPU FFT plan.\n");
        return nullptr;
    }

    auto* w = new PlanWrap();

    w->stream = reinterpret_cast<cudaStream_t>(cu_stream);

    // With Plan3D, Z is the fastest-changing dimension (contiguous); x is the slowest.
    CUFFT_CHECK(api->plan3d(&w->plan_r2c, nx, ny, nz, CUFFT_R2C));
    // The electric-field inverse transforms have identical dimensions and are
    // stored contiguously as [Ex | Ey | Ez], so execute them as one batch.
    int dims[3] = {nx, ny, nz};
    const int complex_dist = nx * ny * (nz / 2 + 1);
    const int real_dist = nx * ny * nz;
    CUFFT_CHECK(api->plan_many(
        &w->plan_c2r,
        3,
        dims,
        nullptr,
        1,
        complex_dist,
        nullptr,
        1,
        real_dist,
        CUFFT_C2R,
        3
    ));

    CUFFT_CHECK(api->set_stream(w->plan_r2c, w->stream));
    CUFFT_CHECK(api->set_stream(w->plan_c2r, w->stream));

    return w;
}


extern "C"
void destroy_plan(void* plan) {
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    if (!w) return;

    const CufftApi* api = cufft_api();
    if (api) {
        api->destroy(w->plan_r2c);
        api->destroy(w->plan_c2r);
    }

    delete w;
}

// https://docs.nvidia.com/cuda/cufft/#cufftexecr2c-and-cufftexecd2z
// Performs a forward real-to-copmlex FFT of rho. Note: This is more efficient
// than complex-to-complex.
extern "C"
void exec_forward(void* plan, float* rho_real, cufftComplex* rho) {
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    if (!w) return;

    const CufftApi* api = cufft_api();
    if (!api) return;

    CUFFT_CHECK(api->exec_r2c(w->plan_r2c, rho_real, rho));
}

extern "C"
void exec_inverse(
    void* plan,
    cufftComplex* ek,
    float* e
){
    auto* w = reinterpret_cast<PlanWrap*>(plan);
    if (!w) return;

    const CufftApi* api = cufft_api();
    if (!api) return;

    CUFFT_CHECK(api->exec_c2r(w->plan_c2r, ek, e));
}
