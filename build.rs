//! We use this to automatically compile CUDA C++ code when building.

#[cfg(any(feature = "cufft", feature = "vkfft"))]
use cc;
#[cfg(feature = "cuda")]
use cuda_setup::{GpuArchitecture, build_ptx};

fn main() {
    // Build non-FFT kernels that are used for both GPU FFT branches.
    #[cfg(feature = "cuda")]
    build_ptx(
        // Select the min supported GPU architecture.
        GpuArchitecture::Rtx3,
        &["src/cuda/kernels.cu"],
        "ewald",
    );

    // cuFFT-specific host-side building.
    //
    // We compile this with `cc` directly rather than with `cuda_setup::build_host`, because that
    // helper emits `cargo:rustc-link-lib=cufft`. Linking cuFFT would make libcufft.so.12 /
    // cufft64_12.dll a load-time dependency of the final executable, so a binary built with this
    // feature would refuse to start at all on a machine without CUDA installed. `cufft.cu` resolves
    // the cuFFT entry points itself with dlopen/LoadLibrary instead, which keeps the CPU fallback
    // reachable. (`cc`'s `.cuda(true)` links the CUDA *runtime* statically, so that adds no
    // load-time dependency either.)
    #[cfg(feature = "cufft")]
    {
        for f in ["src/cuda/cufft.cu", "src/cuda/kernels.cu"] {
            println!("cargo:rerun-if-changed={f}");
        }

        let mut build = cc::Build::new();
        build
            .cuda(true)
            .file("src/cuda/cufft.cu")
            .flag("-O3")
            .flag("-std=c++20")
            .flag(GpuArchitecture::Rtx3.sm_val());

        if cfg!(target_os = "linux") {
            build.flag("-Xcompiler=-fPIC");
        }

        build.compile("spme");

        // `dlopen`/`dlsym`. Folded into libc as of glibc 2.34, but older distros still need it.
        #[cfg(target_os = "linux")]
        println!("cargo:rustc-link-lib=dylib=dl");
    }

    // VkFFT-specifical host-side building
    #[cfg(feature = "vkfft")]
    {
        // Our FFI files.
        println!("cargo:rerun-if-changed=src/cuda/vk_fft.cu");
        println!("cargo:rerun-if-changed=src/cuda/vk_fft.h");
        // This is the vkFFT header
        println!("cargo:rerun-if-changed=third_party/VkFFT/vkFFT.h");

        // Cargo already caches build-script outputs under OUT_DIR. Building via
        // `cc` here also emits the CUDA runtime link directives; a separate
        // archive cache used to omit those directives on cache hits.
        cc::Build::new()
            .cuda(true)
            .files(["src/cuda/vk_fft.cu"])
            .define("VKFFT_BACKEND", Some("1")) // Sets the backend to CUDA.
            .include("third_party/VkFFT/vkFFT")
            .flag_if_supported("-O3")
            .compile("vk_fft");
    }
}
