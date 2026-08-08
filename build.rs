//! We use this to automatically compile CUDA C++ code when building.

#[cfg(feature = "vkfft")]
use cc;
#[cfg(feature = "cufft")]
use cuda_setup::build_host;
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

    // cuFFT-specifical host-side building
    #[cfg(feature = "cufft")]
    build_host(
        // Select the min supported GPU architecture.
        GpuArchitecture::Rtx3,
        &["src/cuda/cufft.cu", "src/cuda/kernels.cu"],
        "spme",
    );

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
