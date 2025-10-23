//! We use this to automatically compile CUDA C++ code when building.

#[cfg(feature = "vkfft")]
use cc;
#[cfg(feature = "cuda")]
use cuda_setup::{GpuArchitecture, build_host, build_ptx};

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

        cc::Build::new()
            .cuda(true)
            .files(["src/cuda/vk_fft.cu"])
            .define("VKFFT_BACKEND", Some("1")) //  Sets the backend to CUDA
            // .define("_CRT_SECURE_NO_WARNINGS", None)
            // .include("src/cuda")
            .include("third_party/VkFFT/vkFFT")
            .flag_if_supported("-O3")
            // .warnings(false)
            .compile("vk_fft");
    }
}
