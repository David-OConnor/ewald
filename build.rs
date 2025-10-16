//! We use this to automatically compile CUDA C++ code when building.

#[cfg(feature = "vkfft")]
use cc;
#[cfg(feature = "cuda")]
use cuda_setup::{GpuArchitecture, build_host};

fn main() {
    // todo: You may need to split the GPU charge spread to a differnet module,
    // todo: So we are not bringing in the cuFFT requirement for the vkFFT branch.
    #[cfg(feature = "cufft")]
    build_host(
        // Select the min supported GPU architecture.
        GpuArchitecture::Rtx3,
        &["src/cuda/spme_cufft.cu", "src/cuda/shared.cu"],
        "spme",
    );

    #[cfg(feature = "vkfft")]
    {
        build_host(GpuArchitecture::Rtx3, &["src/cuda/shared.cu"], "spme");

        println!("cargo:rerun-if-changed=src/cuda/vk_fft.cu");
        println!("cargo:rerun-if-changed=src/cuda/vk_fft.h");
        // This is the vkFFT header
        println!("cargo:rerun-if-changed=third_party/VkFFT/vkFFT.h");

        cc::Build::new()
            .cuda(true)
            .files(["src/cuda/vk_fft.cu"])
            .define("VKFFT_BACKEND", Some("1")) //  Sets the backend to CUDA
            // .define("_CRT_SECURE_NO_WARNINGS", None)
            .include("src")
            .include("third_party/VkFFT/vkFFT")
            .flag_if_supported("-O3")
            // .warnings(false)
            .compile("vk_fft");
    }
}
