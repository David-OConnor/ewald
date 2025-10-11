//! We use this to automatically compile CUDA C++ code when building.

#[cfg(feature = "cufft")]
use cuda_setup::{GpuArchitecture, build_host};

#[cfg(feature = "vkfft")]
use cc;

fn main() {
    #[cfg(feature = "cufft")]
        build_host(
            // Select the min supported GPU architecture.
            GpuArchitecture::Rtx3,
            &["src/cuda/spme.cu"],
            "spme",
        );

    #[cfg(feature = "vkfft")]
    {
        println!("cargo:rerun-if-changed=src/vk_fft.c");
        println!("cargo:rerun-if-changed=src/vk_fft.h");
        // This is the vkFFT header
        println!("cargo:rerun-if-changed=third_party/VkFFT/vkFFT.h");

        cc::Build::new()
            .cuda(true)
            .files(["src/vk_fft.c"])
            .define("VKFFT_BACKEND", Some("1")) //  Sets the backend to CUDA
            .define("_CRT_SECURE_NO_WARNINGS", None)
            .include("src")
            .include("third_party/VkFFT/vkFFT")
            .flag_if_supported("-O3")
            .warnings(false)
            .compile("vk_fft");

        // println!("cargo:rustc-link-lib=nvcuda");
        // println!("cargo:rustc-link-lib=cudart"); // todo: Is this required?
    }
}
