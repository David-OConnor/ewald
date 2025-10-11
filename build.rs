//! We use this to automatically compile CUDA C++ code when building.

use cc;
#[cfg(feature = "cuda")]
use cuda_setup::{GpuArchitecture, build_host};

fn main() {
    // For cuFFT
    #[cfg(feature = "cuda")]
    build_host(
        // Select the min supported GPU architecture.
        GpuArchitecture::Rtx3,
        &["src/cuda/spme.cu"],
        "spme", // This name is currently hard-coded in the Ewald lib.
    );

    // For vkFFT.
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
        // Optional: silence some noisy MSVC warnings coming from VkFFT headers
        // .flag_if_supported("/wd4244")
        // .flag_if_supported("/wd4189")
        // .flag_if_supported("/wd4996")
        .warnings(false)
        .compile("vk_fft");

    // todo: Which? OS-dependent.
    println!("cargo:rustc-link-lib=nvcuda");
    println!("cargo:rustc-link-lib=cudart"); // todo: Is this required?
}
