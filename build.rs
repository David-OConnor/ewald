//! We use this to automatically compile CUDA C++ code when building.

use cc;
#[cfg(feature = "cuda")]
use cuda_setup::{GpuArchitecture, build_host};

fn main() {
    #[cfg(feature = "cuda")]
    build_host(
        // Select the min supported GPU architecture.
        GpuArchitecture::Rtx3,
        &["src/cuda/spme.cu"],
        "spme", // This name is currently hard-coded in the Ewald lib.
    );

    // If using VKFFT:
    // todo: Rerun if .h changed too?
    println!("cargo:rerun-if-changed=wrapper/vk_fft.c");
    cc::Build::new()
        .cuda(true)
        .files(["vk_fft.c"])
        .define("VKFFT_BACKEND", Some("2")) //  Sets teh VKFFT_BACKEND_CUDA backend.
        .include("c")
        // .warnings(false)
        .flag_if_supported("-O3")
        .compile("vk_fft");

    println!("cargo:rustc-link-lib=cuda");
}
