//! We use this to automatically compile CUDA C++ code when building.

#[cfg(feature = "vkfft")]
use std::{
    env, fs,
    path::{Path, PathBuf},
};

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

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        // We set up a location to cache the compiled VkFFT library, as the location Cargo places
        // it in by default changes each build.
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let target_dir = manifest_dir.join("target");
        let cache_dir = target_dir.join("vkfft-cache");
        let _ = fs::create_dir_all(&cache_dir);

        // Observation: On Windows, I'm setting both of these files generated.
        let cache_lib_unix = cache_dir.join("libvk_fft.a");
        let cache_lib_win = cache_dir.join("vk_fft.lib");

        let inputs = [
            "src/cuda/vk_fft.cu",
            "src/cuda/vk_fft.h",
            "third_party/VkFFT/vkFFT/vkFFT.h",
        ];

        let needs_rebuild = cache_is_stale(&cache_lib_unix, &cache_lib_win, &inputs);

        println!("cargo:warning=Needs build {needs_rebuild}");

        if needs_rebuild {
            cc::Build::new()
                .cuda(true)
                .files(["src/cuda/vk_fft.cu"])
                .define("VKFFT_BACKEND", Some("1")) //  Sets the backend to CUDA
                .include("third_party/VkFFT/vkFFT")
                .flag_if_supported("-O3")
                .compile("vk_fft");

            // copy from OUT_DIR to stable cache so next build is fast
            let built_unix = out_dir.join("libvk_fft.a");
            let built_win = out_dir.join("vk_fft.lib");

            if built_unix.exists() {
                let _ = fs::copy(&built_unix, &cache_lib_unix);
            }
            if built_win.exists() {
                let _ = fs::copy(&built_win, &cache_lib_win);
            }
        } else {
            // cache is good: copy FROM cache TO OUT_DIR so the linker sees it
            let dst_unix = out_dir.join("libvk_fft.a");
            let dst_win = out_dir.join("vk_fft.lib");

            if cache_lib_unix.exists() {
                let _ = fs::copy(&cache_lib_unix, &dst_unix);
            }
            if cache_lib_win.exists() {
                let _ = fs::copy(&cache_lib_win, &dst_win);
            }
        }
    }
}

/// Used to determine if we need to rebuild the vkFFT library. E.g. on a fresh git clone or
/// install from crates.io.
#[cfg(feature = "vkfft")]
fn cache_is_stale(vkfft_cache_unix: &Path, vkfft_cache_win: &Path, inputs: &[&str]) -> bool {
    let cache_path = if vkfft_cache_unix.exists() || !vkfft_cache_win.exists() {
        vkfft_cache_unix
    } else {
        vkfft_cache_win
    };

    if !vkfft_cache_unix.exists() && !vkfft_cache_win.exists() {
        return true;
    }

    let cache_meta = match fs::metadata(cache_path) {
        Ok(m) => m,
        Err(_) => return true,
    };

    let cache_time = match cache_meta.modified() {
        Ok(t) => t,
        Err(_) => return true,
    };

    for inp in inputs {
        let meta = match fs::metadata(inp) {
            Ok(m) => m,
            Err(_) => return true,
        };
        let mt = match meta.modified() {
            Ok(t) => t,
            Err(_) => return true,
        };
        if mt > cache_time {
            return true;
        }
    }

    false
}
