//! Experimenting with a lighter, un-license-encumbered FFT library.
//! [VkFFT](https://github.com/DTolm/VkFFT)
//!
//!
use cudarc::driver::{CudaDevice, CudaSlice};

// git submodule add https://github.com/DTolm/VkFFT.git third_party/vkfft

#[repr(C)]
pub struct VkFFTHandle {
    _private: [u8; 0],
}

#[link(name = "vkfft_wrapper")] // built from build.rs
extern "C" {
    fn vkfft_init(handle: *mut VkFFTHandle, size: i32, batch: i32) -> i32;
    fn vkfft_forward(handle: *mut VkFFTHandle, input: u64, output: u64) -> i32;
    fn vkfft_inverse(handle: *mut VkFFTHandle, input: u64, output: u64) -> i32;
    fn vkfft_cleanup(handle: *mut VkFFTHandle);
}

fn run_vkfft() -> Result<(), Box<dyn std::error::Error>> {
    let dev = CudaDevice::new(0)?;

    let n = 1_024;
    let mut data: Vec<f32> = (0..n).map(|x| x as f32).collect();
    let mut d_in = dev.htod_copy(data)?;
    let mut d_out: CudaSlice<f32> = dev.alloc_zeros(n)?;

    unsafe {
        let mut handle = std::mem::zeroed::<VkFFTHandle>();
        let res = vkfft_init(&mut handle, n as i32, 1);
        assert_eq!(res, 0);

        let in_ptr = d_in.as_device_ptr().as_raw() as u64;
        let out_ptr = d_out.as_device_ptr().as_raw() as u64;

        vkfft_forward(&mut handle, in_ptr, out_ptr);

        vkfft_cleanup(&mut handle);
    }

    Ok(())
}