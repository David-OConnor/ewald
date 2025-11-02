//! CPU FFT setup

#[cfg(feature = "cuda")]
use std::{
    sync::Arc,
    ffi::c_void,
};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream};
use realfft::RealFftPlanner;
use rustfft::FftPlanner;

use crate::Complex_;
#[cfg(feature = "cuda")]
use crate::gpu_shared::cuda_slice_to_ptr_mut;

/// Real-to-Complex forward 3D FFT. This approach uses less memory, and is probably faster,
/// than using complex to complex transform (Factor of 2 for the memory).
///
/// Z is the contiguous (fast) dimension; X is the strided (slow) one. This is chosen
/// to be consistent with cuFFT's `Plan3D`'s conventions.
pub fn fft3d_r2c(
    data_r: &mut [f32],
    dims: (usize, usize, usize),
    planner: &mut FftPlanner<f32>,
) -> Vec<Complex_> {
    let (nx, ny, nz) = dims;

    let nzc = nz / 2 + 1;
    let n_cplx = nx * ny * nzc;

    let mut rplanner = RealFftPlanner::<f32>::new();
    let r2c_z = rplanner.plan_fft_forward(nz);

    let fft_y = planner.plan_fft_forward(ny);
    let fft_x = planner.plan_fft_forward(nx);

    let mut out = vec![Complex_::new(0.0, 0.0); n_cplx];

    // Z: R2C rows (contiguous)
    for ix in 0..nx {
        for iy in 0..ny {
            let row_r = ix * (ny * nz) + iy * nz;
            let row_k = ix * (ny * nzc) + iy * nzc;
            let in_row = &mut data_r[row_r..row_r + nz];
            let out_row = &mut out[row_k..row_k + nzc];
            r2c_z.process(in_row, out_row).unwrap();
        }
    }

    // Y: C2C columns
    {
        let mut tmp = vec![Complex_::new(0.0, 0.0); ny];
        for ix in 0..nx {
            for izc in 0..nzc {
                for (j, iy) in (0..ny).enumerate() {
                    tmp[j] = out[ix * (ny * nzc) + iy * nzc + izc];
                }
                fft_y.process(&mut tmp);
                for (j, iy) in (0..ny).enumerate() {
                    out[ix * (ny * nzc) + iy * nzc + izc] = tmp[j];
                }
            }
        }
    }

    // X: C2C columns (most strided)
    {
        let mut tmp = vec![Complex_::new(0.0, 0.0); nx];
        for iy in 0..ny {
            for izc in 0..nzc {
                for (k, ix) in (0..nx).enumerate() {
                    tmp[k] = out[ix * (ny * nzc) + iy * nzc + izc];
                }
                fft_x.process(&mut tmp);
                for (k, ix) in (0..nx).enumerate() {
                    out[ix * (ny * nzc) + iy * nzc + izc] = tmp[k];
                }
            }
        }
    }

    out
}

/// Complex-to-real inverse 3D FFT.
///
/// Z is the contiguous (fast) dimension; X is the strided (slow) one. This is chosen
/// to be consistent with cuFFT's `Plan3D`'s conventions.
pub fn fft3d_c2r(
    data_k: &mut [Complex_],
    dims: (usize, usize, usize),
    planner: &mut FftPlanner<f32>,
) -> Vec<f32> {
    let (nx, ny, nz) = dims;
    let nzc = nz / 2 + 1;

    let mut planner_real = RealFftPlanner::<f32>::new();
    let c2r_z = planner_real.plan_fft_inverse(nz);

    let ifft_y = planner.plan_fft_inverse(ny);
    let ifft_x = planner.plan_fft_inverse(nx);

    // inverse X: C2C
    {
        let mut tmp = vec![Complex_::new(0.0, 0.0); nx];
        for iy in 0..ny {
            for izc in 0..nzc {
                for (k, ix) in (0..nx).enumerate() {
                    tmp[k] = data_k[ix * (ny * nzc) + iy * nzc + izc];
                }
                ifft_x.process(&mut tmp);
                for (k, ix) in (0..nx).enumerate() {
                    data_k[ix * (ny * nzc) + iy * nzc + izc] = tmp[k];
                }
            }
        }
    }

    // inverse Y: C2C
    {
        let mut tmp = vec![Complex_::new(0.0, 0.0); ny];

        for ix in 0..nx {
            for izc in 0..nzc {
                for (j, iy) in (0..ny).enumerate() {
                    tmp[j] = data_k[ix * (ny * nzc) + iy * nzc + izc];
                }
                ifft_y.process(&mut tmp);
                for (j, iy) in (0..ny).enumerate() {
                    data_k[ix * (ny * nzc) + iy * nzc + izc] = tmp[j];
                }
            }
        }
    }

    // Z: C2R rows (contiguous)
    let mut out = vec![0.; nx * ny * nz];
    for ix in 0..nx {
        for iy in 0..ny {
            let row_k = ix * (ny * nzc) + iy * nzc;
            let row_r = ix * (ny * nz) + iy * nz;
            let in_row = &mut data_k[row_k..row_k + nzc];
            let out_row = &mut out[row_r..row_r + nz];

            // Enforce real-signal constraint at DC / Nyquist along Z
            in_row[0].im = 0.0;
            if nz % 2 == 0 {
                in_row[nzc - 1].im = 0.0;
            }

            c2r_z.process(in_row, out_row).unwrap();
        }
    }

    out
}

// // todo: Experimenting
// #[cfg(feature = "cuda")]
// pub fn fft3d_c2r_gpu(
//     data_k: &mut [Complex_],
//     dims: (usize, usize, usize),
//     stream: &Arc<CudaStream>,
// ) -> Vec<f32> {
//     let ex_dev: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();
//     let ey_dev: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();
//     let ez_dev: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();
//
//     let ex_ptr = cuda_slice_to_ptr_mut(&ex_dev, stream);
//     let ey_ptr = cuda_slice_to_ptr_mut(&ey_dev, stream);
//     let ez_ptr = cuda_slice_to_ptr_mut(&ez_dev, stream);
//
//     unsafe {
//         exec_inverse(
//             data.planner_gpu,
//             ekx_ptr,
//             eky_ptr,
//             ekz_ptr,
//             ex_ptr,
//             ey_ptr,
//             ez_ptr,
//         );
//     }
//
//     let ex = stream.memcpy_dtov(&ex_dev).unwrap();
//     let ey = stream.memcpy_dtov(&ey_dev).unwrap();
//     let ez = stream.memcpy_dtov(&ez_dev).unwrap();
// }

#[cfg(feature = "cuda")]
// FFI for GPU FFT functions. These signatures are the same for cuFFT and vkFFT, so we use
// them for both.
unsafe extern "C" {
    pub(crate) fn make_plan(nx: i32, ny: i32, nz: i32, cu_stream: *mut c_void) -> *mut c_void;

    pub(crate) fn destroy_plan(plan: *mut c_void);

    /// A forward real-to-complex FFT, using cuFFT or vkFFT.
    pub(crate) fn exec_forward(plan: *mut c_void, rho_real: *mut c_void, rho: *mut c_void);

    pub(crate) fn exec_inverse(
        plan: *mut c_void,
        exk: *mut c_void,
        eyk: *mut c_void,
        ezk: *mut c_void,
        ex: *mut c_void,
        ey: *mut c_void,
        ez: *mut c_void,
    );
}

#[cfg(feature = "cuda")]
/// Create the GPU plan. Run this at init, or when dimensions change.
pub(crate) fn create_gpu_plan(
    dims: (usize, usize, usize),
    stream: &Arc<CudaStream>,
) -> *mut c_void {
    let raw_stream: *mut c_void = stream.cu_stream() as *mut c_void;
    let (nx, ny, nz) = (dims.0 as i32, dims.1 as i32, dims.2 as i32);

    unsafe { make_plan(nx, ny, nz, raw_stream) }
}
