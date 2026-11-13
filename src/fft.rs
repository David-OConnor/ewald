//! CPU FFT setup

#[cfg(feature = "cuda")]
use std::{ffi::c_void, sync::Arc};

#[cfg(feature = "cuda")]
use cudarc::driver::CudaStream;
use rayon::prelude::*;
use realfft::RealFftPlanner;
use rustfft::FftPlanner;

use crate::Complex_;
// #[cfg(feature = "cuda")]
// use crate::gpu_shared::cuda_slice_to_ptr_mut;

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
    let mut real_planner = RealFftPlanner::new();
    fft3d_r2c_with_planners(data_r, dims, planner, &mut real_planner)
}

/// Internal variant that reuses both complex and real FFT plans across PME steps.
pub(crate) fn fft3d_r2c_with_planners(
    data_r: &mut [f32],
    dims: (usize, usize, usize),
    planner: &mut FftPlanner<f32>,
    real_planner: &mut RealFftPlanner<f32>,
) -> Vec<Complex_> {
    let mut out = Vec::new();
    fft3d_r2c_into(data_r, dims, planner, real_planner, &mut out);
    out
}

pub(crate) fn fft3d_r2c_into(
    data_r: &mut [f32],
    dims: (usize, usize, usize),
    planner: &mut FftPlanner<f32>,
    real_planner: &mut RealFftPlanner<f32>,
    out: &mut Vec<Complex_>,
) {
    let (nx, ny, nz) = dims;

    let nzc = nz / 2 + 1;
    let n_cplx = nx * ny * nzc;

    let r2c_z = real_planner.plan_fft_forward(nz);

    let fft_y = planner.plan_fft_forward(ny);
    let fft_x = planner.plan_fft_forward(nx);

    out.resize(n_cplx, Complex_::new(0.0, 0.0));
    data_r
        .par_chunks_mut(nz)
        .zip(out.par_chunks_mut(nzc))
        .with_min_len(16)
        .for_each_init(
            || r2c_z.make_scratch_vec(),
            |scratch, (input, output)| {
                r2c_z.process_with_scratch(input, output, scratch).unwrap();
            },
        );
    transform_y(out, nx, ny, nzc, &fft_y);
    transform_x(out, nx, ny, nzc, &fft_x);
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
    let mut real_planner = RealFftPlanner::new();
    fft3d_c2r_with_planners(data_k, dims, planner, &mut real_planner)
}

/// Internal variant that reuses both complex and real FFT plans across PME steps.
pub(crate) fn fft3d_c2r_with_planners(
    data_k: &mut [Complex_],
    dims: (usize, usize, usize),
    planner: &mut FftPlanner<f32>,
    real_planner: &mut RealFftPlanner<f32>,
) -> Vec<f32> {
    let mut out = Vec::new();
    fft3d_c2r_into(data_k, dims, planner, real_planner, &mut out);
    out
}

pub(crate) fn fft3d_c2r_into(
    data_k: &mut [Complex_],
    dims: (usize, usize, usize),
    planner: &mut FftPlanner<f32>,
    real_planner: &mut RealFftPlanner<f32>,
    out: &mut Vec<f32>,
) {
    let (nx, ny, nz) = dims;
    let nzc = nz / 2 + 1;

    let c2r_z = real_planner.plan_fft_inverse(nz);

    let ifft_y = planner.plan_fft_inverse(ny);
    let ifft_x = planner.plan_fft_inverse(nx);

    transform_x(data_k, nx, ny, nzc, &ifft_x);
    transform_y(data_k, nx, ny, nzc, &ifft_y);

    out.resize(nx * ny * nz, 0.0);
    data_k
        .par_chunks_mut(nzc)
        .zip(out.par_chunks_mut(nz))
        .with_min_len(16)
        .for_each_init(
            || c2r_z.make_scratch_vec(),
            |scratch, (input, output)| {
                input[0].im = 0.0;
                if nz % 2 == 0 {
                    input[nzc - 1].im = 0.0;
                }

                c2r_z.process_with_scratch(input, output, scratch).unwrap();
            },
        );
}

fn transform_y(
    data: &mut [Complex_],
    nx: usize,
    ny: usize,
    nzc: usize,
    fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
) {
    assert_eq!(data.len(), nx * ny * nzc);
    data.par_chunks_mut(ny * nzc).for_each_init(
        || {
            (
                vec![Complex_::default(); ny],
                vec![Complex_::default(); fft.get_inplace_scratch_len()],
            )
        },
        |(column, scratch), plane| {
            for z in 0..nzc {
                for y in 0..ny {
                    column[y] = plane[y * nzc + z];
                }
                fft.process_with_scratch(column, scratch);
                for y in 0..ny {
                    plane[y * nzc + z] = column[y];
                }
            }
        },
    );
}

fn transform_x(
    data: &mut [Complex_],
    nx: usize,
    ny: usize,
    nzc: usize,
    fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
) {
    let stride = ny * nzc;
    let mut columns = vec![Complex_::default(); data.len()];
    columns.par_chunks_mut(nx).enumerate().for_each_init(
        || vec![Complex_::default(); fft.get_inplace_scratch_len()],
        |scratch, (yz, column)| {
            for x in 0..nx {
                column[x] = data[x * stride + yz];
            }
            fft.process_with_scratch(column, scratch);
        },
    );
    data.par_chunks_mut(stride)
        .enumerate()
        .for_each(|(x, plane)| {
            for yz in 0..stride {
                plane[yz] = columns[yz * nx + x];
            }
        });
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

// Reports whether the cuFFT shared library could be loaded. It is looked up at runtime rather than
// linked, so that a `cufft`-enabled binary still starts where CUDA is absent; see `cufft.cu`.
#[cfg(feature = "cufft")]
unsafe extern "C" {
    pub(crate) fn gpu_fft_available() -> i32;
}

#[cfg(feature = "cuda")]
// FFI for GPU FFT functions. These signatures are the same for cuFFT and vkFFT, so we use
// them for both.
unsafe extern "C" {
    pub(crate) fn make_plan(nx: i32, ny: i32, nz: i32, cu_stream: *mut c_void) -> *mut c_void;

    pub(crate) fn destroy_plan(plan: *mut c_void);

    /// A forward real-to-complex FFT, using cuFFT or vkFFT.
    pub(crate) fn exec_forward(plan: *mut c_void, rho_real: *mut c_void, rho: *mut c_void);

    pub(crate) fn exec_inverse(plan: *mut c_void, ek: *mut c_void, e: *mut c_void);
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
