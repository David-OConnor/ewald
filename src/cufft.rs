//! We use *host-side* cuFFT functions for long-range reciprical FFTs. This module contains FFI
//! bindings between the rust code, and cuFFT FFI functions.

// todo: Organize both this and teh .cu file. REmove unused, make order sensitible, and cyn order.

// todo: Rremove the spme_ prefix here and in the modules

use std::{ffi::c_void, sync::Arc};

use cudarc::driver::CudaStream;

use crate::fft::make_plan;
// unsafe extern "C" {
// pub(crate) fn make_plan(nx: i32, ny: i32, nz: i32, cu_stream: *mut c_void) -> *mut c_void;
//
// pub(crate) fn destroy_plan(plan: *mut c_void);
// }

// /// Create the GPU plan. Run this at init, or when dimensions change.
// pub(crate) fn create_gpu_plan(
//     dims: (usize, usize, usize),
//     stream: &Arc<CudaStream>,
// ) -> *mut c_void {
//     let raw_stream: *mut c_void = stream.cu_stream() as *mut c_void;
//     let (nx, ny, nz) = (dims.0 as i32, dims.1 as i32, dims.2 as i32);
//
//     unsafe { make_plan(nx, ny, nz, raw_stream) }
// }
