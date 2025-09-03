//! We use *host-side* CUDA functions for long-range reciprical FFTs; not just a device kernel.
//! (We recommend you use a kernel that combines the short-range Coulomb force with Lennard Jones logic
//! in your application, instead of using the CPU functionality in this library).

use std::ffi::c_void;

use rustfft::num_complex::Complex;

unsafe extern "C" {
    pub(crate) fn spme_make_plan_c2c(
        nx: i32,
        ny: i32,
        nz: i32,
        cu_stream: *mut c_void, // CUstream / cudaStream_t
    ) -> *mut c_void;

    pub(crate) fn spme_exec_inverse_3_c2c(
        plan: *mut c_void,
        exk: *mut c_void, // cufftComplex*
        eyk: *mut c_void,
        ezk: *mut c_void,
    );

    pub(crate) fn spme_scale_c2c(
        data: *mut c_void, // cufftComplex*
        n: usize,          // number of complex elements (nx*ny*nz)
        scale: f32,
        cu_stream: *mut c_void,
    );

    pub(crate) fn spme_destroy_plan(plan: *mut c_void);
}

/// For CUDA serialization
pub(crate) fn flatten_cplx_vec(v: &[Complex<f64>]) -> Vec<f32> {
    let mut result = Vec::with_capacity(v.len() * 2);

    for v_ in v {
        result.push(v_.re as f32);
        result.push(v_.im as f32);
    }

    result
}

/// For CUDA deserialization
pub(crate) fn unflatten_cplx_vec(v: &[f32]) -> Vec<Complex<f64>> {
    let mut result = Vec::with_capacity(v.len() / 2);

    for i in 0..v.len() / 2 {
        result.push(Complex::new(v[i * 2] as f64, v[i * 2 + 1] as f64));
    }

    result
}
