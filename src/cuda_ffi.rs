//! We use *host-side* CUDA functions for long-range reciprical FFTs; not just a device kernel.
//! (We recommend you use a kernel that combines the short-range Coulomb force with Lennard Jones logic
//! in your application, instead of using the CPU functionality in this library).

// todo: Organize both this and teh .cu file. REmove unused, make order sensitible, and cyn order.

use std::ffi::c_void;

use rustfft::num_complex::Complex;

unsafe extern "C" {
    pub(crate) fn spme_make_plan_r2c_c2r_many(
        nx: i32,
        ny: i32,
        nz: i32,
        cu_stream: *mut c_void, // CUstream / cudaStream_t
    ) -> *mut c_void;

    pub(crate) fn spme_exec_inverse_ExEyEz_c2r(
        plan: *mut c_void,
        exk: *mut c_void, // cufftComplex*
        eyk: *mut c_void,
        ezk: *mut c_void,
        ex: *mut c_void,
        ey: *mut c_void,
        ez: *mut c_void,
    );

    pub(crate) fn spme_scale_ExEyEz_after_c2r(
        ex: *mut c_void,
        ey: *mut c_void,
        ez: *mut c_void,
        nx: i32,
        ny: i32,
        nz: i32,
        cu_stream: *mut c_void,
    );

    pub(crate) fn spme_apply_ghat_and_grad_launch(
        rho: *const c_void,
        exk: *mut c_void,
        eyk: *mut c_void,
        ezk: *mut c_void,
        kx: *const c_void,
        ky: *const c_void,
        kz: *const c_void,
        bx: *const c_void,
        by: *const c_void,
        bz: *const c_void,
        nx: i32,
        ny: i32,
        nz: i32,
        vol: f32,
        alpha: f32,
        cu_stream: *mut c_void,
    );

    pub(crate) fn spme_destroy_plan_r2c_c2r_many(plan: *mut c_void);

    pub(crate) fn spme_energy_half_spectrum_launch(
        rho_k: *const c_void,
        kx: *const c_void,
        ky: *const c_void,
        kz: *const c_void,
        bx: *const c_void,
        by: *const c_void,
        bz: *const c_void,
        nx: i32,
        ny: i32,
        nz: i32,
        vol: f32,
        alpha: f32,
        partial_sums: *mut c_void, // device buffer double[blocks]
        blocks: i32,
        threads: i32,
        cu_stream: *mut c_void,
    );

    pub(crate) fn spme_scatter_rho_4x4x4_launch(
        pos: *const c_void,
        q: *const c_void,
        rho: *mut c_void,
        n_atoms: i32,
        nx: i32,
        ny: i32,
        nz: i32,
        lx: f32,
        ly: f32,
        lz: f32,
        cu_stream: *mut c_void,
    );

    pub(crate) fn spme_exec_forward_r2c(
        plan: *mut c_void,
        rho_real: *mut c_void,
        rho_k: *mut c_void,
    );

    pub(crate) fn spme_gather_forces_to_atoms_launch(
        pos: *const c_void,
        ex: *const c_void,
        ey: *const c_void,
        ez: *const c_void,
        q: *const c_void,
        out_f: *mut c_void,
        n_atoms: i32,
        nx: i32,
        ny: i32,
        nz: i32,
        lx: f32,
        ly: f32,
        lz: f32,
        inv_n: f32,
        cu_stream: *mut c_void,
    );
}

/// For CUDA serialization
pub(crate) fn flatten_cplx_vec(v: &[Complex<f32>]) -> Vec<f32> {
    let mut result = Vec::with_capacity(v.len() * 2);

    for v_ in v {
        result.push(v_.re);
        result.push(v_.im);
    }

    result
}

/// For CUDA deserialization
pub(crate) fn unflatten_cplx_vec(v: &[f32]) -> Vec<Complex<f32>> {
    let mut result = Vec::with_capacity(v.len() / 2);

    for i in 0..v.len() / 2 {
        result.push(Complex::new(v[i * 2], v[i * 2 + 1]));
    }

    result
}
