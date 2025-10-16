// Used by both cuFFT and vkFFT.

use std::ffi::c_void;

unsafe extern "C" {
    pub(crate) fn apply_ghat_and_grad_launch(
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

    pub(crate) fn energy_half_spectrum_launch(
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
}
