use ewald_rs;
use lin_alg::f64::Vec3;
use pyo3::prelude::*;

#[pyclass]
struct PmeRecip {
    inner: ewald_rs::PmeRecip,
}

#[pymethods]
impl PmeRecip {
    #[new]
    fn new(n: (usize, usize, usize), l: (f64, f64, f64), alpha: f64) -> Self {
        Self {
            inner: ewald_rs::PmeRecip::new(n, l, alpha),
        }
    }

    fn forces(&mut self, posits: Vec<[f64; 3]>, q: Vec<f64>) -> (Vec<[f64; 3]>, f64) {
        let posits: Vec<_> = posits.iter().map(|p| Vec3::from_slice(p).unwrap()).collect();
        let (f, e) = self.inner.forces(&posits, &q);

        let f = f.iter().map(|r| r.to_arr()).collect();

        (f, e)
    }

    // todo: forces_gpu A/R.

    fn __repr__(&self) -> String {
        String::from("PME Recip data")
        // No rust debug, due to the FftPlanner.
        // format!("{:?}", self.inner)
    }
}

#[pyfunction]
fn force_coulomb_short_range(
    dir: [f64; 3],
    dist: f64,
    inv_dist: f64,
    q_0: f64,
    q_1: f64,
    cutoff_dist: f64,
    alpha: f64,
) -> ([f64; 3], f64) {
    let dir = Vec3::from_slice(&dir).unwrap();
    let result = ewald_rs::force_coulomb_short_range(
        dir,
        dist,
        inv_dist,
        q_0,
        q_1,
        cutoff_dist,
        alpha,
    );

    (result.0.to_arr(), result.1)
}

#[pymodule]
fn ewald(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PmeRecip>()?;
    m.add_function(wrap_pyfunction!(force_coulomb_short_range, m)?)?;

    Ok(())
}
