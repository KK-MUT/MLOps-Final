use pyo3::prelude::*;

#[pyfunction]
fn say_hello() -> PyResult<String> {
    Ok("Hello from Rust!".to_string())
}

#[pymodule]
fn arisa_dsml(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(say_hello, m)?)?;
    Ok(())
}