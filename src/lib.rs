mod fbuf;

use std::io::{self, BufRead};

use pyo3::prelude::*;

use fbuf::F64Buf;

#[inline(always)]
fn max<'a>(x: &'a f64, y: &'a f64) -> &'a f64 {
    if x > y { x } else { y }
}

#[inline]
fn logsumexp(arr: &[f64]) -> f64 {
    let max = arr.iter().reduce(|a, b| max(a, b)).unwrap();
    let sumexp: f64 = arr.iter().map(|a| (a - max).exp()).sum();
    sumexp.ln() + max
}

#[inline(always)]
fn logaddexp(x: &f64, y: &f64) -> f64 {
    max(x, y) + (-(x - y).abs()).exp().ln_1p()
}

#[inline]
fn logaddexp_batch(arr_x: &[f64], arr_y: &[f64], arr_out: &mut [f64]) {
    for (z, (x, y)) in arr_out.iter_mut().zip(arr_x.iter().zip(arr_y)) {
        *z = logaddexp(x, y);
    }
}

struct Values {
    expected_word_counts: Vec<f64>,
    sent_count: f64,
    expected_seqlen: f64,
}

impl Values {
    fn zeros(vocab_size: usize) -> Self {
        Self {
            expected_word_counts: vec![0f64; vocab_size],
            sent_count: 0f64,
            expected_seqlen: 0f64,
        }
    }
}

#[inline]
fn read_i32(mut reader: impl BufRead, sz: &mut usize) -> io::Result<i32> {
    let mut buf1 = [0u8; std::mem::size_of::<i32>()];
    reader.read_exact(&mut buf1)?;
    *sz = sz.checked_sub(std::mem::size_of::<i32>()).unwrap();
    Ok(i32::from_ne_bytes(buf1))
}

fn isclose(a: f64, b: f64) -> bool {
    let rtol = 1e-5;
    let atol = 1e-8;

    (a - b).abs() <= atol + rtol * b.abs()
}

fn update_values_once(
    values: &mut Values,
    mut dag_reader: impl BufRead,
    hat_logprobs: &[f64],
    sz: &mut usize,
) -> io::Result<()> {
    if *sz == 0 {
        return Ok(());
    }

    let n = read_i32(&mut dag_reader, sz)? as usize;
    let mut bscores = F64Buf::zeros(n + 1);
    let mut fscores = F64Buf::zeros(n + 1);
    let mut seqlenbuf = F64Buf::zeros(n + 1);
    let mut lsebuf = F64Buf::zeros(n);
    let mut lsebuf2 = F64Buf::zeros(n);

    // Compute backward scores.
    for start in (0..n).rev() {
        lsebuf[start..].copy_from_slice(&bscores[start + 1..]);
        let mut a = read_i32(&mut dag_reader, sz)?;
        while a != -1 {
            let j = a as usize;
            let v_idx = read_i32(&mut dag_reader, sz)? as usize;
            lsebuf[j - 1] += hat_logprobs[v_idx];
            a = read_i32(&mut dag_reader, sz)?;
        }
        bscores[start] = logsumexp(&lsebuf[start..]);
    }
    let z = bscores[0];

    // Compute forward scores, seqlenbuf, and update expected counts.
    let _n = read_i32(&mut dag_reader, sz)? as usize; // should equal `n`
    assert_eq!(n, _n);
    for end in 1..=n {
        lsebuf[..end].copy_from_slice(&fscores[..end]);
        logaddexp_batch(
            &fscores[..end],
            &seqlenbuf[..end],
            &mut lsebuf2[..end],
        );
        let mut a = read_i32(&mut dag_reader, sz)?;
        while a != -1 {
            let i = a as usize;
            assert!(i < end);
            let v_idx = read_i32(&mut dag_reader, sz)? as usize;
            let lp = hat_logprobs[v_idx];
            lsebuf[i] += lp;
            lsebuf2[i] += lp;
            values.expected_word_counts[v_idx] +=
                (lp + fscores[i] + bscores[end] - z).exp();
            a = read_i32(&mut dag_reader, sz)?;
        }
        fscores[end] = logsumexp(&lsebuf[..end]);
        seqlenbuf[end] = logsumexp(&lsebuf2[..end]);
    }
    assert!(isclose(fscores[n], z));

    values.expected_seqlen += seqlenbuf[n];
    values.sent_count += 1.0;

    Ok(())
}

/// A Python module implemented in Rust. The name of this module must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
mod _core {
    use std::fs::File;
    use std::io::BufReader;

    use pyo3::exceptions::PyIOError;
    use pyo3::prelude::*;

    #[pyfunction]
    fn hello_from_bin() -> String {
        "Hello from jieba-em-train!".to_string()
    }

    /// Compute the values (expected_counts, expected_seqlen, sent_count) from
    /// dag_file[offset..offset + sz] where `offset` and `sz` are measured in
    /// bytes. The values are initialized to zero.
    #[pyfunction]
    fn compute_values(
        dag_path: &str,
        offset: i64,
        mut sz: usize,
        hat_logprobs: Vec<f64>,
    ) -> PyResult<(Vec<f64>, f64, f64)> {
        let mut values = crate::Values::zeros(hat_logprobs.len());
        let mut dag_reader =
            BufReader::new(File::open(dag_path).map_err(PyIOError::new_err)?);
        dag_reader
            .seek_relative(offset)
            .map_err(PyIOError::new_err)?;
        while sz > 0 {
            crate::update_values_once(
                &mut values,
                &mut dag_reader,
                &hat_logprobs,
                &mut sz,
            )
            .map_err(PyIOError::new_err)?;
        }
        Ok((
            values.expected_word_counts,
            values.expected_seqlen,
            values.sent_count,
        ))
    }
}
