mod fbuf;

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, ErrorKind, Read, Seek, SeekFrom};

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
        lsebuf[start..n].copy_from_slice(&bscores[start + 1..n + 1]);
        let mut a = read_i32(&mut dag_reader, sz)?;
        while a != -1 {
            let j = a as usize;
            let v_idx = read_i32(&mut dag_reader, sz)? as usize;
            lsebuf[j - 1] += hat_logprobs[v_idx];
            a = read_i32(&mut dag_reader, sz)?;
        }
        bscores[start] = logsumexp(&lsebuf[start..n]);
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

fn compute_values(
    hat_logprobs: &[f64],
    dag_name: &str,
    offset: usize,
    mut sz: usize,
) -> io::Result<(Vec<f64>, f64, f64)> {
    let dag_path = format!("{dag_name}.dag");
    let mut values = crate::Values::zeros(hat_logprobs.len());
    let file = File::open(dag_path)?;
    if sz == 0 {
        let file_len = file.metadata()?.len();
        sz = (file_len as usize).saturating_sub(offset);
    }
    let mut dag_reader = BufReader::new(file);
    dag_reader.seek(SeekFrom::Start(offset as u64))?;
    while sz > 0 {
        crate::update_values_once(
            &mut values,
            &mut dag_reader,
            &hat_logprobs,
            &mut sz,
        )?;
    }
    Ok((
        values.expected_word_counts,
        values.expected_seqlen,
        values.sent_count,
    ))
}

fn read_i64(mut reader: impl Read) -> io::Result<i64> {
    let mut buf1 = [0u8; std::mem::size_of::<i64>()];
    reader.read_exact(&mut buf1)?;
    Ok(i64::from_ne_bytes(buf1))
}

fn get_splits(
    dag_name: &str,
    n_jobs: u32,
) -> io::Result<(Vec<usize>, Vec<usize>)> {
    let offset_path = format!("{dag_name}.dag.offset");
    let mut file = File::open(offset_path)?;
    const DATA_SIZE: u64 = std::mem::size_of::<i64>() as u64;
    let num_sentences = file.metadata()?.len() / DATA_SIZE as u64;
    let step = (num_sentences as f32 / n_jobs as f32 + 1.0) as u64;
    let mut i = 0;
    let mut offsets = Vec::new();
    while i < num_sentences {
        file.seek(SeekFrom::Start(i * DATA_SIZE))?;
        offsets.push(read_i64(&mut file)? as usize);
        i += step;
    }
    let n_jobs = offsets.len();
    let mut sizes = vec![0usize; n_jobs];
    for i in 0..n_jobs - 1 {
        sizes[i] = offsets[i + 1] - offsets[i];
    }
    Ok((offsets, sizes))
}

fn read_i32_unchecked(reader: impl BufRead) -> io::Result<i32> {
    let mut sz = std::mem::size_of::<i32>();
    read_i32(reader, &mut sz)
}

fn read_i32_unchecked_opt(reader: impl BufRead) -> io::Result<Option<i32>> {
    match read_i32_unchecked(reader) {
        Ok(v) => Ok(Some(v)),
        Err(err) => match err.kind() {
            ErrorKind::UnexpectedEof => Ok(None),
            _ => Err(err),
        },
    }
}

fn suggest_tau(dag_name: &str) -> io::Result<f64> {
    let dag_path = format!("{dag_name}.dag");
    let file = File::open(dag_path)?;
    let mut dag_reader = BufReader::new(file);
    let mut n_hist: HashMap<i32, u32> = HashMap::new();
    let mut n_cnt: u64 = 0;
    loop {
        let n = match read_i32_unchecked_opt(&mut dag_reader)? {
            Some(n) => n,
            None => break,
        };
        for _ in 0..n {
            let mut a = read_i32_unchecked(&mut dag_reader)?;
            while a != -1 {
                a = read_i32_unchecked(&mut dag_reader)?;
            }
        }
        let _n = read_i32_unchecked(&mut dag_reader)?;
        assert_eq!(n, _n);
        for _ in 0..n {
            let mut a = read_i32_unchecked(&mut dag_reader)?;
            while a != -1 {
                a = read_i32_unchecked(&mut dag_reader)?;
            }
        }
        n_hist.entry(n).and_modify(|x| *x += 1).or_insert(1);
        n_cnt += 1;
    }
    let mut median_position = n_cnt / 2;
    let mut median_bin = 0;
    let mut keys: Vec<_> = n_hist.keys().collect();
    keys.sort();
    for k in keys {
        let bin_cnt = n_hist[k] as u64;
        if median_position < bin_cnt {
            median_bin = *k;
            break;
        }
        median_position -= bin_cnt;
    }
    let tau = 1.0 / (median_bin as f64 + 1.0);
    Ok(tau)
}

/// A Python module implemented in Rust. The name of this module must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
mod _core {
    use pyo3::exceptions::PyIOError;
    use pyo3::prelude::*;
    use rayon::prelude::*;

    /// Suggest an initial tau based on median sentence length. In general the
    /// initial tau should be no smaller than this value.
    #[pyfunction]
    fn suggest_tau(dag_name: &str) -> PyResult<f64> {
        crate::suggest_tau(dag_name).map_err(PyIOError::new_err)
    }

    /// Compute the values (expected_counts, expected_seqlen, sent_count) from
    /// dag_file[offset..offset + sz] where `offset` and `sz` are measured in
    /// bytes. The values are initialized to zero. If `sz` is 0, read till end.
    #[pyfunction]
    #[pyo3(signature = (hat_logprobs, dag_name, offset = 0, sz = 0))]
    fn compute_values(
        hat_logprobs: Vec<f64>,
        dag_name: &str,
        offset: usize,
        sz: usize,
    ) -> PyResult<(Vec<f64>, f64, f64)> {
        crate::compute_values(&hat_logprobs, dag_name, offset, sz)
            .map_err(PyIOError::new_err)
    }

    #[pyfunction]
    #[pyo3(signature = (hat_logprobs, dag_name, n_jobs = 1))]
    fn compute_values_parallel(
        hat_logprobs: Vec<f64>,
        dag_name: &str,
        n_jobs: u32,
    ) -> PyResult<(Vec<f64>, f64, f64)> {
        let (offsets, sizes) = if n_jobs <= 1 {
            (vec![0], vec![0])
        } else {
            crate::get_splits(dag_name, n_jobs)?
        };
        let splits: Vec<_> = offsets.into_iter().zip(sizes).collect();
        let mut values = crate::Values::zeros(hat_logprobs.len());
        let results: Vec<_> = splits
            .par_iter()
            .map(|(o, s)| {
                crate::compute_values(&hat_logprobs, dag_name, *o, *s)
            })
            .collect();
        for res in results {
            let (word_counts, seqlen, cnt) = res?;
            for (cum, wc) in
                values.expected_word_counts.iter_mut().zip(word_counts)
            {
                *cum += wc;
            }
            values.expected_seqlen += seqlen;
            values.sent_count += cnt;
        }
        Ok((
            values.expected_word_counts,
            values.expected_seqlen,
            values.sent_count,
        ))
    }
}
