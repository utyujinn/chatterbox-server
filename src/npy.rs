// A reader for NumPy `.npy` files (the golden data of the tests and the mel filterbank): little endian f32, f64, i64 and i32 arrays.

use anyhow::{anyhow, bail, ensure, Context, Result};
use std::path::Path;

#[derive(Debug)]
pub enum Data {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
    I32(Vec<i32>),
}

#[derive(Debug)]
pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Data,
}

pub fn read(path: &Path) -> Result<Npy> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    parse(&bytes).with_context(|| format!("parsing {}", path.display()))
}

pub fn parse(bytes: &[u8]) -> Result<Npy> {
    ensure!(bytes.len() > 10 && &bytes[..6] == b"\x93NUMPY", "not a .npy file");
    let (header_len, header_start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
        v => bail!("unsupported .npy version {v}"),
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])?;
    ensure!(header.contains("'fortran_order': False"), "Fortran order is not supported");
    let descr = header.split("'descr':").nth(1).and_then(|r| r.split('\'').nth(1)).ok_or_else(|| anyhow!("no descr"))?;
    let shape_text = header.split("'shape':").nth(1).and_then(|r| r.split('(').nth(1)).and_then(|r| r.split(')').next()).ok_or_else(|| anyhow!("no shape"))?;
    let shape: Vec<usize> = shape_text.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::parse).collect::<Result<_, _>>()?;
    let count: usize = shape.iter().product();
    let body = &bytes[header_start + header_len..];
    let data = match descr {
        "<f4" => Data::F32(body.chunks_exact(4).take(count).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()),
        "<f8" => Data::F64(body.chunks_exact(8).take(count).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect()),
        "<i8" => Data::I64(body.chunks_exact(8).take(count).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect()),
        "<i4" => Data::I32(body.chunks_exact(4).take(count).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()),
        other => bail!("unsupported dtype {other}"),
    };
    let have = match &data {
        Data::F32(v) => v.len(),
        Data::F64(v) => v.len(),
        Data::I64(v) => v.len(),
        Data::I32(v) => v.len(),
    };
    ensure!(have == count, "the file holds {have} values, the shape wants {count}");
    Ok(Npy { shape, data })
}

impl Npy {
    pub fn f32(self) -> Result<Vec<f32>> {
        match self.data {
            Data::F32(v) => Ok(v),
            _ => bail!("not an f32 array"),
        }
    }
    pub fn f64(self) -> Result<Vec<f64>> {
        match self.data {
            Data::F64(v) => Ok(v),
            _ => bail!("not an f64 array"),
        }
    }
    pub fn i64(self) -> Result<Vec<i64>> {
        match self.data {
            Data::I64(v) => Ok(v),
            _ => bail!("not an i64 array"),
        }
    }
}
