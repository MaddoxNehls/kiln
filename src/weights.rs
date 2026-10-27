//! Loading tensors from a memory-mapped `.safetensors` file.
//!
//! A safetensors file is an 8-byte header length, a JSON header (name ->
//! dtype, shape, byte range), then one flat byte buffer. Memory-mapping it means
//! the OS pages weights in on demand straight from the page cache, without a
//! read() into a temporary 270 MB buffer. Kiln then widens bf16 -> f32 into
//! owned memory; a later bf16/quantized kernel could compute on the mapping
//! directly with zero copies.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::tensor::Matrix;

pub struct WeightFile {
    mmap: Mmap,
}

impl WeightFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the mapping is read-only and lives as long as `self`. The usual
        // caveat applies: another process truncating the file while it is mapped
        // would be undefined behavior, as with any mmap of a shared file.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("memory-mapping {}", path.display()))?;
        Ok(Self { mmap })
    }

    pub fn tensors(&self) -> Result<Tensors<'_>> {
        Ok(Tensors(
            SafeTensors::deserialize(&self.mmap).context("parsing safetensors header")?,
        ))
    }
}

pub struct Tensors<'a>(SafeTensors<'a>);

impl Tensors<'_> {
    /// Load a tensor as f32, checking it has exactly `shape`.
    pub fn load(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let view = self
            .0
            .tensor(name)
            .with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            view.shape() == shape,
            "{name}: expected shape {shape:?}, found {:?}",
            view.shape()
        );
        let bytes = view.data();
        Ok(match view.dtype() {
            // Little-endian per the format spec. Byte-wise decoding also sidesteps
            // alignment: a tensor's offset in the file is not 4-byte aligned in general.
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect(),
            // bf16 is the top half of an f32 (same 8-bit exponent), so widening
            // is exact: shift the bits left by 16.
            Dtype::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| bf16::from_le_bytes(b).to_f32())
                .collect(),
            Dtype::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| f16::from_le_bytes(b).to_f32())
                .collect(),
            other => bail!("{name}: unsupported dtype {other:?}"),
        })
    }

    pub fn load_matrix(&self, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
        Ok(Matrix::new(self.load(name, &[rows, cols])?, rows, cols))
    }
}
