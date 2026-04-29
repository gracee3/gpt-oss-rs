//! SafeTensors GPU loader -- loads weights directly to CUDA device memory.
//!
//! Memory-maps the safetensors file(s), parses the header to find tensor
//! metadata, then uploads each tensor's raw bytes to GPU.
//!
//! Supports two dtype modes:
//! - `GpuDType::F32`: all weights widened to f32 (original path)
//! - `GpuDType::F16`: f16 kept as-is, bf16 narrowed to f16, f32 narrowed to f16
//!   Halves VRAM and enables hgemm.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use gpt_oss_core::error::{LLMError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FilteredTensorDecision {
    record_shape: bool,
    upload: bool,
}

fn filtered_f16_tensor_decision(
    dtype_str: &str,
    tensor_name: &str,
    should_load: &mut impl FnMut(&str) -> bool,
) -> FilteredTensorDecision {
    if dtype_str == "U8" {
        return FilteredTensorDecision {
            record_shape: true,
            upload: false,
        };
    }

    FilteredTensorDecision {
        record_shape: true,
        upload: should_load(tensor_name),
    }
}

/// Load all raw `U8` tensors from safetensors files into host memory.
///
/// GPT-OSS stores MXFP4 expert blocks/scales as `U8`, so these tensors must
/// survive loader setup even though they are not uploaded through the normal
/// f32/f16 weight maps.
pub fn load_u8_weights_to_host(path: &Path) -> Result<HashMap<String, Vec<u8>>> {
    load_u8_weights_to_host_filtered(path, |_| true)
}

/// Load raw `U8` tensors from safetensors files into host memory while allowing
/// callers to skip U8 payloads that are unnecessary for the current shard.
pub fn load_u8_weights_to_host_filtered<F>(
    path: &Path,
    should_load: F,
) -> Result<HashMap<String, Vec<u8>>>
where
    F: FnMut(&str) -> bool,
{
    if path.is_dir() {
        load_sharded_u8_to_host_filtered(path, should_load)
    } else {
        load_single_u8_to_host_filtered(path, should_load)
    }
}

fn load_single_u8_to_host_filtered<F>(
    path: &Path,
    mut should_load: F,
) -> Result<HashMap<String, Vec<u8>>>
where
    F: FnMut(&str) -> bool,
{
    let mut file = std::fs::File::open(path)?;
    let (header, data_start, file_len) = parse_safetensors_header_from_file(&mut file, path)?;
    let mut weights = HashMap::new();

    for (name, meta) in &header {
        if name == "__metadata__" {
            continue;
        }

        let Some((dtype_str, start, end)) = parse_u8_tensor_meta(meta, name, data_start, file_len)?
        else {
            continue;
        };

        if dtype_str == "U8" && should_load(name.as_str()) {
            let mut tensor_bytes = vec![0u8; end - start];
            file.seek(SeekFrom::Start(start as u64))?;
            file.read_exact(&mut tensor_bytes)?;
            weights.insert(name.clone(), tensor_bytes);
        }
    }

    Ok(weights)
}

fn load_sharded_u8_to_host_filtered<F>(
    dir: &Path,
    mut should_load: F,
) -> Result<HashMap<String, Vec<u8>>>
where
    F: FnMut(&str) -> bool,
{
    let shard_files = collect_safetensor_shards(dir)?;
    let mut all_weights = HashMap::new();

    for shard_path in &shard_files {
        all_weights.extend(load_single_u8_to_host_filtered(
            shard_path,
            &mut should_load,
        )?);
    }

    Ok(all_weights)
}

fn collect_safetensor_shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut shard_files: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();
    shard_files.sort();

    if shard_files.is_empty() {
        return Err(LLMError::ModelError(format!(
            "no .safetensors files found in {}",
            dir.display()
        )));
    }

    Ok(shard_files)
}

fn parse_safetensors_header_from_file(
    file: &mut std::fs::File,
    path: &Path,
) -> Result<(HashMap<String, serde_json::Value>, usize, usize)> {
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes).map_err(|e| {
        LLMError::ModelError(format!(
            "failed to read safetensors header length from {}: {e}",
            path.display()
        ))
    })?;

    let header_size = u64::from_le_bytes(header_len_bytes) as usize;
    let file_len = file.metadata()?.len() as usize;
    if 8 + header_size > file_len {
        return Err(LLMError::ModelError(format!(
            "header size exceeds file length in {}",
            path.display()
        )));
    }

    let mut header_bytes = vec![0u8; header_size];
    file.seek(SeekFrom::Start(8))?;
    file.read_exact(&mut header_bytes)?;

    let header_str = std::str::from_utf8(&header_bytes)
        .map_err(|e| LLMError::ModelError(format!("invalid header utf8: {e}")))?;
    let header: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
        .map_err(|e| LLMError::SerializationError(format!("header json: {e}")))?;

    Ok((header, 8 + header_size, file_len))
}

fn parse_u8_tensor_meta<'a>(
    meta: &'a serde_json::Value,
    name: &str,
    data_start: usize,
    file_len: usize,
) -> Result<Option<(&'a str, usize, usize)>> {
    let obj = meta
        .as_object()
        .ok_or_else(|| LLMError::ModelError(format!("tensor {} has non-object meta", name)))?;

    let dtype_str = obj
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| LLMError::ModelError(format!("tensor {} missing dtype", name)))?;
    if dtype_str != "U8" {
        return Ok(None);
    }

    let offsets = obj
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| LLMError::ModelError(format!("tensor {} missing data_offsets", name)))?;
    if offsets.len() != 2 {
        return Err(LLMError::ModelError(format!(
            "tensor {} has {} offsets, expected 2",
            name,
            offsets.len()
        )));
    }

    let start = offsets[0].as_u64().unwrap_or(0) as usize;
    let end = offsets[1].as_u64().unwrap_or(0) as usize;
    if end < start {
        return Err(LLMError::ModelError(format!(
            "tensor {} has reversed data offsets",
            name
        )));
    }

    let abs_start = data_start + start;
    let abs_end = data_start + end;
    if abs_end > file_len {
        return Err(LLMError::ModelError(format!(
            "tensor {} data range [{}, {}) exceeds file size {}",
            name, abs_start, abs_end, file_len
        )));
    }

    Ok(Some((dtype_str, abs_start, abs_end)))
}

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;

    use cudarc::driver::{CudaSlice, CudaStream};
    use gpt_oss_core::error::{LLMError, Result};
    use memmap2::Mmap;
    use tracing::{debug, info};

    /// Target dtype for GPU weight storage.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum GpuDType {
        /// Widen everything to f32 (legacy path).
        F32,
        /// Keep f16 as-is, convert bf16->f16, narrow f32->f16. Halves VRAM.
        F16,
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Load all safetensors weights as f32 (legacy API, unchanged signature).
    pub fn load_weights_to_gpu(
        path: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<HashMap<String, CudaSlice<f32>>> {
        load_weights_to_gpu_with_shapes(path, stream).map(|(weights, _shapes)| weights)
    }

    /// Load all safetensors weights as f32 and preserve tensor shapes.
    pub fn load_weights_to_gpu_with_shapes(
        path: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<(HashMap<String, CudaSlice<f32>>, HashMap<String, Vec<usize>>)> {
        load_weights_to_gpu_with_shapes_filtered(path, stream, |_| true)
    }

    /// Load safetensors weights as f32 while allowing callers to skip uploading
    /// tensors they know are unnecessary on the current path. Shape metadata is
    /// still preserved for skipped tensors so downstream sharding logic can
    /// continue to operate on the full model topology.
    pub fn load_weights_to_gpu_with_shapes_filtered<F>(
        path: &Path,
        stream: &Arc<CudaStream>,
        should_load: F,
    ) -> Result<(HashMap<String, CudaSlice<f32>>, HashMap<String, Vec<usize>>)>
    where
        F: FnMut(&str) -> bool,
    {
        if path.is_dir() {
            load_sharded_to_gpu(path, stream, should_load)
        } else {
            load_single_to_gpu(path, stream, should_load)
        }
    }

    /// Load all safetensors weights as f16 on GPU.
    ///
    /// F16 weights are uploaded directly (zero widen), BF16 are converted to
    /// f16 on the host, and f32 weights are narrowed to f16. This halves VRAM
    /// usage and enables the hgemm (half-precision GEMM) path.
    pub fn load_weights_to_gpu_f16(
        path: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<HashMap<String, CudaSlice<half::f16>>> {
        load_weights_to_gpu_f16_with_shapes(path, stream).map(|(weights, _shapes)| weights)
    }

    /// Load all safetensors weights as f16 on GPU and preserve tensor shapes.
    pub fn load_weights_to_gpu_f16_with_shapes(
        path: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<(
        HashMap<String, CudaSlice<half::f16>>,
        HashMap<String, Vec<usize>>,
    )> {
        load_weights_to_gpu_f16_with_shapes_filtered(path, stream, |_| true)
    }

    /// Load safetensors weights as f16 while allowing callers to skip uploading
    /// tensors they know are unnecessary on the current path. Shape metadata is
    /// still preserved for skipped tensors so future sharded allocation can
    /// inspect the full model topology.
    pub fn load_weights_to_gpu_f16_with_shapes_filtered<F>(
        path: &Path,
        stream: &Arc<CudaStream>,
        should_load: F,
    ) -> Result<(
        HashMap<String, CudaSlice<half::f16>>,
        HashMap<String, Vec<usize>>,
    )>
    where
        F: FnMut(&str) -> bool,
    {
        if path.is_dir() {
            load_sharded_to_gpu_f16(path, stream, should_load)
        } else {
            load_single_to_gpu_f16(path, stream, should_load)
        }
    }

    // -----------------------------------------------------------------------
    // F32 path (unchanged)
    // -----------------------------------------------------------------------

    fn load_single_to_gpu<F>(
        path: &Path,
        stream: &Arc<CudaStream>,
        mut should_load: F,
    ) -> Result<(HashMap<String, CudaSlice<f32>>, HashMap<String, Vec<usize>>)>
    where
        F: FnMut(&str) -> bool,
    {
        info!("gpu_loader: memory-mapping {}", path.display());

        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            LLMError::ModelError(format!("mmap failed for {}: {}", path.display(), e))
        })?;
        let data: &[u8] = &mmap;

        let (header, data_start) = parse_safetensors_header(data, path)?;

        let mut weights: HashMap<String, CudaSlice<f32>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        for (name, meta) in &header {
            if name == "__metadata__" {
                continue;
            }

            let (dtype_str, shape, tensor_bytes) = parse_tensor_meta(meta, name, data, data_start)?;
            let numel: usize = shape.iter().product();

            if dtype_str == "U8" {
                debug!(tensor = name.as_str(), shape = ?shape, "skipping U8 tensor on f32 GPU path");
                shapes.insert(name.clone(), shape);
                continue;
            }

            if !should_load(name.as_str()) {
                debug!(
                    tensor = name.as_str(),
                    dtype = dtype_str,
                    shape = ?shape,
                    "skipping tensor upload on filtered f32 GPU path"
                );
                shapes.insert(name.clone(), shape);
                continue;
            }

            let f32_host = convert_to_f32(tensor_bytes, dtype_str, numel, name)?;

            let gpu_slice = stream.clone_htod(&f32_host).map_err(|e| {
                LLMError::GpuError(format!(
                    "clone_htod failed for tensor {} ({} floats): {}",
                    name,
                    f32_host.len(),
                    e
                ))
            })?;

            debug!(
                tensor = name.as_str(),
                dtype = dtype_str,
                shape = ?shape,
                numel = numel,
                "uploaded tensor to GPU (f32)"
            );

            weights.insert(name.clone(), gpu_slice);
            shapes.insert(name.clone(), shape);
        }

        info!(
            "gpu_loader: loaded {} tensors from {} to GPU (f32)",
            weights.len(),
            path.display()
        );
        Ok((weights, shapes))
    }

    fn load_sharded_to_gpu<F>(
        dir: &Path,
        stream: &Arc<CudaStream>,
        mut should_load: F,
    ) -> Result<(HashMap<String, CudaSlice<f32>>, HashMap<String, Vec<usize>>)>
    where
        F: FnMut(&str) -> bool,
    {
        let shard_files = collect_shards(dir)?;

        info!(
            "gpu_loader: loading {} shards from {} to GPU (f32)",
            shard_files.len(),
            dir.display()
        );

        let mut all_weights: HashMap<String, CudaSlice<f32>> = HashMap::new();
        let mut all_shapes: HashMap<String, Vec<usize>> = HashMap::new();
        for shard_path in &shard_files {
            let (shard_weights, shard_shapes) =
                load_single_to_gpu(shard_path, stream, &mut should_load)?;
            all_weights.extend(shard_weights);
            all_shapes.extend(shard_shapes);
        }

        info!(
            "gpu_loader: loaded {} total tensors from {} shards (f32)",
            all_weights.len(),
            shard_files.len()
        );
        Ok((all_weights, all_shapes))
    }

    // -----------------------------------------------------------------------
    // F16 path (new)
    // -----------------------------------------------------------------------

    fn load_single_to_gpu_f16(
        path: &Path,
        stream: &Arc<CudaStream>,
        mut should_load: impl FnMut(&str) -> bool,
    ) -> Result<(
        HashMap<String, CudaSlice<half::f16>>,
        HashMap<String, Vec<usize>>,
    )> {
        info!("gpu_loader: memory-mapping {} (f16 mode)", path.display());

        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            LLMError::ModelError(format!("mmap failed for {}: {}", path.display(), e))
        })?;
        let data: &[u8] = &mmap;

        let (header, data_start) = parse_safetensors_header(data, path)?;

        let mut weights: HashMap<String, CudaSlice<half::f16>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        for (name, meta) in &header {
            if name == "__metadata__" {
                continue;
            }

            let (dtype_str, shape, tensor_bytes) = parse_tensor_meta(meta, name, data, data_start)?;
            let numel: usize = shape.iter().product();
            let decision =
                super::filtered_f16_tensor_decision(dtype_str, name.as_str(), &mut should_load);

            if dtype_str == "U8" {
                debug!(tensor = name.as_str(), shape = ?shape, "skipping U8 tensor on f16 GPU path");
                if decision.record_shape {
                    shapes.insert(name.clone(), shape);
                }
                continue;
            }

            if !decision.upload {
                debug!(
                    tensor = name.as_str(),
                    dtype = dtype_str,
                    shape = ?shape,
                    "skipping tensor upload on filtered f16 GPU path"
                );
                if decision.record_shape {
                    shapes.insert(name.clone(), shape);
                }
                continue;
            }

            let f16_host = convert_to_f16(tensor_bytes, dtype_str, numel, name)?;

            let gpu_slice = stream.clone_htod(&f16_host).map_err(|e| {
                LLMError::GpuError(format!(
                    "clone_htod failed for tensor {} ({} f16 elems): {}",
                    name,
                    f16_host.len(),
                    e
                ))
            })?;

            debug!(
                tensor = name.as_str(),
                dtype = dtype_str,
                shape = ?shape,
                numel = numel,
                "uploaded tensor to GPU (f16)"
            );

            weights.insert(name.clone(), gpu_slice);
            shapes.insert(name.clone(), shape);
        }

        info!(
            "gpu_loader: loaded {} tensors from {} to GPU (f16)",
            weights.len(),
            path.display()
        );
        Ok((weights, shapes))
    }

    fn load_sharded_to_gpu_f16(
        dir: &Path,
        stream: &Arc<CudaStream>,
        mut should_load: impl FnMut(&str) -> bool,
    ) -> Result<(
        HashMap<String, CudaSlice<half::f16>>,
        HashMap<String, Vec<usize>>,
    )> {
        let shard_files = collect_shards(dir)?;

        info!(
            "gpu_loader: loading {} shards from {} to GPU (f16)",
            shard_files.len(),
            dir.display()
        );

        let mut all_weights: HashMap<String, CudaSlice<half::f16>> = HashMap::new();
        let mut all_shapes: HashMap<String, Vec<usize>> = HashMap::new();
        for shard_path in &shard_files {
            let (shard_weights, shard_shapes) =
                load_single_to_gpu_f16(shard_path, stream, &mut should_load)?;
            all_weights.extend(shard_weights);
            all_shapes.extend(shard_shapes);
        }

        info!(
            "gpu_loader: loaded {} total tensors from {} shards (f16)",
            all_weights.len(),
            shard_files.len()
        );
        Ok((all_weights, all_shapes))
    }

    // -----------------------------------------------------------------------
    // Shared helpers
    // -----------------------------------------------------------------------

    /// Parse the safetensors header from raw mmap bytes.
    fn parse_safetensors_header(
        data: &[u8],
        path: &Path,
    ) -> Result<(HashMap<String, serde_json::Value>, usize)> {
        if data.len() < 8 {
            return Err(LLMError::ModelError(
                "safetensors file too small for header".into(),
            ));
        }

        let header_size = u64::from_le_bytes(
            data[..8]
                .try_into()
                .map_err(|_| LLMError::ModelError("invalid header size bytes".into()))?,
        ) as usize;

        if 8 + header_size > data.len() {
            return Err(LLMError::ModelError(
                "header size exceeds file length".into(),
            ));
        }

        let header_bytes = &data[8..8 + header_size];
        let header_str = std::str::from_utf8(header_bytes)
            .map_err(|e| LLMError::ModelError(format!("invalid header utf8: {}", e)))?;
        let header: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| LLMError::SerializationError(format!("header json: {}", e)))?;

        Ok((header, 8 + header_size))
    }

    /// Extract dtype, shape, and byte slice for a single tensor from header metadata.
    fn parse_tensor_meta<'a, 'b>(
        meta: &'b serde_json::Value,
        name: &str,
        data: &'a [u8],
        data_start: usize,
    ) -> Result<(&'b str, Vec<usize>, &'a [u8])> {
        let obj = meta
            .as_object()
            .ok_or_else(|| LLMError::ModelError(format!("tensor {} has non-object meta", name)))?;

        let dtype_str = obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LLMError::ModelError(format!("tensor {} missing dtype", name)))?;

        let shape: Vec<usize> = obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| LLMError::ModelError(format!("tensor {} missing shape", name)))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| LLMError::ModelError("invalid shape element".into()))
            })
            .collect::<Result<Vec<_>>>()?;

        let offsets = obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| LLMError::ModelError(format!("tensor {} missing data_offsets", name)))?;

        if offsets.len() != 2 {
            return Err(LLMError::ModelError(format!(
                "tensor {} has {} offsets, expected 2",
                name,
                offsets.len()
            )));
        }

        let start = offsets[0].as_u64().unwrap_or(0) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;
        let abs_start = data_start + start;
        let abs_end = data_start + end;

        if abs_end > data.len() {
            return Err(LLMError::ModelError(format!(
                "tensor {} data range [{}, {}) exceeds file size {}",
                name,
                abs_start,
                abs_end,
                data.len()
            )));
        }

        Ok((dtype_str, shape, &data[abs_start..abs_end]))
    }

    /// Collect sorted shard file paths from a directory.
    fn collect_shards(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
        let mut shard_files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();
        shard_files.sort();

        if shard_files.is_empty() {
            return Err(LLMError::ModelError(format!(
                "no .safetensors files found in {}",
                dir.display()
            )));
        }
        Ok(shard_files)
    }

    // -----------------------------------------------------------------------
    // Dtype conversion
    // -----------------------------------------------------------------------

    /// Convert raw tensor bytes to `Vec<f32>` based on the safetensors dtype string.
    ///
    /// Supported dtypes: F32 (zero-copy reinterpret), F16, BF16 (widened to f32).
    fn convert_to_f32(
        bytes: &[u8],
        dtype_str: &str,
        numel: usize,
        tensor_name: &str,
    ) -> Result<Vec<f32>> {
        match dtype_str {
            "F32" => {
                if bytes.len() != numel * 4 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} F32 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                let mut out = vec![0f32; numel];
                // SAFETY: f32 is Pod, byte count verified.
                let src =
                    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, numel) };
                out.copy_from_slice(src);
                Ok(out)
            }
            "F16" => {
                if bytes.len() != numel * 2 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} F16 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                let mut out = Vec::with_capacity(numel);
                for i in 0..numel {
                    let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    let val = half::f16::from_bits(bits);
                    out.push(val.to_f32());
                }
                Ok(out)
            }
            "BF16" => {
                if bytes.len() != numel * 2 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} BF16 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                let mut out = Vec::with_capacity(numel);
                for i in 0..numel {
                    let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    let val = half::bf16::from_bits(bits);
                    out.push(val.to_f32());
                }
                Ok(out)
            }
            _ => Err(LLMError::ModelError(format!(
                "gpu_loader: unsupported dtype '{}' for tensor '{}', only F32/F16/BF16 supported",
                dtype_str, tensor_name
            ))),
        }
    }

    /// Convert raw tensor bytes to `Vec<half::f16>` for the f16 GPU path.
    ///
    /// - F16: reinterpret bytes directly as half::f16 (no conversion).
    /// - BF16: convert bf16 -> f16 on the host (no intermediate f32 widen).
    /// - F32: narrow f32 -> f16.
    fn convert_to_f16(
        bytes: &[u8],
        dtype_str: &str,
        numel: usize,
        tensor_name: &str,
    ) -> Result<Vec<half::f16>> {
        match dtype_str {
            "F16" => {
                if bytes.len() != numel * 2 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} F16 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                // Direct reinterpret -- no conversion needed.
                let mut out = vec![half::f16::ZERO; numel];
                // SAFETY: half::f16 is repr(transparent) over u16, 2 bytes each,
                // byte count verified above. Source is valid mmap data.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr() as *mut u8,
                        bytes.len(),
                    );
                }
                Ok(out)
            }
            "BF16" => {
                // Convert bf16 -> f16 directly without widening to f32.
                // bf16 has 8-bit exponent + 7-bit mantissa
                // f16  has 5-bit exponent + 10-bit mantissa
                // We go bf16 -> f32 -> f16 per element. The bf16->f32 step is
                // a cheap bit shift (no real work), and f32->f16 is the
                // standard narrowing. This avoids allocating a full f32 buffer.
                if bytes.len() != numel * 2 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} BF16 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                let mut out = Vec::with_capacity(numel);
                for i in 0..numel {
                    let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    let bf = half::bf16::from_bits(bits);
                    // bf16->f32 is a trivial bit shift, then f32->f16 narrow
                    out.push(half::f16::from_f32(bf.to_f32()));
                }
                Ok(out)
            }
            "F32" => {
                if bytes.len() != numel * 4 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} F32 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                // Narrow f32 -> f16
                let mut out = Vec::with_capacity(numel);
                // SAFETY: f32 is Pod, byte count verified.
                let src =
                    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, numel) };
                for &v in src {
                    out.push(half::f16::from_f32(v));
                }
                Ok(out)
            }
            _ => Err(LLMError::ModelError(format!(
                "gpu_loader: unsupported dtype '{}' for tensor '{}', only F32/F16/BF16 supported",
                dtype_str, tensor_name
            ))),
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::{
    load_weights_to_gpu, load_weights_to_gpu_f16, load_weights_to_gpu_f16_with_shapes,
    load_weights_to_gpu_f16_with_shapes_filtered, load_weights_to_gpu_with_shapes,
    load_weights_to_gpu_with_shapes_filtered, GpuDType,
};

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[usize], &[u8])]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();

        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            let end = data.len();

            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }

        let header_json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_json);
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    fn unique_temp_dir(test_name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gpt_oss_u8_filtered_loader_{test_name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn module_compiles() {
        assert!(true);
    }

    #[test]
    fn f16_filter_uploads_selected_non_u8_tensor() {
        let mut should_load = |name: &str| name == "model.layers.0.self_attn.q_proj.weight";

        let decision = super::filtered_f16_tensor_decision(
            "F16",
            "model.layers.0.self_attn.q_proj.weight",
            &mut should_load,
        );

        assert!(decision.record_shape);
        assert!(decision.upload);
    }

    #[test]
    fn f16_filter_preserves_shape_for_skipped_non_u8_tensor() {
        let mut should_load = |name: &str| name == "model.layers.0.self_attn.q_proj.weight";

        let decision = super::filtered_f16_tensor_decision(
            "BF16",
            "model.layers.1.self_attn.q_proj.weight",
            &mut should_load,
        );

        assert!(decision.record_shape);
        assert!(!decision.upload);
    }

    #[test]
    fn f16_filter_preserves_shape_and_skips_u8_tensor() {
        let mut should_load = |_name: &str| true;

        let decision = super::filtered_f16_tensor_decision(
            "U8",
            "model.layers.0.mlp.experts.gate_up_proj_blocks",
            &mut should_load,
        );

        assert!(decision.record_shape);
        assert!(!decision.upload);
    }

    #[test]
    fn u8_filtered_loader_selects_one_u8_tensor_from_single_file() {
        let dir = unique_temp_dir("single_select");
        let path = dir.join("model.safetensors");
        write_safetensors(
            &path,
            &[
                (
                    "model.layers.0.mlp.experts.gate_up_proj_blocks",
                    "U8",
                    &[3],
                    &[1, 2, 3],
                ),
                (
                    "model.layers.1.mlp.experts.gate_up_proj_blocks",
                    "U8",
                    &[2],
                    &[4, 5],
                ),
            ],
        );

        let weights = super::load_u8_weights_to_host_filtered(&path, |name| {
            name == "model.layers.1.mlp.experts.gate_up_proj_blocks"
        })
        .unwrap();

        assert_eq!(weights.len(), 1);
        assert_eq!(
            weights["model.layers.1.mlp.experts.gate_up_proj_blocks"],
            vec![4, 5]
        );
    }

    #[test]
    fn u8_filtered_loader_ignores_non_u8_even_when_selected() {
        let dir = unique_temp_dir("ignore_non_u8");
        let path = dir.join("model.safetensors");
        write_safetensors(
            &path,
            &[
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    "F16",
                    &[2],
                    &[0, 0, 0, 0],
                ),
                (
                    "model.layers.0.mlp.experts.down_proj_scales",
                    "U8",
                    &[1],
                    &[7],
                ),
            ],
        );

        let weights = super::load_u8_weights_to_host_filtered(&path, |_| true).unwrap();

        assert_eq!(weights.len(), 1);
        assert!(weights.contains_key("model.layers.0.mlp.experts.down_proj_scales"));
        assert!(!weights.contains_key("model.layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn u8_filtered_loader_returns_empty_when_filter_rejects_all() {
        let dir = unique_temp_dir("reject_all");
        let path = dir.join("model.safetensors");
        write_safetensors(
            &path,
            &[(
                "model.layers.0.mlp.experts.down_proj_blocks",
                "U8",
                &[2],
                &[9, 8],
            )],
        );

        let weights = super::load_u8_weights_to_host_filtered(&path, |_| false).unwrap();

        assert!(weights.is_empty());
    }

    #[test]
    fn u8_unfiltered_wrapper_loads_all_u8_tensors() {
        let dir = unique_temp_dir("wrapper");
        let path = dir.join("model.safetensors");
        write_safetensors(
            &path,
            &[
                (
                    "model.layers.0.mlp.experts.gate_up_proj_blocks",
                    "U8",
                    &[1],
                    &[1],
                ),
                (
                    "model.layers.1.mlp.experts.gate_up_proj_blocks",
                    "U8",
                    &[1],
                    &[2],
                ),
                (
                    "model.layers.2.self_attn.q_proj.weight",
                    "F32",
                    &[1],
                    &[0, 0, 0, 0],
                ),
            ],
        );

        let unfiltered = super::load_u8_weights_to_host(&path).unwrap();
        let filtered_all = super::load_u8_weights_to_host_filtered(&path, |_| true).unwrap();

        assert_eq!(unfiltered, filtered_all);
        assert_eq!(unfiltered.len(), 2);
    }

    #[test]
    fn u8_filtered_loader_selects_across_sharded_directory() {
        let dir = unique_temp_dir("sharded");
        std::fs::write(dir.join("README.md"), "ignored").unwrap();
        write_safetensors(
            &dir.join("model-00002-of-00002.safetensors"),
            &[(
                "model.layers.12.mlp.experts.down_proj_scales",
                "U8",
                &[1],
                &[12],
            )],
        );
        write_safetensors(
            &dir.join("model-00001-of-00002.safetensors"),
            &[(
                "model.layers.0.mlp.experts.down_proj_scales",
                "U8",
                &[1],
                &[0],
            )],
        );

        let weights = super::load_u8_weights_to_host_filtered(&dir, |name| {
            name == "model.layers.0.mlp.experts.down_proj_scales"
                || name == "model.layers.12.mlp.experts.down_proj_scales"
        })
        .unwrap();

        let mut names = weights.keys().cloned().collect::<Vec<_>>();
        names.sort();
        assert_eq!(
            names,
            vec![
                "model.layers.0.mlp.experts.down_proj_scales".to_string(),
                "model.layers.12.mlp.experts.down_proj_scales".to_string()
            ]
        );
        assert_eq!(
            weights["model.layers.0.mlp.experts.down_proj_scales"],
            vec![0]
        );
        assert_eq!(
            weights["model.layers.12.mlp.experts.down_proj_scales"],
            vec![12]
        );
    }
}
