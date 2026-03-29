use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use crate::cache::KernelCache;
use crate::jit::{JitCompiler, JitError};

// ---------------------------------------------------------------------------
// ModelConfig
// ---------------------------------------------------------------------------

pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub num_layers: usize,
}

impl ModelConfig {
    fn qkv_dim(&self) -> usize {
        (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
    }

    fn gateup_dim(&self) -> usize {
        self.intermediate_size * 2
    }
}

// ---------------------------------------------------------------------------
// CompiledFusedKernel
// ---------------------------------------------------------------------------

pub struct CompiledFusedKernel {
    pub name: String,
    pub ptx: Vec<u8>,
    pub function_name: String,
    pub hidden_size: usize,
    pub output_dim: usize,
}

// ---------------------------------------------------------------------------
// TemplateEngine
// ---------------------------------------------------------------------------

pub struct TemplateEngine {
    templates: HashMap<String, String>,
}

impl TemplateEngine {
    pub fn new() -> Self {
        let mut templates = HashMap::new();
        templates.insert(
            "fused_norm_qkv_gemv".into(),
            include_str!("templates/fused_norm_qkv_gemv.cu.template").into(),
        );
        templates.insert(
            "fused_norm_gateup_gemv".into(),
            include_str!("templates/fused_norm_gateup_gemv.cu.template").into(),
        );
        templates.insert(
            "fused_silu_down_gemv".into(),
            include_str!("templates/fused_silu_down_gemv.cu.template").into(),
        );
        templates.insert(
            "fused_add_norm_qkv_gemv".into(),
            include_str!("templates/fused_add_norm_qkv_gemv.cu.template").into(),
        );
        templates.insert(
            "fused_add_norm_gateup_gemv".into(),
            include_str!("templates/fused_add_norm_gateup_gemv.cu.template").into(),
        );
        Self { templates }
    }

    pub fn load_from_dir(dir: &Path) -> io::Result<Self> {
        let mut templates = HashMap::new();
        if dir.exists() {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .map_or(false, |f| f.ends_with(".cu.template"))
                {
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .strip_suffix(".cu")
                        .unwrap_or("")
                        .to_string();
                    if !name.is_empty() {
                        templates.insert(name, std::fs::read_to_string(&path)?);
                    }
                }
            }
        }
        Ok(Self { templates })
    }

    pub fn instantiate(
        &self,
        template_name: &str,
        vars: &HashMap<String, String>,
    ) -> Result<String, CompilerError> {
        let src = self
            .templates
            .get(template_name)
            .ok_or_else(|| CompilerError::TemplateNotFound(template_name.to_string()))?;
        let mut out = src.clone();
        for (key, val) in vars {
            let placeholder = format!("{{{{{}}}}}", key);
            out = out.replace(&placeholder, val);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// FusionCompiler -- top-level orchestrator
// ---------------------------------------------------------------------------

pub struct FusionCompiler {
    jit: JitCompiler,
    cache: KernelCache,
    templates: TemplateEngine,
}

impl FusionCompiler {
    pub fn new(cache_dir: PathBuf) -> Result<Self, CompilerError> {
        Ok(Self {
            jit: JitCompiler::new()?,
            cache: KernelCache::new(cache_dir),
            templates: TemplateEngine::new(),
        })
    }

    pub fn with_templates(cache_dir: PathBuf, templates: TemplateEngine) -> Result<Self, CompilerError> {
        Ok(Self {
            jit: JitCompiler::new()?,
            cache: KernelCache::new(cache_dir),
            templates,
        })
    }

    /// Compile all fused kernels for a given model configuration.
    /// Returns compiled kernels ready to be loaded into the GPU context.
    pub fn compile_for_model(
        &self,
        config: &ModelConfig,
    ) -> Result<Vec<CompiledFusedKernel>, CompilerError> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let qkv_dim = config.qkv_dim();
        let gateup_dim = config.gateup_dim();

        eprintln!(
            "Compiling 5 fused kernels for model (hidden={}, intermediate={})...",
            hidden, intermediate
        );

        let specs: Vec<KernelSpec> = vec![
            KernelSpec {
                template: "fused_norm_qkv_gemv",
                name: format!("fused_norm_qkv_gemv_{}_{}", hidden, qkv_dim),
                in_dim: hidden,
                out_dim: qkv_dim,
            },
            KernelSpec {
                template: "fused_norm_gateup_gemv",
                name: format!("fused_norm_gateup_gemv_{}_{}", hidden, gateup_dim),
                in_dim: hidden,
                out_dim: gateup_dim,
            },
            KernelSpec {
                template: "fused_silu_down_gemv",
                name: format!("fused_silu_down_gemv_{}_{}", intermediate, hidden),
                in_dim: intermediate,
                out_dim: hidden,
            },
            KernelSpec {
                template: "fused_add_norm_qkv_gemv",
                name: format!("fused_add_norm_qkv_gemv_{}_{}", hidden, qkv_dim),
                in_dim: hidden,
                out_dim: qkv_dim,
            },
            KernelSpec {
                template: "fused_add_norm_gateup_gemv",
                name: format!("fused_add_norm_gateup_gemv_{}_{}", hidden, gateup_dim),
                in_dim: hidden,
                out_dim: gateup_dim,
            },
        ];

        let mut compiled = Vec::with_capacity(specs.len());

        for (i, spec) in specs.iter().enumerate() {
            let cache_key = KernelCache::key_for(
                &spec.name,
                &[spec.in_dim, spec.out_dim],
                self.jit.arch(),
            );

            let ptx = match self.cache.get(&cache_key) {
                Some(cached) => {
                    eprintln!("  [{}/{}] {} (cached)", i + 1, specs.len(), spec.name);
                    cached
                }
                None => {
                    eprintln!("  [{}/{}] {} (compiling...)", i + 1, specs.len(), spec.name);
                    let vars = self.make_template_vars(config, spec);
                    let source = self.templates.instantiate(spec.template, &vars)?;
                    let func_name = format!("{}_kernel", spec.name);
                    let ptx = self.jit.compile_to_ptx(&source, &func_name)?;
                    if let Err(e) = self.cache.put(&cache_key, &ptx) {
                        eprintln!("    warning: failed to cache {}: {}", spec.name, e);
                    }
                    ptx
                }
            };

            let func_name = format!("{}_kernel", spec.name);
            compiled.push(CompiledFusedKernel {
                name: spec.name.clone(),
                ptx,
                function_name: func_name,
                hidden_size: spec.in_dim,
                output_dim: spec.out_dim,
            });
        }

        eprintln!("All {} fused kernels ready.", compiled.len());
        Ok(compiled)
    }

    fn make_template_vars(&self, config: &ModelConfig, spec: &KernelSpec) -> HashMap<String, String> {
        let mut vars = HashMap::new();
        vars.insert("HIDDEN_SIZE".into(), config.hidden_size.to_string());
        vars.insert("INTERMEDIATE_SIZE".into(), config.intermediate_size.to_string());
        vars.insert("NUM_HEADS".into(), config.num_heads.to_string());
        vars.insert("NUM_KV_HEADS".into(), config.num_kv_heads.to_string());
        vars.insert("HEAD_DIM".into(), config.head_dim.to_string());
        vars.insert("RMS_NORM_EPS".into(), format!("{:.10e}", config.rms_norm_eps));
        vars.insert("NUM_LAYERS".into(), config.num_layers.to_string());
        vars.insert("QKV_DIM".into(), config.qkv_dim().to_string());
        vars.insert("GATEUP_DIM".into(), config.gateup_dim().to_string());
        vars.insert("IN_DIM".into(), spec.in_dim.to_string());
        vars.insert("OUT_DIM".into(), spec.out_dim.to_string());
        vars.insert("KERNEL_NAME".into(), format!("{}_kernel", spec.name));

        let block_size: usize = 256;
        let tiles_k = (spec.in_dim + block_size - 1) / block_size;
        vars.insert("BLOCK_SIZE".into(), block_size.to_string());
        vars.insert("TILES_K".into(), tiles_k.to_string());

        vars
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

struct KernelSpec {
    template: &'static str,
    name: String,
    in_dim: usize,
    out_dim: usize,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum CompilerError {
    NvccNotFound,
    NvccFailed(String),
    TemplateNotFound(String),
    Io(String),
    Jit(JitError),
}

impl std::fmt::Display for CompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NvccNotFound => write!(f, "nvcc not found -- install CUDA toolkit or set NVCC env var"),
            Self::NvccFailed(msg) => write!(f, "nvcc compilation failed: {msg}"),
            Self::TemplateNotFound(name) => write!(f, "template not found: {name}"),
            Self::Io(msg) => write!(f, "IO error: {msg}"),
            Self::Jit(e) => write!(f, "JIT error: {e}"),
        }
    }
}

impl std::error::Error for CompilerError {}

impl From<io::Error> for CompilerError {
    fn from(e: io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

impl From<JitError> for CompilerError {
    fn from(e: JitError) -> Self {
        Self::Jit(e)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 1536,
            intermediate_size: 8960,
            num_heads: 12,
            num_kv_heads: 2,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            num_layers: 28,
        }
    }

    #[test]
    fn qkv_dim_calculation() {
        let cfg = test_config();
        // (12 + 2*2) * 128 = 16 * 128 = 2048
        assert_eq!(cfg.qkv_dim(), 2048);
    }

    #[test]
    fn gateup_dim_calculation() {
        let cfg = test_config();
        assert_eq!(cfg.gateup_dim(), 17920);
    }

    #[test]
    fn template_instantiation() {
        let mut templates = HashMap::new();
        templates.insert(
            "test".into(),
            "#define HIDDEN {{HIDDEN_SIZE}}\n#define EPS {{RMS_NORM_EPS}}\n".into(),
        );
        let engine = TemplateEngine { templates };

        let mut vars = HashMap::new();
        vars.insert("HIDDEN_SIZE".into(), "1536".into());
        vars.insert("RMS_NORM_EPS".into(), "1.0000000000e-6".into());

        let result = engine.instantiate("test", &vars).unwrap();
        assert!(result.contains("#define HIDDEN 1536"));
        assert!(result.contains("#define EPS 1.0000000000e-6"));
    }

    #[test]
    fn template_not_found() {
        let engine = TemplateEngine {
            templates: HashMap::new(),
        };
        let vars = HashMap::new();
        assert!(engine.instantiate("nonexistent", &vars).is_err());
    }

    #[test]
    fn builtin_templates_load() {
        let engine = TemplateEngine::new();
        assert!(engine.templates.contains_key("fused_norm_qkv_gemv"));
        assert!(engine.templates.contains_key("fused_norm_gateup_gemv"));
        assert!(engine.templates.contains_key("fused_silu_down_gemv"));
        assert!(engine.templates.contains_key("fused_add_norm_qkv_gemv"));
        assert!(engine.templates.contains_key("fused_add_norm_gateup_gemv"));
        assert_eq!(engine.templates.len(), 5);
    }

    #[test]
    fn builtin_template_substitution() {
        let engine = TemplateEngine::new();
        let cfg = test_config();
        let spec = KernelSpec {
            template: "fused_norm_qkv_gemv",
            name: format!("fused_norm_qkv_gemv_{}_{}", cfg.hidden_size, cfg.qkv_dim()),
            in_dim: cfg.hidden_size,
            out_dim: cfg.qkv_dim(),
        };

        let compiler = FusionCompiler {
            jit: match JitCompiler::new() {
                Ok(j) => j,
                Err(_) => return, // no nvcc, skip
            },
            cache: KernelCache::new(std::env::temp_dir().join("rvllm_test_unused")),
            templates: engine,
        };

        let vars = compiler.make_template_vars(&cfg, &spec);
        let source = compiler.templates.instantiate("fused_norm_qkv_gemv", &vars).unwrap();
        assert!(source.contains("#define HIDDEN_SIZE 1536"));
        assert!(source.contains("#define OUT_DIM 2048"));
        assert!(!source.contains("{{"));
    }

    #[test]
    fn compiler_error_display() {
        let e = CompilerError::NvccNotFound;
        assert!(e.to_string().contains("nvcc not found"));

        let e = CompilerError::TemplateNotFound("foo".into());
        assert!(e.to_string().contains("foo"));
    }

    #[test]
    fn kernel_spec_naming() {
        let cfg = test_config();
        let qkv_dim = cfg.qkv_dim();
        let gateup_dim = cfg.gateup_dim();

        assert_eq!(
            format!("fused_norm_qkv_gemv_{}_{}", cfg.hidden_size, qkv_dim),
            "fused_norm_qkv_gemv_1536_2048"
        );
        assert_eq!(
            format!("fused_norm_gateup_gemv_{}_{}", cfg.hidden_size, gateup_dim),
            "fused_norm_gateup_gemv_1536_17920"
        );
        assert_eq!(
            format!("fused_silu_down_gemv_{}_{}", cfg.intermediate_size, cfg.hidden_size),
            "fused_silu_down_gemv_8960_1536"
        );
    }
}
