//! Explicit Hugging Face snapshot download support.
//!
//! This module downloads only the files needed by the native GPT-OSS loader.
//! `hf-hub` supplies file locking and resumable `.part` downloads; this layer
//! adds revision pinning and a content manifest without attempting model load.

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};

use gpt_oss_core::prelude::{LLMError, Result};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const FETCH_MANIFEST_FILENAME: &str = "gpt-oss-rs-fetch-manifest.json";
const FETCH_MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchOptions {
    pub model: String,
    pub revision: String,
    /// Hugging Face hub cache root (the directory containing `models--*`).
    pub cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestFile {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotManifest {
    pub format_version: u32,
    pub model: String,
    pub requested_revision: String,
    pub resolved_revision: String,
    pub files: Vec<ManifestFile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchResult {
    pub snapshot_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest: SnapshotManifest,
}

/// Download a revision-pinned native GPT-OSS snapshot and write its manifest.
pub fn fetch_snapshot(options: &FetchOptions) -> Result<FetchResult> {
    if options.model.trim().is_empty() {
        return Err(LLMError::ConfigError(
            "fetch model must not be empty".into(),
        ));
    }
    if options.revision.trim().is_empty() {
        return Err(LLMError::ConfigError(
            "fetch revision must not be empty".into(),
        ));
    }

    let mut builder = ApiBuilder::from_env();
    if let Some(cache_dir) = &options.cache_dir {
        builder = builder.with_cache_dir(cache_dir.clone());
    }
    let api = builder
        .build()
        .map_err(|error| LLMError::ModelError(format!("failed to initialize hf-hub: {error}")))?;
    let repo = api.repo(Repo::with_revision(
        options.model.clone(),
        RepoType::Model,
        options.revision.clone(),
    ));
    let info = repo.info().map_err(|error| {
        LLMError::ModelError(format!(
            "failed to resolve {} at revision {}: {error}",
            options.model, options.revision
        ))
    })?;
    let filenames = select_native_snapshot_files(
        info.siblings
            .iter()
            .map(|sibling| sibling.rfilename.as_str()),
    )?;

    let mut downloaded = Vec::with_capacity(filenames.len());
    for filename in &filenames {
        let path = repo.get(filename).map_err(|error| {
            LLMError::ModelError(format!("failed to download {filename}: {error}"))
        })?;
        downloaded.push((filename.clone(), path));
    }

    let snapshot_dir = downloaded
        .first()
        .and_then(|(_, path)| ancestor_for_relative_file(path, &filenames[0]))
        .ok_or_else(|| LLMError::ModelError("download produced no snapshot directory".into()))?;
    let manifest = build_manifest(&options.model, &options.revision, &info.sha, &downloaded)?;
    let manifest_path = snapshot_dir.join(FETCH_MANIFEST_FILENAME);
    write_manifest_atomic(&manifest_path, &manifest)?;

    Ok(FetchResult {
        snapshot_dir,
        manifest_path,
        manifest,
    })
}

fn select_native_snapshot_files<'a>(
    files: impl IntoIterator<Item = &'a str>,
) -> Result<Vec<String>> {
    let mut selected = files
        .into_iter()
        .filter(|filename| is_native_snapshot_file(filename))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    selected.sort();
    selected.dedup();

    if !selected.iter().any(|filename| filename == "config.json") {
        return Err(LLMError::ModelError(
            "model repository does not contain config.json".into(),
        ));
    }
    if !selected.iter().any(|filename| filename == "tokenizer.json") {
        return Err(LLMError::ModelError(
            "model repository does not contain tokenizer.json".into(),
        ));
    }
    if !selected
        .iter()
        .any(|filename| filename.ends_with(".safetensors"))
    {
        return Err(LLMError::ModelError(
            "model repository does not contain SafeTensors weights".into(),
        ));
    }
    Ok(selected)
}

fn is_native_snapshot_file(filename: &str) -> bool {
    let path = Path::new(filename);
    if path.is_absolute()
        || filename.starts_with("original/")
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return false;
    }

    filename.ends_with(".safetensors")
        || filename.ends_with(".safetensors.index.json")
        || matches!(
            filename,
            "config.json"
                | "generation_config.json"
                | "tokenizer.json"
                | "tokenizer_config.json"
                | "special_tokens_map.json"
                | "added_tokens.json"
                | "chat_template.jinja"
                | "merges.txt"
                | "vocab.json"
        )
}

fn ancestor_for_relative_file(path: &Path, relative: &str) -> Option<PathBuf> {
    let depth = Path::new(relative).components().count();
    let mut ancestor = path;
    for _ in 0..depth {
        ancestor = ancestor.parent()?;
    }
    Some(ancestor.to_path_buf())
}

fn build_manifest(
    model: &str,
    requested_revision: &str,
    resolved_revision: &str,
    files: &[(String, PathBuf)],
) -> Result<SnapshotManifest> {
    let mut entries = Vec::with_capacity(files.len());
    for (relative, path) in files {
        let metadata = std::fs::metadata(path).map_err(|error| {
            LLMError::ModelError(format!("failed to stat {}: {error}", path.display()))
        })?;
        entries.push(ManifestFile {
            path: relative.clone(),
            size: metadata.len(),
            sha256: sha256_file(path)?,
        });
    }
    entries.sort_by(|left, right| left.path.cmp(&right.path));

    Ok(SnapshotManifest {
        format_version: FETCH_MANIFEST_VERSION,
        model: model.to_string(),
        requested_revision: requested_revision.to_string(),
        resolved_revision: resolved_revision.to_string(),
        files: entries,
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn write_manifest_atomic(path: &Path, manifest: &SnapshotManifest) -> Result<()> {
    let parent = path.parent().ok_or_else(|| {
        LLMError::ModelError(format!("manifest path {} has no parent", path.display()))
    })?;
    std::fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        FETCH_MANIFEST_FILENAME,
        std::process::id()
    ));
    let result = (|| -> Result<()> {
        let mut file = File::create(&temporary)?;
        serde_json::to_writer_pretty(&mut file, manifest).map_err(|error| {
            LLMError::ModelError(format!("failed to serialize fetch manifest: {error}"))
        })?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        std::fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn selection_is_sorted_and_excludes_non_native_formats() {
        let selected = select_native_snapshot_files([
            "model.gguf",
            "original/model.safetensors",
            "tokenizer.json",
            "model-00002-of-00002.safetensors",
            "README.md",
            "config.json",
            "model-00001-of-00002.safetensors",
            "model.safetensors.index.json",
        ])
        .unwrap();
        assert_eq!(selected[0], "config.json");
        assert!(selected.iter().all(|file| file != "model.gguf"));
        assert!(selected
            .iter()
            .all(|file| file != "original/model.safetensors"));
        assert_eq!(selected.last().unwrap(), "tokenizer.json");
    }

    #[test]
    fn selection_rejects_incomplete_repository() {
        assert!(select_native_snapshot_files(["config.json", "tokenizer.json"]).is_err());
        assert!(select_native_snapshot_files(["config.json", "model.safetensors"]).is_err());
    }

    #[test]
    fn manifest_hashes_files_and_writes_atomically() {
        let temp = tempdir().unwrap();
        let config = temp.path().join("config.json");
        std::fs::write(&config, b"abc").unwrap();
        let manifest = build_manifest(
            "openai/gpt-oss-20b",
            "main",
            "deadbeef",
            &[("config.json".into(), config)],
        )
        .unwrap();
        assert_eq!(manifest.files[0].size, 3);
        assert_eq!(
            manifest.files[0].sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        let path = temp.path().join(FETCH_MANIFEST_FILENAME);
        write_manifest_atomic(&path, &manifest).unwrap();
        let reread: SnapshotManifest =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(reread, manifest);
        assert!(!temp
            .path()
            .join(format!(
                ".{}.{}.tmp",
                FETCH_MANIFEST_FILENAME,
                std::process::id()
            ))
            .exists());
    }
}
