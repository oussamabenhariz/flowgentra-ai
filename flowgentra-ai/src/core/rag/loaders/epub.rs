//! EPUB Loader
//!
//! Loads text content from `.epub` e-books. An EPUB file is a ZIP archive
//! containing XHTML files. This loader:
//! 1. Reads `META-INF/container.xml` to find the OPF package file.
//! 2. Parses the OPF to get the reading order (spine).
//! 3. Extracts text from each XHTML file in spine order.
//!
//! Each chapter / spine item becomes one [`LoadedDocument`].

use std::collections::HashMap;
use std::io::Read;

use serde_json::json;

use crate::core::rag::document_loader::{FileType, LoadedDocument};

/// Loads chapters from an `.epub` file as documents.
pub struct EpubLoader {
    path: String,
    /// If true, concatenate all chapters into a single document.
    single_document: bool,
}

impl EpubLoader {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into(), single_document: false }
    }

    pub fn as_single_document(mut self) -> Self {
        self.single_document = true;
        self
    }

    pub fn load(&self) -> Result<Vec<LoadedDocument>, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(&self.path)?;
        let mut archive = zip::ZipArchive::new(file)?;

        // Find OPF root
        let opf_path = find_opf_path(&mut archive)?;
        let opf_dir = opf_path
            .rfind('/')
            .map(|i| &opf_path[..=i])
            .unwrap_or("");

        // Parse spine from OPF
        let spine_hrefs = parse_spine(&mut archive, &opf_path, opf_dir)?;

        let source = self.path.clone();
        let mut docs = Vec::new();

        for (i, href) in spine_hrefs.iter().enumerate() {
            let full_path = if href.starts_with('/') {
                href.trim_start_matches('/').to_string()
            } else {
                format!("{opf_dir}{href}")
            };

            let text = match archive.by_name(&full_path) {
                Ok(mut f) => {
                    let mut content = String::new();
                    f.read_to_string(&mut content).ok();
                    strip_html_tags(&content)
                }
                Err(_) => continue,
            };

            if text.trim().is_empty() {
                continue;
            }

            let mut metadata = HashMap::new();
            metadata.insert("source".to_string(), json!(source));
            metadata.insert("chapter_index".to_string(), json!(i));
            metadata.insert("file".to_string(), json!(full_path));

            docs.push(LoadedDocument {
                id: format!("epub_chapter_{i}"),
                text,
                source: format!("{}#chapter_{i}", self.path),
                file_type: FileType::PlainText,
                metadata,
            });
        }

        if self.single_document && !docs.is_empty() {
            let combined = docs.iter().map(|d| d.text.as_str()).collect::<Vec<_>>().join("\n\n");
            let mut metadata = HashMap::new();
            metadata.insert("source".to_string(), json!(source));
            metadata.insert("chapters".to_string(), json!(docs.len()));
            return Ok(vec![LoadedDocument {
                id: format!("epub_{}", self.path),
                text: combined,
                source: self.path.clone(),
                file_type: FileType::PlainText,
                metadata,
            }]);
        }

        Ok(docs)
    }
}

fn find_opf_path(archive: &mut zip::ZipArchive<std::fs::File>) -> Result<String, Box<dyn std::error::Error>> {
    let mut container = String::new();
    archive.by_name("META-INF/container.xml")?.read_to_string(&mut container)?;
    // Extract full-path attribute from <rootfile full-path="...">
    let path = container
        .split("full-path=\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .unwrap_or("OEBPS/content.opf")
        .to_string();
    Ok(path)
}

fn parse_spine(
    archive: &mut zip::ZipArchive<std::fs::File>,
    opf_path: &str,
    _opf_dir: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut opf_content = String::new();
    archive.by_name(opf_path)?.read_to_string(&mut opf_content)?;

    // Build id → href map from <item> elements
    let mut id_to_href: HashMap<String, String> = HashMap::new();
    for item_part in opf_content.split("<item ").skip(1) {
        let attrs = item_part.split('>').next().unwrap_or("");
        let id = extract_attr(attrs, "id").unwrap_or_default();
        let href = extract_attr(attrs, "href").unwrap_or_default();
        let media_type = extract_attr(attrs, "media-type").unwrap_or_default();
        if media_type.contains("xhtml") || media_type.contains("html") {
            id_to_href.insert(id, href);
        }
    }

    // Extract spine order
    let spine_section = opf_content
        .split("<spine")
        .nth(1)
        .and_then(|s| s.split("</spine>").next())
        .unwrap_or("");

    let mut hrefs = Vec::new();
    for itemref in spine_section.split("<itemref ").skip(1) {
        let idref = extract_attr(itemref, "idref").unwrap_or_default();
        if let Some(href) = id_to_href.get(&idref) {
            hrefs.push(href.clone());
        }
    }

    Ok(hrefs)
}

fn extract_attr(s: &str, name: &str) -> Option<String> {
    let prefix = format!("{name}=\"");
    s.split(&prefix)
        .nth(1)
        .and_then(|part| part.split('"').next())
        .map(str::to_string)
}

fn strip_html_tags(html: &str) -> String {
    let mut out = String::new();
    let mut inside_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => inside_tag = true,
            '>' => inside_tag = false,
            _ if !inside_tag => out.push(ch),
            _ => {}
        }
    }
    // Collapse whitespace
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}
