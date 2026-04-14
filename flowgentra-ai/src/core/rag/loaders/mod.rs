//! Extended Document Loaders
//!
//! Provides loaders for sources beyond the basic file-type loaders in
//! `document_loader.rs`. All loaders return [`LoadedDocument`] instances that
//! can be fed directly into `IngestionPipeline`.
//!
//! ## Available loaders
//!
//! | Loader | Source |
//! |---|---|
//! | [`WebLoader`] | HTTP/HTTPS URLs |
//! | [`CsvLoader`] | `.csv` files |
//! | [`JsonLoader`] | `.json` (array of objects) |
//! | [`JsonlLoader`] | `.jsonl` / `.ndjson` |
//! | [`DocxLoader`] | `.docx` Word documents |
//! | [`ExcelLoader`] | `.xlsx` Excel spreadsheets |
//! | [`EpubLoader`] | `.epub` e-books |
//! | [`DirectoryLoader`] | Recursive directory |
//! | [`SitemapLoader`] | XML sitemaps |
//! | [`RecursiveUrlLoader`] | Web crawling |
//! | [`WikipediaLoader`] | Wikipedia articles |
//! | [`ArxivLoader`] | arXiv paper abstracts |
//! | [`RssFeedLoader`] | RSS / Atom feeds |
//! | [`YouTubeLoader`] | YouTube video metadata |
//! | [`S3Loader`] | AWS S3 objects |
//! | [`DataFrameLoader`] | Tabular/DataFrame rows |
//! | [`GitLoader`] | Git repository files |

pub mod web;
pub mod csv;
pub mod json_loader;
pub mod docx;
pub mod directory;
pub mod excel;
pub mod epub;
pub mod sitemap;
pub mod recursive_url;
pub mod wikipedia;
pub mod arxiv;
pub mod rss;
pub mod youtube;
pub mod s3;
pub mod dataframe;
pub mod git;

pub use web::{WebLoader, WebLoaderConfig};
pub use csv::CsvLoader;
pub use json_loader::{JsonLoader, JsonlLoader};
pub use docx::DocxLoader;
pub use directory::{DirectoryLoader, DirectoryLoaderConfig};
pub use excel::ExcelLoader;
pub use epub::EpubLoader;
pub use sitemap::{SitemapConfig, SitemapLoader};
pub use recursive_url::{RecursiveUrlConfig, RecursiveUrlLoader};
pub use wikipedia::WikipediaLoader;
pub use arxiv::ArxivLoader;
pub use rss::RssFeedLoader;
pub use youtube::YouTubeLoader;
pub use s3::S3Loader;
pub use dataframe::DataFrameLoader;
pub use git::{GitLoader, GitLoaderConfig};
