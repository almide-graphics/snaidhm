//! Native implementation of the `font_data` extern namespace `src/ttf.almd`
//! reads a font through: the bytes of the one loaded font, read big-endian.
//!
//! In the browser the page holds the font in an `ArrayBuffer` and answers the
//! same calls through a `DataView` (ceangal `host/runtime.js`,
//! `createFontImports`). Natively the bytes are loaded from a file by `load`,
//! declared in `src/native/font.almd`.
//!
//! A read past the end answers 0 where the `DataView` throws: a truncated or
//! malformed font should cost its glyphs, not bring down the program drawing
//! them.

use std::sync::{OnceLock, RwLock};

fn bytes() -> &'static RwLock<Vec<u8>> {
    static FONT: OnceLock<RwLock<Vec<u8>>> = OnceLock::new();
    FONT.get_or_init(|| RwLock::new(Vec::new()))
}

/// `N` bytes at `offset`, or `None` when any of them is out of range.
fn read<const N: usize>(offset: i64) -> Option<[u8; N]> {
    let font = match bytes().read() {
        Ok(f) => f,
        Err(poisoned) => poisoned.into_inner(),
    };
    let start = usize::try_from(offset).ok()?;
    let slice = font.get(start..start.checked_add(N)?)?;
    slice.try_into().ok()
}

/// Make the font at `path` the one every `font_data` read answers from.
/// `false` (with the reason on stderr) when the file cannot be read; the
/// previously loaded font, if any, stays.
pub fn load(path: &str) -> bool {
    match std::fs::read(path) {
        Ok(data) => {
            let mut font = match bytes().write() {
                Ok(f) => f,
                Err(poisoned) => poisoned.into_inner(),
            };
            *font = data;
            true
        }
        Err(e) => {
            eprintln!("[snaidhm/font] cannot read {path}: {e}");
            false
        }
    }
}

// ── The `font_data` namespace: one `pub fn` per `@extern(rust, "crate::font_data", ...)` ──

pub fn len() -> i64 {
    match bytes().read() {
        Ok(f) => f.len() as i64,
        Err(poisoned) => poisoned.into_inner().len() as i64,
    }
}

pub fn u8(offset: i64) -> i64 {
    read::<1>(offset).map_or(0, |b| i64::from(b[0]))
}

pub fn i8(offset: i64) -> i64 {
    read::<1>(offset).map_or(0, |b| i64::from(b[0] as i8))
}

pub fn u16be(offset: i64) -> i64 {
    read::<2>(offset).map_or(0, |b| i64::from(u16::from_be_bytes(b)))
}

pub fn i16be(offset: i64) -> i64 {
    read::<2>(offset).map_or(0, |b| i64::from(i16::from_be_bytes(b)))
}

pub fn u32be(offset: i64) -> i64 {
    read::<4>(offset).map_or(0, |b| i64::from(u32::from_be_bytes(b)))
}
