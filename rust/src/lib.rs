//! pic2vec Rust core - ONNX inference via tract, image preprocessing via image crate.
//!
//! Exposes a small C ABI so the C++ DuckDB extension layer can drive model
//! management and image-to-embedding without depending on ONNX Runtime C++.

use std::ffi::{c_char, CStr, CString};
use std::path::Path;
use std::sync::OnceLock;

use image::imageops::FilterType;
use ndarray::Array4;
use parking_lot::Mutex;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::internal::format_err;

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

type RunnableModel =
    SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

struct LoadedModel {
    plan: RunnableModel,
    input_height: usize,
    input_width: usize,
    embed_dim: usize,
}

struct Registry {
    models: Mutex<std::collections::HashMap<String, LoadedModel>>,
    default_name: Mutex<Option<String>>,
}

static REGISTRY: OnceLock<Registry> = OnceLock::new();
static LAST_ERROR: OnceLock<Mutex<String>> = OnceLock::new();

fn registry() -> &'static Registry {
    REGISTRY.get_or_init(|| Registry {
        models: Mutex::new(std::collections::HashMap::new()),
        default_name: Mutex::new(None),
    })
}

fn set_last_error(msg: impl Into<String>) {
    let cell = LAST_ERROR.get_or_init(|| Mutex::new(String::new()));
    *cell.lock() = msg.into();
}

fn get_last_error() -> String {
    LAST_ERROR
        .get()
        .map(|m| m.lock().clone())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

unsafe fn cstr_to_str<'a>(p: *const c_char) -> Result<&'a str, &'static str> {
    if p.is_null() {
        return Err("null pointer");
    }
    CStr::from_ptr(p).to_str().map_err(|_| "invalid utf-8")
}

fn build_runnable_from_path(path: &Path) -> TractResult<(RunnableModel, usize, usize, usize)> {
    let mut typed = tract_onnx::onnx().model_for_path(path)?;
    finalize(&mut typed)
}

fn build_runnable_from_bytes(bytes: &[u8]) -> TractResult<(RunnableModel, usize, usize, usize)> {
    let mut reader = std::io::Cursor::new(bytes);
    let mut typed = tract_onnx::onnx().model_for_read(&mut reader)?;
    finalize(&mut typed)
}

fn finalize(
    model: &mut tract_onnx::tract_hir::infer::InferenceModel,
) -> TractResult<(RunnableModel, usize, usize, usize)> {
    // Try common input sizes in order; first one that yields a runnable plan wins.
    // EUPE ViT-T is 224, EUPE ViT-S/B are 256, some torchvision models use 384.
    for &(h, w) in &[(224usize, 224usize), (256, 256), (384, 384)] {
        let mut candidate = model.clone();
        if candidate
            .set_input_fact(0, f32::fact([1, 3, h, w]).into())
            .is_err()
        {
            continue;
        }
        let typed = match candidate.into_typed() {
            Ok(t) => t,
            Err(_) => continue,
        };
        let optimized = match typed.into_optimized() {
            Ok(o) => o,
            Err(_) => continue,
        };
        let plan = match optimized.into_runnable() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let dummy = Array4::<f32>::zeros((1, 3, h, w));
        if let Ok(out) = plan.run(tvec!(Tensor::from(dummy).into())) {
            let view = out[0].to_array_view::<f32>()?;
            let embed_dim = view.shape().last().copied().unwrap_or(0);
            return Ok((plan, h, w, embed_dim));
        }
    }
    Err(format_err!("could not infer input H/W (tried 224, 256, 384); model may need a custom shape"))
}

fn preprocess_image(
    bytes: &[u8],
    target_h: usize,
    target_w: usize,
) -> Result<Array4<f32>, String> {
    let img = image::load_from_memory(bytes).map_err(|e| format!("decode: {e}"))?;
    let resized = img.resize_exact(target_w as u32, target_h as u32, FilterType::Triangle);
    let rgb = resized.to_rgb8();

    // Build [1, 3, H, W] with ImageNet normalization.
    let mut out = Array4::<f32>::zeros((1, 3, target_h, target_w));
    let (w, h) = rgb.dimensions();
    debug_assert_eq!(w as usize, target_w);
    debug_assert_eq!(h as usize, target_h);
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x, y);
            for c in 0..3 {
                let v = p[c] as f32 / 255.0;
                out[[0, c, y as usize, x as usize]] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }
    Ok(out)
}

fn embed_inner(
    image_bytes: &[u8],
    model_name: Option<&str>,
) -> Result<Vec<f32>, String> {
    let reg = registry();
    let resolved_name: String = match model_name {
        Some(n) => n.to_string(),
        None => reg
            .default_name
            .lock()
            .clone()
            .ok_or_else(|| "no model loaded; call load_model first".to_string())?,
    };
    let models = reg.models.lock();
    let model = models
        .get(&resolved_name)
        .ok_or_else(|| format!("model '{resolved_name}' not loaded"))?;

    let input = preprocess_image(image_bytes, model.input_height, model.input_width)?;
    let out = model
        .plan
        .run(tvec!(Tensor::from(input).into()))
        .map_err(|e| format!("inference: {e}"))?;
    let view = out[0]
        .to_array_view::<f32>()
        .map_err(|e| format!("output view: {e}"))?;
    Ok(view.iter().copied().collect())
}

// ---------------------------------------------------------------------------
// FFI - Model management
// ---------------------------------------------------------------------------

/// Load an ONNX model from a file path. Returns 0 on success, -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn pic2vec_rs_load_model(
    path: *const c_char,
    name: *const c_char,
) -> i32 {
    let path_str = match cstr_to_str(path) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("path: {e}"));
            return -1;
        }
    };
    let name_str = match cstr_to_str(name) {
        Ok(s) => s.to_string(),
        Err(e) => {
            set_last_error(format!("name: {e}"));
            return -1;
        }
    };

    match build_runnable_from_path(Path::new(path_str)) {
        Ok((plan, h, w, dim)) => {
            let reg = registry();
            let mut models = reg.models.lock();
            models.insert(
                name_str.clone(),
                LoadedModel {
                    plan,
                    input_height: h,
                    input_width: w,
                    embed_dim: dim,
                },
            );
            let mut default = reg.default_name.lock();
            if default.is_none() {
                *default = Some(name_str);
            }
            0
        }
        Err(e) => {
            set_last_error(format!("load: {e}"));
            -1
        }
    }
}

/// Load an ONNX model from an in-memory byte buffer. Returns 0/-1.
#[no_mangle]
pub unsafe extern "C" fn pic2vec_rs_load_model_from_memory(
    data: *const u8,
    size: usize,
    name: *const c_char,
) -> i32 {
    if data.is_null() || size == 0 {
        set_last_error("empty model bytes");
        return -1;
    }
    let bytes = std::slice::from_raw_parts(data, size);
    let name_str = match cstr_to_str(name) {
        Ok(s) => s.to_string(),
        Err(e) => {
            set_last_error(format!("name: {e}"));
            return -1;
        }
    };

    match build_runnable_from_bytes(bytes) {
        Ok((plan, h, w, dim)) => {
            let reg = registry();
            let mut models = reg.models.lock();
            models.insert(
                name_str.clone(),
                LoadedModel {
                    plan,
                    input_height: h,
                    input_width: w,
                    embed_dim: dim,
                },
            );
            let mut default = reg.default_name.lock();
            if default.is_none() {
                *default = Some(name_str);
            }
            0
        }
        Err(e) => {
            set_last_error(format!("load_from_memory: {e}"));
            -1
        }
    }
}

/// Unload a model. Returns 1 if removed, 0 if not found.
#[no_mangle]
pub unsafe extern "C" fn pic2vec_rs_unload_model(name: *const c_char) -> i32 {
    let name_str = match cstr_to_str(name) {
        Ok(s) => s.to_string(),
        Err(_) => return 0,
    };
    let reg = registry();
    let mut models = reg.models.lock();
    let removed = models.remove(&name_str).is_some();
    if removed {
        let mut default = reg.default_name.lock();
        if default.as_deref() == Some(name_str.as_str()) {
            *default = models.keys().next().cloned();
        }
        1
    } else {
        0
    }
}

/// List loaded models as a newline-separated UTF-8 string. Caller must free with pic2vec_rs_free_string.
#[no_mangle]
pub extern "C" fn pic2vec_rs_list_models() -> *mut c_char {
    let reg = registry();
    let models = reg.models.lock();
    let names: Vec<String> = models.keys().cloned().collect();
    let joined = names.join("\n");
    CString::new(joined)
        .map(|s| s.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

/// Get embed_dim of a loaded model (or 0 if not found).
#[no_mangle]
pub unsafe extern "C" fn pic2vec_rs_model_dim(name: *const c_char) -> usize {
    let name_str = match cstr_to_str(name) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let reg = registry();
    let models = reg.models.lock();
    models.get(name_str).map(|m| m.embed_dim).unwrap_or(0)
}

/// Last error message. Caller must free with pic2vec_rs_free_string.
#[no_mangle]
pub extern "C" fn pic2vec_rs_last_error() -> *mut c_char {
    CString::new(get_last_error())
        .map(|s| s.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

/// Free a string returned by this library.
#[no_mangle]
pub unsafe extern "C" fn pic2vec_rs_free_string(p: *mut c_char) {
    if !p.is_null() {
        let _ = CString::from_raw(p);
    }
}

// ---------------------------------------------------------------------------
// FFI - Inference
// ---------------------------------------------------------------------------

/// Embed an image from a file path. On success writes embedding to *out_data
/// (heap-allocated; caller must free with pic2vec_rs_free_floats) and length
/// to *out_len. Returns 0/-1.
#[no_mangle]
pub unsafe extern "C" fn pic2vec_rs_embed_path(
    image_path: *const c_char,
    model_name: *const c_char,
    out_data: *mut *mut f32,
    out_len: *mut usize,
) -> i32 {
    let path_str = match cstr_to_str(image_path) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("image_path: {e}"));
            return -1;
        }
    };
    let name_opt = if model_name.is_null() {
        None
    } else {
        match cstr_to_str(model_name) {
            Ok(s) => Some(s),
            Err(e) => {
                set_last_error(format!("model_name: {e}"));
                return -1;
            }
        }
    };

    let bytes = match std::fs::read(path_str) {
        Ok(b) => b,
        Err(e) => {
            set_last_error(format!("read {path_str}: {e}"));
            return -1;
        }
    };

    match embed_inner(&bytes, name_opt) {
        Ok(vec) => {
            let mut boxed = vec.into_boxed_slice();
            *out_data = boxed.as_mut_ptr();
            *out_len = boxed.len();
            std::mem::forget(boxed);
            0
        }
        Err(e) => {
            set_last_error(e);
            -1
        }
    }
}

/// Embed an image from a byte buffer.
#[no_mangle]
pub unsafe extern "C" fn pic2vec_rs_embed_blob(
    data: *const u8,
    size: usize,
    model_name: *const c_char,
    out_data: *mut *mut f32,
    out_len: *mut usize,
) -> i32 {
    if data.is_null() || size == 0 {
        set_last_error("empty blob");
        return -1;
    }
    let bytes = std::slice::from_raw_parts(data, size);
    let name_opt = if model_name.is_null() {
        None
    } else {
        match cstr_to_str(model_name) {
            Ok(s) => Some(s),
            Err(e) => {
                set_last_error(format!("model_name: {e}"));
                return -1;
            }
        }
    };

    match embed_inner(bytes, name_opt) {
        Ok(vec) => {
            let mut boxed = vec.into_boxed_slice();
            *out_data = boxed.as_mut_ptr();
            *out_len = boxed.len();
            std::mem::forget(boxed);
            0
        }
        Err(e) => {
            set_last_error(e);
            -1
        }
    }
}

/// Free a float array returned by pic2vec_rs_embed_*.
#[no_mangle]
pub unsafe extern "C" fn pic2vec_rs_free_floats(ptr: *mut f32, len: usize) {
    if !ptr.is_null() && len > 0 {
        let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, len));
    }
}

// ---------------------------------------------------------------------------
// FFI - Patch-level diff
// ---------------------------------------------------------------------------

/// Compare two images patch-by-patch.
/// Requires the loaded model to have a second output containing patch
/// tokens of shape [1, num_patches, dim] (e.g. EUPE re-exported with
/// `output_names=['cls','patches']`).
/// Writes a heap-allocated f32 array of length `num_patches` containing
/// per-patch cosine similarity to *out_sims, and `num_patches` to *out_n.
/// Caller MUST free with pic2vec_rs_free_floats.
/// Returns 0/-1.
#[no_mangle]
pub unsafe extern "C" fn pic2vec_rs_diff_patches(
    path_a: *const c_char,
    path_b: *const c_char,
    model_name: *const c_char,
    out_sims: *mut *mut f32,
    out_n: *mut usize,
) -> i32 {
    let pa = match cstr_to_str(path_a) {
        Ok(s) => s,
        Err(e) => { set_last_error(format!("path_a: {e}")); return -1; }
    };
    let pb = match cstr_to_str(path_b) {
        Ok(s) => s,
        Err(e) => { set_last_error(format!("path_b: {e}")); return -1; }
    };
    let name_opt = if model_name.is_null() {
        None
    } else {
        match cstr_to_str(model_name) {
            Ok(s) => Some(s),
            Err(e) => { set_last_error(format!("model_name: {e}")); return -1; }
        }
    };

    let reg = registry();
    let resolved: String = match name_opt {
        Some(n) => n.to_string(),
        None => match reg.default_name.lock().clone() {
            Some(n) => n,
            None => { set_last_error("no model loaded"); return -1; }
        }
    };
    let models = reg.models.lock();
    let model = match models.get(&resolved) {
        Some(m) => m,
        None => { set_last_error(format!("model '{resolved}' not loaded")); return -1; }
    };

    // Helper: run inference, return patches tensor as Vec<f32>.
    let run_for_patches = |path: &str| -> Result<(Vec<f32>, usize, usize), String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        let img = preprocess_image(&bytes, model.input_height, model.input_width)
            .map_err(|e| e)?;
        let outs = model.plan.run(tvec!(Tensor::from(img).into()))
            .map_err(|e| format!("inference: {e}"))?;
        // Expect at least 2 outputs: [0]=cls, [1]=patches with shape [1, N, dim]
        if outs.len() < 2 {
            return Err("model has no patch output (expected 2 outputs: cls, patches)".to_string());
        }
        let view = outs[1].to_array_view::<f32>()
            .map_err(|e| format!("patches view: {e}"))?;
        let shape = view.shape();
        if shape.len() != 3 {
            return Err(format!("unexpected patch tensor shape {shape:?} (expected [1, N, dim])"));
        }
        let n = shape[1];
        let dim = shape[2];
        Ok((view.iter().copied().collect(), n, dim))
    };

    let (vec_a, n_a, dim_a) = match run_for_patches(pa) { Ok(t) => t, Err(e) => { set_last_error(e); return -1; } };
    let (vec_b, n_b, dim_b) = match run_for_patches(pb) { Ok(t) => t, Err(e) => { set_last_error(e); return -1; } };
    if n_a != n_b || dim_a != dim_b {
        set_last_error(format!("patch shape mismatch a=[{n_a},{dim_a}] b=[{n_b},{dim_b}]"));
        return -1;
    }

    let mut sims = Vec::<f32>::with_capacity(n_a);
    for i in 0..n_a {
        let a = &vec_a[i * dim_a..(i + 1) * dim_a];
        let b = &vec_b[i * dim_a..(i + 1) * dim_a];
        let mut dot: f32 = 0.0;
        let mut na: f32 = 0.0;
        let mut nb: f32 = 0.0;
        for k in 0..dim_a {
            dot += a[k] * b[k];
            na += a[k] * a[k];
            nb += b[k] * b[k];
        }
        let denom = na.sqrt() * nb.sqrt();
        sims.push(if denom < 1e-12 { 0.0 } else { dot / denom });
    }

    let mut boxed = sims.into_boxed_slice();
    *out_sims = boxed.as_mut_ptr();
    *out_n = boxed.len();
    std::mem::forget(boxed);
    0
}

// ---------------------------------------------------------------------------
// FFI - diff_save (annotated rectangle overlay)
// ---------------------------------------------------------------------------

fn draw_rect_border(
    img: &mut image::RgbImage,
    x0: u32, y0: u32, x1: u32, y1: u32,
    color: image::Rgb<u8>,
    thickness: u32,
) {
    let (w, h) = img.dimensions();
    let x0 = x0.min(w.saturating_sub(1));
    let y0 = y0.min(h.saturating_sub(1));
    let x1 = x1.min(w.saturating_sub(1));
    let y1 = y1.min(h.saturating_sub(1));
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    for t in 0..thickness {
        // top + bottom
        for x in x0..=x1 {
            if y0 + t < h { img.put_pixel(x, y0 + t, color); }
            if y1 >= t { img.put_pixel(x, y1 - t, color); }
        }
        // left + right
        for y in y0..=y1 {
            if x0 + t < w { img.put_pixel(x0 + t, y, color); }
            if x1 >= t { img.put_pixel(x1 - t, y, color); }
        }
    }
}

/// Draw red rectangles on each input image at the patches whose cosine
/// similarity is below `threshold`, then write the annotated copies to
/// `out_a` / `out_b`. Returns the count of highlighted patches on success,
/// or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn pic2vec_rs_diff_save(
    path_a: *const c_char,
    path_b: *const c_char,
    out_a: *const c_char,
    out_b: *const c_char,
    model_name: *const c_char,
    threshold: f64,
) -> i32 {
    let pa = match cstr_to_str(path_a) { Ok(s) => s, Err(e) => { set_last_error(format!("path_a: {e}")); return -1; } };
    let pb = match cstr_to_str(path_b) { Ok(s) => s, Err(e) => { set_last_error(format!("path_b: {e}")); return -1; } };
    let oa = match cstr_to_str(out_a) { Ok(s) => s, Err(e) => { set_last_error(format!("out_a: {e}")); return -1; } };
    let ob = match cstr_to_str(out_b) { Ok(s) => s, Err(e) => { set_last_error(format!("out_b: {e}")); return -1; } };

    // Reuse the patch-diff routine to get per-patch sims.
    let mut sims_ptr: *mut f32 = std::ptr::null_mut();
    let mut n: usize = 0;
    let rc = pic2vec_rs_diff_patches(path_a, path_b, model_name, &mut sims_ptr, &mut n);
    if rc != 0 {
        return -1;
    }
    let sims_slice = std::slice::from_raw_parts(sims_ptr, n).to_vec();
    pic2vec_rs_free_floats(sims_ptr, n);

    let side = (n as f64).sqrt() as usize;
    if side * side != n {
        set_last_error(format!("non-square patch grid ({n})"));
        return -1;
    }

    // Find which patches to highlight + the most-different one (for a thicker border).
    let mut diffs: Vec<(usize, usize, f32)> = Vec::new();
    let mut min_idx = 0usize;
    let mut min_val = f32::MAX;
    for k in 0..n {
        if sims_slice[k] < min_val {
            min_val = sims_slice[k];
            min_idx = k;
        }
        if (sims_slice[k] as f64) < threshold {
            diffs.push((k / side, k % side, sims_slice[k]));
        }
    }

    let red = image::Rgb([255u8, 0, 0]);
    let dark_red = image::Rgb([180u8, 0, 0]);

    let render = |src: &str, dst: &str| -> Result<(), String> {
        let mut img = image::open(src)
            .map_err(|e| format!("open {src}: {e}"))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        for &(r, c, _sim) in &diffs {
            let x0 = (c as u32 * w) / side as u32;
            let y0 = (r as u32 * h) / side as u32;
            let x1 = ((c as u32 + 1) * w) / side as u32 - 1;
            let y1 = ((r as u32 + 1) * h) / side as u32 - 1;
            // Thicker border + darker color for the most-different patch.
            let is_min = (r * side + c) == min_idx;
            let color = if is_min { red } else { dark_red };
            let thick = if is_min { 5 } else { 3 };
            draw_rect_border(&mut img, x0, y0, x1, y1, color, thick);
        }
        img.save(dst).map_err(|e| format!("save {dst}: {e}"))?;
        Ok(())
    };

    if let Err(e) = render(pa, oa) { set_last_error(e); return -1; }
    if let Err(e) = render(pb, ob) { set_last_error(e); return -1; }

    diffs.len() as i32
}

/// Render a heatmap overlay: each 14x14 patch is tinted red proportionally
/// to (1 - similarity). Already-similar patches are barely tinted; very
/// different patches are saturated red. Writes annotated copies to `out_a`
/// and `out_b`. Returns 0 on success, -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn pic2vec_rs_diff_heatmap(
    path_a: *const c_char,
    path_b: *const c_char,
    out_a: *const c_char,
    out_b: *const c_char,
    model_name: *const c_char,
) -> i32 {
    let pa = match cstr_to_str(path_a) { Ok(s) => s, Err(e) => { set_last_error(format!("path_a: {e}")); return -1; } };
    let pb = match cstr_to_str(path_b) { Ok(s) => s, Err(e) => { set_last_error(format!("path_b: {e}")); return -1; } };
    let oa = match cstr_to_str(out_a) { Ok(s) => s, Err(e) => { set_last_error(format!("out_a: {e}")); return -1; } };
    let ob = match cstr_to_str(out_b) { Ok(s) => s, Err(e) => { set_last_error(format!("out_b: {e}")); return -1; } };

    let mut sims_ptr: *mut f32 = std::ptr::null_mut();
    let mut n: usize = 0;
    let rc = pic2vec_rs_diff_patches(path_a, path_b, model_name, &mut sims_ptr, &mut n);
    if rc != 0 {
        return -1;
    }
    let sims = std::slice::from_raw_parts(sims_ptr, n).to_vec();
    pic2vec_rs_free_floats(sims_ptr, n);

    let side = (n as f64).sqrt() as usize;
    if side * side != n {
        set_last_error(format!("non-square patch grid ({n})"));
        return -1;
    }

    // Normalize so the most-different patch maps to alpha 1.0 - this gives
    // visible signal even when all patches are very similar (e.g. avg=0.999).
    let min_sim = sims.iter().copied().fold(f32::INFINITY, f32::min);
    let max_diff = (1.0 - min_sim).max(1e-6);

    let render = |src: &str, dst: &str| -> Result<(), String> {
        let mut img = image::open(src)
            .map_err(|e| format!("open {src}: {e}"))?
            .to_rgb8();
        let (w, h) = img.dimensions();

        for r in 0..side {
            for c in 0..side {
                let sim = sims[r * side + c];
                let diff = (1.0 - sim).max(0.0);
                // intensity in [0, 1]; non-linear curve to dampen near-equal patches.
                let intensity = (diff / max_diff).powf(0.7);
                if intensity < 0.05 { continue; }
                let alpha = (255.0 * 0.55 * intensity) as u32; // cap alpha at ~140

                let x0 = (c as u32 * w) / side as u32;
                let y0 = (r as u32 * h) / side as u32;
                let x1 = ((c as u32 + 1) * w) / side as u32;
                let y1 = ((r as u32 + 1) * h) / side as u32;

                for y in y0..y1.min(h) {
                    for x in x0..x1.min(w) {
                        let p = img.get_pixel_mut(x, y);
                        // Blend red (255, 0, 0) onto p with computed alpha.
                        let a = alpha;
                        let inv = 255 - a as u32;
                        p[0] = ((255 * a + p[0] as u32 * inv) / 255) as u8;
                        p[1] = (p[1] as u32 * inv / 255) as u8;
                        p[2] = (p[2] as u32 * inv / 255) as u8;
                    }
                }
            }
        }
        img.save(dst).map_err(|e| format!("save {dst}: {e}"))?;
        Ok(())
    };

    if let Err(e) = render(pa, oa) { set_last_error(e); return -1; }
    if let Err(e) = render(pb, ob) { set_last_error(e); return -1; }
    0
}

// ---------------------------------------------------------------------------
// FFI - Image info
// ---------------------------------------------------------------------------

/// Probe an image's metadata. Outputs are written to *out_width, *out_height,
/// *out_channels (1=L, 2=LA, 3=RGB, 4=RGBA), and a heap-allocated format
/// string is written to *out_format (caller frees with pic2vec_rs_free_string).
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn pic2vec_rs_image_info(
    path: *const c_char,
    out_width: *mut u32,
    out_height: *mut u32,
    out_channels: *mut u32,
    out_format: *mut *mut c_char,
) -> i32 {
    let path_str = match cstr_to_str(path) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("path: {e}"));
            return -1;
        }
    };
    let bytes = match std::fs::read(path_str) {
        Ok(b) => b,
        Err(e) => {
            set_last_error(format!("read {path_str}: {e}"));
            return -1;
        }
    };
    let format = match image::guess_format(&bytes) {
        Ok(f) => format!("{f:?}").to_lowercase(),
        Err(_) => "unknown".to_string(),
    };
    let img = match image::load_from_memory(&bytes) {
        Ok(i) => i,
        Err(e) => {
            set_last_error(format!("decode: {e}"));
            return -1;
        }
    };
    *out_width = img.width();
    *out_height = img.height();
    *out_channels = match img.color() {
        image::ColorType::L8 | image::ColorType::L16 => 1,
        image::ColorType::La8 | image::ColorType::La16 => 2,
        image::ColorType::Rgb8 | image::ColorType::Rgb16 | image::ColorType::Rgb32F => 3,
        image::ColorType::Rgba8 | image::ColorType::Rgba16 | image::ColorType::Rgba32F => 4,
        _ => 0,
    };
    *out_format = match CString::new(format) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    };
    0
}

// ---------------------------------------------------------------------------
// FFI - Perceptual hash (dHash, 64-bit)
// ---------------------------------------------------------------------------

/// Compute a 64-bit perceptual hash (dHash) of an image.
/// dHash: resize to 9x8 grayscale, compare adjacent pixels horizontally,
/// each comparison contributes 1 bit -> 64 bits total.
/// Two images with Hamming distance <= ~10 are typically near-duplicates.
/// Returns the hash on success, 0 on failure (and sets last_error).
#[no_mangle]
pub unsafe extern "C" fn pic2vec_rs_phash(path: *const c_char, out_hash: *mut u64) -> i32 {
    let path_str = match cstr_to_str(path) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("path: {e}"));
            return -1;
        }
    };
    let bytes = match std::fs::read(path_str) {
        Ok(b) => b,
        Err(e) => {
            set_last_error(format!("read {path_str}: {e}"));
            return -1;
        }
    };
    let img = match image::load_from_memory(&bytes) {
        Ok(i) => i,
        Err(e) => {
            set_last_error(format!("decode: {e}"));
            return -1;
        }
    };
    // Resize to 9x8 grayscale
    let small = img
        .resize_exact(9, 8, FilterType::Triangle)
        .to_luma8();
    let mut hash: u64 = 0;
    for y in 0..8 {
        for x in 0..8 {
            let left = small.get_pixel(x, y)[0];
            let right = small.get_pixel(x + 1, y)[0];
            if left > right {
                hash |= 1u64 << (y * 8 + x);
            }
        }
    }
    *out_hash = hash;
    0
}

// ---------------------------------------------------------------------------
// FFI - Batch embedding
// ---------------------------------------------------------------------------

/// Embed multiple image paths in one call. Output is a single contiguous
/// f32 array of length n_paths * embed_dim (caller writes embed_dim by
/// reading from a single-image embed first, or by calling pic2vec_rs_model_dim).
/// `paths` is an array of n null-terminated UTF-8 strings.
/// Writes the heap-allocated buffer to *out_data and total length to *out_len.
/// Returns 0/-1.
#[no_mangle]
pub unsafe extern "C" fn pic2vec_rs_embed_batch(
    paths: *const *const c_char,
    n_paths: usize,
    model_name: *const c_char,
    out_data: *mut *mut f32,
    out_total_len: *mut usize,
    out_per_embed: *mut usize,
) -> i32 {
    if paths.is_null() || n_paths == 0 {
        set_last_error("empty batch");
        return -1;
    }
    let name_opt = if model_name.is_null() {
        None
    } else {
        match cstr_to_str(model_name) {
            Ok(s) => Some(s),
            Err(e) => {
                set_last_error(format!("model_name: {e}"));
                return -1;
            }
        }
    };

    let mut all = Vec::<f32>::new();
    let mut per: usize = 0;
    for i in 0..n_paths {
        let p = *paths.add(i);
        let path_str = match cstr_to_str(p) {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("paths[{i}]: {e}"));
                return -1;
            }
        };
        let bytes = match std::fs::read(path_str) {
            Ok(b) => b,
            Err(e) => {
                set_last_error(format!("read paths[{i}] {path_str}: {e}"));
                return -1;
            }
        };
        match embed_inner(&bytes, name_opt) {
            Ok(v) => {
                if per == 0 {
                    per = v.len();
                } else if v.len() != per {
                    set_last_error(format!(
                        "paths[{i}]: dim mismatch ({} vs {per})",
                        v.len()
                    ));
                    return -1;
                }
                all.extend(v);
            }
            Err(e) => {
                set_last_error(format!("paths[{i}]: {e}"));
                return -1;
            }
        }
    }

    let mut boxed = all.into_boxed_slice();
    *out_data = boxed.as_mut_ptr();
    *out_total_len = boxed.len();
    *out_per_embed = per;
    std::mem::forget(boxed);
    0
}
