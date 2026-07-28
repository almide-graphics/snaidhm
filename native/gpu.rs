//! Native (wgpu) implementation of snaidhm's `gpu` extern namespace.
//!
//! Every `@extern(wasm, "gpu", "<name>")` in `src/web/gpu.almd` carries a
//! matching `@extern(rust, "crate::gpu", "<name>")`, so the same Almide source
//! compiles for wasm (JS host: `ceangal/examples/demo/ceangal_runtime.js`) and
//! for native. This module is the native host, and it mirrors that JS host's
//! semantics function for function.
//!
//! ## Design
//!
//! * **One handle table.** Like the JS host's `handles` array, every GPU object
//!   lives in a single `Vec<Res>` indexed by an `i64` handle. Index 0 is the
//!   null handle, so a handle of `0` is always "nothing" — the same convention
//!   the JS host uses.
//! * **Process-global, lazily initialised.** The externs are free functions, so
//!   the runtime hangs off a `OnceLock<Mutex<GpuState>>`. wgpu's `Device`,
//!   `Queue`, `RenderPass<'static>` and friends are all `Send + Sync`, so the
//!   mutex is sound. Adapter/device acquisition happens on first use via
//!   `pollster::block_on`.
//! * **Never panics on first use.** If no adapter or device can be acquired
//!   (headless CI, no GPU), `ensure_device` records the failure once, warns on
//!   stderr, and every entry point degrades to a no-op returning the null
//!   handle instead of aborting the host program.
//!
//! ## Scope: headless
//!
//! There is no window or surface yet. `configure_canvas` therefore creates an
//! offscreen colour target instead of configuring a swapchain, and
//! `begin_render_pass` renders into that target. Wiring a real winit surface is
//! the next task; the only functions that need to change for it are
//! `configure_canvas`, `ensure_target` and `finish_and_submit` (to present).

use std::sync::{Mutex, OnceLock};

// ── Vendored WGSL ─────────────────────────────────────────────────────────
//
// The JS host fetches these over HTTP and builds
// `SHADERS = [raster, raster, text, image]`. `create_shader`'s second argument
// is that array index (NOT a pointer), and the JS host falls back to index 0
// for an out-of-range index. snaidhm calls `create_shader(device, 0, 0)`,
// `(device, 1, 0)` and `(device, 3, 0)`; the duplicated raster entry at index 1
// is what makes index 1 the quad/fullscreen shader. Mirrored exactly.

const RASTER_WGSL: &str = include_str!("wgsl/raster.wgsl");
const TEXT_WGSL: &str = include_str!("wgsl/text.wgsl");
const IMAGE_WGSL: &str = include_str!("wgsl/image.wgsl");

const SHADERS: [&str; 4] = [RASTER_WGSL, RASTER_WGSL, TEXT_WGSL, IMAGE_WGSL];

/// Colour format of the offscreen target. `get_preferred_format` hands back a
/// handle to exactly this, and every pipeline's colour target uses it.
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

/// Offscreen target size. A surface would supply this; headless, it is a
/// default. Both dimensions are multiples of the raster shader's 16x16
/// workgroup so a full-coverage dispatch tiles the target exactly.
const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;

// ── Handle table ──────────────────────────────────────────────────────────

/// A pass in flight. `forget_lifetime()` detaches the pass from the `&mut`
/// borrow of its encoder so both can live in the handle table at once; wgpu
/// still enforces "no encoder use while a pass is open" at runtime.
enum Pass {
    Render(wgpu::RenderPass<'static>),
    Compute(wgpu::ComputePass<'static>),
}

/// One slot of the handle table.
///
/// `Encoder`/`Pass` are `Option` because they are consumed (`finish`, or
/// dropped to end the pass) while their handle is still live — taking the
/// payload out also lets a method hold `&mut` on one slot and `&` on another
/// without fighting the borrow checker.
enum Res {
    Null,
    /// Stand-in for the canvas context the JS host returns from
    /// `configure_canvas`. Nothing reads through it; it only has to be a
    /// distinct non-zero handle.
    Context,
    Format(wgpu::TextureFormat),
    Shader(wgpu::ShaderModule),
    Buffer(wgpu::Buffer),
    ComputePipeline(wgpu::ComputePipeline),
    RenderPipeline(wgpu::RenderPipeline),
    BindGroup(wgpu::BindGroup),
    Texture(wgpu::Texture),
    Sampler(wgpu::Sampler),
    Encoder(Option<wgpu::CommandEncoder>),
    Pass(Option<Pass>),
}

/// One entry accumulated by `begin_bindings` / `add_*_binding`, drained by
/// `create_bound_group`. Stored as handles, resolved at bind-group creation —
/// same as the JS host's `_bindingEntries`.
enum Binding {
    Buffer(i64),
    Texture(i64),
    Sampler(i64),
}

// ── Runtime state ─────────────────────────────────────────────────────────

#[derive(Default)]
struct GpuState {
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    /// Set once adapter/device acquisition has failed, so we warn once and
    /// then degrade quietly instead of re-probing on every call.
    init_failed: bool,

    res: Vec<Res>,
    /// Slots emptied by `end_pass` / `finish_and_submit`. Encoders and passes
    /// are strictly frame-scoped, so recycling their slots keeps the table from
    /// growing without bound across frames. Long-lived resources (buffers,
    /// pipelines, ...) are never released and so never recycled.
    free_slots: Vec<usize>,

    /// Offscreen colour target created by `configure_canvas`.
    target: Option<wgpu::Texture>,
    target_view: Option<wgpu::TextureView>,
    /// Depth target for the 3D pass, with the size it was built for. Rebuilt by
    /// `set_depth_size` when that size changes; `None` until first asked for.
    depth_view: Option<wgpu::TextureView>,
    depth_size: (u32, u32),
    /// Cached handle returned by `get_preferred_format`, so repeated calls
    /// return the same value (the JS host allocates a fresh handle each time;
    /// a stable one is strictly better and nothing depends on the difference).
    format_handle: i64,

    pending_bindings: Vec<Binding>,

    /// `begin_data` / `push_f32` / `push_u32` staging area. Values are stored
    /// as raw 32-bit patterns, which is what the JS host's Float32Array /
    /// Uint32Array views over one ArrayBuffer amount to.
    data: Vec<u32>,

    /// 1x1 transparent texture and a linear sampler, used when a texture or
    /// sampler handle does not resolve. Natively nothing creates textures yet
    /// (the `gpu` namespace has no texture-creation entry point — the JS host
    /// makes them outside it and passes handles in), so without these a
    /// `create_bound_group` over a texture binding could not be satisfied.
    fallback_texture: Option<wgpu::Texture>,
    fallback_sampler: Option<wgpu::Sampler>,
}

fn state() -> &'static Mutex<GpuState> {
    static GPU: OnceLock<Mutex<GpuState>> = OnceLock::new();
    GPU.get_or_init(|| Mutex::new(GpuState::default()))
}

/// Run `f` against the global runtime. Poisoning is recovered from rather than
/// propagated: a panic in one extern must not turn every later GPU call into a
/// second panic.
fn with<R>(default: R, f: impl FnOnce(&mut GpuState) -> R) -> R {
    let mut guard = match state().lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    if !guard.ensure_device() {
        return default;
    }
    f(&mut guard)
}

impl GpuState {
    // ── Device ────────────────────────────────────────────────────────────

    /// Acquire instance → adapter → device → queue on first use. Returns
    /// `false` (without panicking) when there is no usable adapter.
    fn ensure_device(&mut self) -> bool {
        if self.device.is_some() {
            return true;
        }
        if self.init_failed {
            return false;
        }
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }));
        let Some(adapter) = adapter else {
            self.init_failed = true;
            eprintln!("[snaidhm/gpu] no wgpu adapter available — GPU calls are no-ops");
            return false;
        };
        // The raster shader binds 12 storage buffers across groups 0/1/3, over
        // the 8 of `Limits::default()`. The JS host asks for
        // `maxStorageBuffersPerShaderStage: 10`; natively, ask the adapter for
        // everything it has.
        let desc = wgpu::DeviceDescriptor {
            label: Some("snaidhm"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::default(),
        };
        match pollster::block_on(adapter.request_device(&desc, None)) {
            Ok((device, queue)) => {
                self.device = Some(device);
                self.queue = Some(queue);
                if self.res.is_empty() {
                    self.res.push(Res::Null); // handle 0 == null
                }
                true
            }
            Err(e) => {
                self.init_failed = true;
                eprintln!("[snaidhm/gpu] wgpu device request failed: {e} — GPU calls are no-ops");
                false
            }
        }
    }

    fn device(&self) -> &wgpu::Device {
        self.device.as_ref().expect("device checked by `with`")
    }

    fn queue(&self) -> &wgpu::Queue {
        self.queue.as_ref().expect("queue checked by `with`")
    }

    // ── Handle table ──────────────────────────────────────────────────────

    fn alloc(&mut self, res: Res) -> i64 {
        if let Some(slot) = self.free_slots.pop() {
            self.res[slot] = res;
            return slot as i64;
        }
        self.res.push(res);
        (self.res.len() - 1) as i64
    }

    /// Release a frame-scoped slot (pass or encoder) back to the free list.
    ///
    /// Any other slot kind — including one that is already `Null` — is left
    /// alone, so a double `end_pass` cannot push the same index onto the free
    /// list twice and hand one slot out as two live handles.
    fn release(&mut self, handle: i64) {
        let Some(idx) = self.index(handle) else { return };
        if !matches!(self.res[idx], Res::Pass(_) | Res::Encoder(_)) {
            return;
        }
        self.res[idx] = Res::Null;
        self.free_slots.push(idx);
    }

    /// Resolve a handle to a table index. Handle 0 and out-of-range handles
    /// resolve to nothing.
    fn index(&self, handle: i64) -> Option<usize> {
        if handle <= 0 {
            return None;
        }
        let idx = handle as usize;
        if idx < self.res.len() { Some(idx) } else { None }
    }

    fn get(&self, handle: i64) -> Option<&Res> {
        self.index(handle).map(|i| &self.res[i])
    }

    fn buffer(&self, handle: i64) -> Option<&wgpu::Buffer> {
        match self.get(handle) {
            Some(Res::Buffer(b)) => Some(b),
            _ => None,
        }
    }

    fn take_encoder(&mut self, handle: i64) -> Option<wgpu::CommandEncoder> {
        match self.index(handle).map(|i| &mut self.res[i]) {
            Some(Res::Encoder(slot)) => slot.take(),
            _ => None,
        }
    }

    fn put_encoder(&mut self, handle: i64, enc: wgpu::CommandEncoder) {
        if let Some(Res::Encoder(slot)) = self.index(handle).map(|i| &mut self.res[i]) {
            *slot = Some(enc);
        }
    }

    fn take_pass(&mut self, handle: i64) -> Option<Pass> {
        match self.index(handle).map(|i| &mut self.res[i]) {
            Some(Res::Pass(slot)) => slot.take(),
            _ => None,
        }
    }

    fn put_pass(&mut self, handle: i64, pass: Pass) {
        if let Some(Res::Pass(slot)) = self.index(handle).map(|i| &mut self.res[i]) {
            *slot = Some(pass);
        }
    }

    /// Take a pass, run `f` over it, put it back. Every pass command goes
    /// through here: taking the payload out drops the borrow on the table, so
    /// `f` can freely look up pipelines, bind groups and buffers.
    fn on_pass(&mut self, handle: i64, f: impl FnOnce(&Self, &mut Pass)) {
        let Some(mut pass) = self.take_pass(handle) else { return };
        f(self, &mut pass);
        self.put_pass(handle, pass);
    }

    // ── Lazily created resources ──────────────────────────────────────────

    /// Create the offscreen colour target if it does not exist yet.
    /// Build the depth target at `w`x`h` if absent or the size changed.
    fn ensure_depth(&mut self, w: u32, h: u32) {
        let size = (w.max(1), h.max(1));
        if self.depth_view.is_some() && self.depth_size == size {
            return;
        }
        let tex = self.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("snaidhm depth"),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.depth_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
        self.depth_size = size;
    }

    fn ensure_target(&mut self) {
        if self.target_view.is_some() {
            return;
        }
        let texture = self.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("snaidhm offscreen target"),
            size: wgpu::Extent3d {
                width: DEFAULT_WIDTH,
                height: DEFAULT_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TARGET_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.target_view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.target = Some(texture);
    }

    fn ensure_fallbacks(&mut self) {
        if self.fallback_texture.is_none() {
            let texture = self.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("snaidhm fallback texture"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue().write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[0u8, 0, 0, 0],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            self.fallback_texture = Some(texture);
        }
        if self.fallback_sampler.is_none() {
            self.fallback_sampler = Some(self.device().create_sampler(&wgpu::SamplerDescriptor {
                label: Some("snaidhm fallback sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }));
        }
    }

    /// The colour format a pipeline should target. Mirrors the JS host, which
    /// ignores the format argument and always uses `_format`; here the handle
    /// is honoured when it resolves, and `TARGET_FORMAT` is the fallback.
    fn resolve_format(&self, handle: i64) -> wgpu::TextureFormat {
        match self.get(handle) {
            Some(Res::Format(f)) => *f,
            _ => TARGET_FORMAT,
        }
    }
}

/// Alpha blending shared by the text and image pipelines — the same
/// premultiply-friendly setup the JS host spells out inline.
const ALPHA_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

// ══════════════════════════════════════════════════════════════════════════
// Extern entry points — one `pub fn` per `@extern(rust, "crate::gpu", ...)`.
// Almide types map as: Int → i64, Float → f64, Unit → ().
// ══════════════════════════════════════════════════════════════════════════

// ── Device / context ──────────────────────────────────────────────────────

/// Headless stand-in for configuring a canvas: creates the offscreen colour
/// target every render pass draws into, and returns a context handle.
pub fn configure_canvas(_device: i64, _format: i64) -> i64 {
    with(0, |s| {
        s.ensure_target();
        s.alloc(Res::Context)
    })
}

pub fn get_preferred_format() -> i64 {
    with(0, |s| {
        if s.format_handle == 0 {
            s.format_handle = s.alloc(Res::Format(TARGET_FORMAT));
        }
        s.format_handle
    })
}

// ── Shader ────────────────────────────────────────────────────────────────

/// `code_ptr` is a shader *index*, not a pointer (see `SHADERS`). Out-of-range
/// indices fall back to index 0, matching `SHADERS[idx] || SHADERS[0]`.
pub fn create_shader(_device: i64, code_ptr: i64, _code_len: i64) -> i64 {
    with(0, |s| {
        let idx = usize::try_from(code_ptr).unwrap_or(0);
        let source = SHADERS.get(idx).copied().unwrap_or(SHADERS[0]);
        let module = s.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("snaidhm shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        s.alloc(Res::Shader(module))
    })
}

// ── Buffer ────────────────────────────────────────────────────────────────

pub fn create_buffer(_device: i64, size: i64, usage: i64) -> i64 {
    with(0, |s| {
        let buffer = s.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("snaidhm buffer"),
            size: size.max(0) as u64,
            usage: wgpu::BufferUsages::from_bits_truncate(usage as u32),
            mapped_at_creation: false,
        });
        s.alloc(Res::Buffer(buffer))
    })
}

/// No-op natively.
///
// TODO: `data_ptr`/`data_len` address wasm linear memory, which does not exist
// in a native build — there is no host memory region to read the bytes from.
// snaidhm never calls this (it uploads through `begin_data`/`push_*`/
// `flush_to_buffer` instead, which is pointer-free by design); if a native
// caller ever needs raw uploads, add a slice-taking extern rather than
// pretending an integer is a pointer here.
pub fn write_buffer(_device: i64, _buffer: i64, _data_ptr: i64, _data_len: i64) {}

pub fn write_f32_at(_device: i64, buffer: i64, byte_offset: i64, value: f64) {
    with((), |s| {
        let Some(buf) = s.buffer(buffer) else { return };
        s.queue().write_buffer(buf, byte_offset.max(0) as u64, &(value as f32).to_ne_bytes());
    })
}

pub fn write_u32_at(_device: i64, buffer: i64, byte_offset: i64, value: i64) {
    with((), |s| {
        let Some(buf) = s.buffer(buffer) else { return };
        s.queue().write_buffer(buf, byte_offset.max(0) as u64, &(value as u32).to_ne_bytes());
    })
}

// ── Buffer data streaming ─────────────────────────────────────────────────

pub fn begin_data() {
    with((), |s| s.data.clear())
}

pub fn push_f32(value: f64) {
    with((), |s| s.data.push((value as f32).to_bits()))
}

pub fn push_u32(value: i64) {
    with((), |s| s.data.push(value as u32))
}

pub fn flush_to_buffer(_device: i64, buffer: i64) {
    with((), |s| {
        if let Some(buf) = s.buffer(buffer) {
            s.queue().write_buffer(buf, 0, bytemuck::cast_slice(&s.data));
        }
        s.data.clear();
    })
}

// ── Pipelines ─────────────────────────────────────────────────────────────

/// Fullscreen quad pipeline: `vs_fullscreen` / `fs_fullscreen`, no vertex
/// buffers, no blending.
pub fn create_render_pipeline(
    _device: i64,
    shader: i64,
    _vs_ptr: i64,
    _vs_len: i64,
    _fs_ptr: i64,
    _fs_len: i64,
    format: i64,
) -> i64 {
    with(0, |s| {
        let target_format = s.resolve_format(format);
        let Some(Res::Shader(module)) = s.get(shader) else { return 0 };
        let pipeline = s.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("snaidhm quad pipeline"),
            layout: None, // == WebGPU's `layout: "auto"`
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_fullscreen"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        s.alloc(Res::RenderPipeline(pipeline))
    })
}

/// SDF text pipeline. Vertex layout: pos(f32x2) + uv(f32x2) + color(f32x4),
/// stride 32.
pub fn create_text_pipeline(_device: i64, shader: i64, format: i64) -> i64 {
    let attrs = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4];
    create_vertex_pipeline(shader, format, "snaidhm text pipeline", 32, &attrs)
}

/// Textured quad pipeline. Vertex layout: pos(f32x2) + uv(f32x2), stride 16.
pub fn create_image_pipeline(_device: i64, shader: i64, format: i64) -> i64 {
    let attrs = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2];
    create_vertex_pipeline(shader, format, "snaidhm image pipeline", 16, &attrs)
}

/// Shared body of the two `vs_main`/`fs_main` alpha-blended vertex pipelines.
fn create_vertex_pipeline(
    shader: i64,
    format: i64,
    label: &str,
    stride: u64,
    attributes: &[wgpu::VertexAttribute],
) -> i64 {
    with(0, |s| {
        let target_format = s.resolve_format(format);
        let Some(Res::Shader(module)) = s.get(shader) else { return 0 };
        let pipeline = s.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: None,
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: stride,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(ALPHA_BLEND),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        s.alloc(Res::RenderPipeline(pipeline))
    })
}

/// `entry_id` is ignored; the compute entry point is always `fine`, as in the
/// JS host.
pub fn create_compute_pipeline(_device: i64, shader: i64, _entry_id: i64) -> i64 {
    with(0, |s| {
        let Some(Res::Shader(module)) = s.get(shader) else { return 0 };
        let pipeline = s.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("snaidhm compute pipeline"),
            layout: None,
            module,
            entry_point: Some("fine"),
            compilation_options: Default::default(),
            cache: None,
        });
        s.alloc(Res::ComputePipeline(pipeline))
    })
}

// ── Bind groups ───────────────────────────────────────────────────────────

pub fn begin_bindings() {
    with((), |s| s.pending_bindings.clear())
}

pub fn add_buffer_binding(buffer: i64) {
    with((), |s| s.pending_bindings.push(Binding::Buffer(buffer)))
}

pub fn add_texture_binding(texture: i64) {
    with((), |s| s.pending_bindings.push(Binding::Texture(texture)))
}

pub fn add_sampler_binding(sampler: i64) {
    with((), |s| s.pending_bindings.push(Binding::Sampler(sampler)))
}

/// Build a bind group for `group_idx` of `pipeline` from the bindings pushed
/// since `begin_bindings`, in push order (binding N == the Nth push). Drains
/// the pending list, like the JS host.
pub fn create_bound_group(_device: i64, pipeline: i64, group_idx: i64) -> i64 {
    with(0, |s| {
        s.ensure_fallbacks();
        let layout = match s.get(pipeline) {
            Some(Res::ComputePipeline(p)) => p.get_bind_group_layout(group_idx.max(0) as u32),
            Some(Res::RenderPipeline(p)) => p.get_bind_group_layout(group_idx.max(0) as u32),
            _ => {
                s.pending_bindings.clear();
                return 0;
            }
        };

        // Texture views must outlive the entries that borrow them, so they are
        // materialised into their own vec first.
        let views: Vec<Option<wgpu::TextureView>> = s
            .pending_bindings
            .iter()
            .map(|b| match b {
                Binding::Texture(h) => {
                    let texture = match s.get(*h) {
                        Some(Res::Texture(t)) => t,
                        _ => s.fallback_texture.as_ref().expect("ensured above"),
                    };
                    Some(texture.create_view(&wgpu::TextureViewDescriptor::default()))
                }
                _ => None,
            })
            .collect();

        let entries: Vec<wgpu::BindGroupEntry> = s
            .pending_bindings
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                let resource = match b {
                    Binding::Buffer(h) => s.buffer(*h)?.as_entire_binding(),
                    Binding::Texture(_) => {
                        wgpu::BindingResource::TextureView(views[i].as_ref()?)
                    }
                    Binding::Sampler(h) => {
                        let sampler = match s.get(*h) {
                            Some(Res::Sampler(sm)) => sm,
                            _ => s.fallback_sampler.as_ref().expect("ensured above"),
                        };
                        wgpu::BindingResource::Sampler(sampler)
                    }
                };
                Some(wgpu::BindGroupEntry { binding: i as u32, resource })
            })
            .collect();

        let bind_group = s.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("snaidhm bind group"),
            layout: &layout,
            entries: &entries,
        });
        drop(entries);
        drop(views);
        s.pending_bindings.clear();
        s.alloc(Res::BindGroup(bind_group))
    })
}

// ── Command encoding ──────────────────────────────────────────────────────

pub fn begin_encoder(_device: i64) -> i64 {
    with(0, |s| {
        let encoder = s
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("snaidhm encoder") });
        s.alloc(Res::Encoder(Some(encoder)))
    })
}

pub fn begin_compute_pass(encoder: i64) -> i64 {
    with(0, |s| {
        let Some(mut enc) = s.take_encoder(encoder) else { return 0 };
        let pass = enc
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("snaidhm compute pass"),
                timestamp_writes: None,
            })
            .forget_lifetime();
        s.put_encoder(encoder, enc);
        s.alloc(Res::Pass(Some(Pass::Compute(pass))))
    })
}

/// Renders into the offscreen target, clearing it to `(r, g, b, a)`.
///
/// The JS host uses `_context.getCurrentTexture()`; headless, the target from
/// `configure_canvas` stands in for the swapchain image. Calling this without
/// `configure_canvas` first creates the target on demand.
/// Create (or resize) the depth target the 3D pass renders against.
pub fn set_depth_size(_device: i64, w: i64, h: i64) {
    with((), |s| s.ensure_depth(w.max(1) as u32, h.max(1) as u32))
}

/// Pipeline for the standard mesh vertex layout: pos(3) + normal(3) + uv(2),
/// 32-byte stride, depth `less`, back faces culled, CCW front (glTF's winding).
pub fn create_mesh_pipeline(
    _device: i64,
    shader: i64,
    _format: i64,
    cull: i64,
    blend: i64,
    depth_write: i64,
) -> i64 {
    let cull_mode = match cull {
        0 => None,
        2 => Some(wgpu::Face::Front),
        _ => Some(wgpu::Face::Back),
    };
    with(0, |s| {
        let Some(Res::Shader(module)) = s.get(shader) else { return 0 };
        let pipeline = s.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("snaidhm mesh pipeline"),
            layout: None, // == WebGPU's `layout: "auto"`
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 32,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 2 },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TARGET_FORMAT,
                    blend: if blend == 1 { Some(wgpu::BlendState::ALPHA_BLENDING) } else { None },
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: depth_write != 0,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        s.alloc(Res::RenderPipeline(pipeline))
    })
}

/// `begin_render_pass` with the depth attachment bound. `load` chooses whether
/// the colour target is cleared (0) or preserved (1). Depth always clears.
pub fn begin_render_pass_3d(encoder: i64, r: f64, g: f64, b: f64, a: f64, load: i64) -> i64 {
    with(0, |s| {
        s.ensure_target();
        s.ensure_depth(DEFAULT_WIDTH, DEFAULT_HEIGHT);
        let Some(mut enc) = s.take_encoder(encoder) else { return 0 };
        let pass = {
            let view = s.target_view.as_ref().expect("ensured above");
            let depth = s.depth_view.as_ref().expect("ensured above");
            enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("snaidhm 3D pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if load == 1 {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a })
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            })
            .forget_lifetime()
        };
        s.put_encoder(encoder, enc);
        s.alloc(Res::Pass(Some(Pass::Render(pass))))
    })
}

/// An empty RGBA8 texture at its final size.
pub fn create_texture(_device: i64, w: i64, h: i64) -> i64 {
    with(0, |s| {
        let tex = s.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("snaidhm texture"),
            size: wgpu::Extent3d {
                width: w.max(1) as u32,
                height: h.max(1) as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        s.alloc(Res::Texture(tex))
    })
}

/// No-op natively, like `write_buffer`: `ptr` addresses wasm linear memory,
/// which does not exist here, and decoding PNG/JPEG would mean pulling in an
/// image codec this crate deliberately does not carry. The texture stays at its
/// cleared contents rather than showing garbage. A native caller that needs
/// real pixels uploads them through its own path.
pub fn upload_encoded_image(_device: i64, _texture: i64, _ptr: i64, _len: i64) {}

pub fn create_sampler(_device: i64, filter: i64, wrap: i64) -> i64 {
    with(0, |s| {
        let f = if filter == 0 { wgpu::FilterMode::Nearest } else { wgpu::FilterMode::Linear };
        let w = if wrap == 1 { wgpu::AddressMode::Repeat } else { wgpu::AddressMode::ClampToEdge };
        let sampler = s.device().create_sampler(&wgpu::SamplerDescriptor {
            label: Some("snaidhm sampler"),
            mag_filter: f,
            min_filter: f,
            address_mode_u: w,
            address_mode_v: w,
            ..Default::default()
        });
        s.alloc(Res::Sampler(sampler))
    })
}

/// Draw a range of the bound index buffer — one draw per material needs this.
pub fn draw_indexed_from(pass: i64, first_index: i64, index_count: i64) {
    with((), |s| {
        s.on_pass(pass, |_, p| {
            let Pass::Render(rp) = p else { return };
            let first = first_index.max(0) as u32;
            rp.draw_indexed(first..first + index_count.max(0) as u32, 0, 0..1);
        })
    })
}

/// Bind a u16 index buffer.
pub fn set_index_buffer_u16(pass: i64, buffer: i64) {
    with((), |s| {
        s.on_pass(pass, |s, p| {
            let (Pass::Render(rp), Some(buf)) = (p, s.buffer(buffer)) else { return };
            rp.set_index_buffer(buf.slice(..), wgpu::IndexFormat::Uint16);
        })
    })
}

/// No-op natively, like `write_buffer`: `src_ptr` addresses wasm linear memory,
/// which does not exist here. A native caller uploads through its own path.
pub fn write_buffer_at(_device: i64, _buffer: i64, _dst_offset: i64, _src_ptr: i64, _len: i64) {}

pub fn begin_render_pass(encoder: i64, r: f64, g: f64, b: f64, a: f64) -> i64 {
    with(0, |s| {
        s.ensure_target();
        let Some(mut enc) = s.take_encoder(encoder) else { return 0 };
        let pass = {
            let view = s.target_view.as_ref().expect("ensured above");
            enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("snaidhm render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            })
            .forget_lifetime()
        };
        s.put_encoder(encoder, enc);
        s.alloc(Res::Pass(Some(Pass::Render(pass))))
    })
}

/// Ends the pass (dropping a wgpu pass is what ends it) and frees its handle.
pub fn end_pass(pass: i64) {
    with((), |s| {
        drop(s.take_pass(pass));
        s.release(pass);
    })
}

pub fn finish_and_submit(_device: i64, encoder: i64) {
    with((), |s| {
        let Some(enc) = s.take_encoder(encoder) else { return };
        s.queue().submit(std::iter::once(enc.finish()));
        s.release(encoder);
    })
}

// ── Pass commands ─────────────────────────────────────────────────────────

pub fn set_pipeline(pass: i64, pipeline: i64) {
    with((), |s| {
        s.on_pass(pass, |s, p| match (p, s.get(pipeline)) {
            (Pass::Render(rp), Some(Res::RenderPipeline(pl))) => rp.set_pipeline(pl),
            (Pass::Compute(cp), Some(Res::ComputePipeline(pl))) => cp.set_pipeline(pl),
            _ => {}
        })
    })
}

pub fn set_bind_group(pass: i64, index: i64, bind_group: i64) {
    with((), |s| {
        s.on_pass(pass, |s, p| {
            let Some(Res::BindGroup(bg)) = s.get(bind_group) else { return };
            let index = index.max(0) as u32;
            match p {
                Pass::Render(rp) => rp.set_bind_group(index, bg, &[]),
                Pass::Compute(cp) => cp.set_bind_group(index, bg, &[]),
            }
        })
    })
}

pub fn set_vertex_buffer(pass: i64, slot: i64, buffer: i64) {
    with((), |s| {
        s.on_pass(pass, |s, p| {
            let (Pass::Render(rp), Some(buf)) = (p, s.buffer(buffer)) else { return };
            rp.set_vertex_buffer(slot.max(0) as u32, buf.slice(..));
        })
    })
}

/// Index buffers are always `uint32`, as in the JS host.
pub fn set_index_buffer(pass: i64, buffer: i64) {
    with((), |s| {
        s.on_pass(pass, |s, p| {
            let (Pass::Render(rp), Some(buf)) = (p, s.buffer(buffer)) else { return };
            rp.set_index_buffer(buf.slice(..), wgpu::IndexFormat::Uint32);
        })
    })
}

pub fn draw(pass: i64, vertex_count: i64) {
    with((), |s| {
        s.on_pass(pass, |_, p| {
            let Pass::Render(rp) = p else { return };
            rp.draw(0..vertex_count.max(0) as u32, 0..1);
        })
    })
}

pub fn draw_indexed(pass: i64, index_count: i64) {
    with((), |s| {
        s.on_pass(pass, |_, p| {
            let Pass::Render(rp) = p else { return };
            rp.draw_indexed(0..index_count.max(0) as u32, 0, 0..1);
        })
    })
}

pub fn dispatch_workgroups(pass: i64, x: i64, y: i64, z: i64) {
    with((), |s| {
        s.on_pass(pass, |_, p| {
            let Pass::Compute(cp) = p else { return };
            cp.dispatch_workgroups(x.max(0) as u32, y.max(0) as u32, z.max(0) as u32);
        })
    })
}

// ── Debug ─────────────────────────────────────────────────────────────────

pub fn log_int(value: i64) {
    eprintln!("[gpu] {value}");
}

/// No-op natively.
///
// TODO: `ptr`/`len` address wasm linear memory. A native build has no such
// memory to decode a string out of, and inventing one would fabricate output
// that does not correspond to what the caller passed. Logging a string
// natively needs a String-typed extern (Almide `String` lowers to `&str` in the
// generated wrapper), not a pointer pair.
pub fn log_str(_ptr: i64, _len: i64) {}
