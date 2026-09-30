//! Native window for snaidhm: the on-screen counterpart of the browser canvas.
//!
//! Backs `src/native/window.almd` (`@extern(rust, "crate::window", ...)`).
//! There is no wasm side: in the browser the page owns the canvas and the
//! frame loop, and hands both to the module.
//!
//! ## Who runs the loop
//!
//! The Almide program does. Instead of handing control to winit's `run_app`,
//! each `pump` / `wait` call lets winit process what is pending and returns, so
//! a frame loop is an ordinary `while window.pump() { ... }` in Almide. winit
//! calls this "pumping" the event loop (`EventLoopExtPumpEvents`); it is
//! supported on macOS, Windows, Wayland and X11. The loop stays on the thread
//! that opened the window, which is the main thread — macOS requires that of
//! an event loop, and the generated `main` runs Almide's `main` on it.
//!
//! ## Frames
//!
//! Returning to the event loop is what ends a frame: `pump` and `wait` first
//! present the image the frame's render passes drew (`gpu::present_frame`),
//! the native counterpart of a `requestAnimationFrame` callback returning.
//!
//! ## State
//!
//! A winit event loop is `!Send`, so the window lives in a thread-local, not in
//! a global like the GPU runtime.

use std::cell::RefCell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Window, WindowAttributes, WindowId};

/// How long `open` waits for the platform to deliver the event that allows a
/// window to be created. Normally one pump; the bound only turns a platform
/// that never delivers it into an error instead of a hang.
const OPEN_DEADLINE: Duration = Duration::from_secs(5);

struct Host {
    event_loop: EventLoop<()>,
    app: App,
}

#[derive(Default)]
struct App {
    /// Attributes of the window to create at the next `resumed`. winit only
    /// lets a window be created from inside the event loop.
    pending: Option<WindowAttributes>,
    window: Option<Arc<Window>>,
    /// Set when window creation or surface attachment failed.
    failed: bool,
    close_requested: bool,
    /// Set by a resize, cleared by the `resized()` that reports it.
    resized: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let Some(attrs) = self.pending.take() else { return };
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("[snaidhm/window] cannot create the window: {e}");
                self.failed = true;
                return;
            }
        };
        let size = window.inner_size();
        if !crate::gpu::attach_surface(window.clone().into(), size.width, size.height) {
            self.failed = true;
            return;
        }
        self.window = Some(window);
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.close_requested = true,
            // A scale-factor change is followed by the `Resized` that carries
            // its new physical size, so this one arm covers both.
            WindowEvent::Resized(size) => {
                crate::gpu::resize_surface(size.width, size.height);
                self.resized = true;
            }
            _ => {}
        }
    }
}

thread_local! {
    static HOST: RefCell<Option<Host>> = const { RefCell::new(None) };
}

/// Run `f` against the open window, or return `default` when there is none.
fn with_host<R>(default: R, f: impl FnOnce(&mut Host) -> R) -> R {
    HOST.with(|h| match h.borrow_mut().as_mut() {
        Some(host) => f(host),
        None => default,
    })
}

impl Host {
    /// Process events, blocking for at most `timeout` (`None`: until one
    /// arrives). Returns whether the window is still wanted.
    fn pump(&mut self, timeout: Option<Duration>) -> bool {
        crate::gpu::present_frame();
        if let PumpStatus::Exit(_) = self.event_loop.pump_app_events(timeout, &mut self.app) {
            return false;
        }
        !self.app.close_requested
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Extern entry points — one `pub fn` per `@extern(rust, "crate::window", ...)`.
// ══════════════════════════════════════════════════════════════════════════

/// Open the window, `width` x `height` in logical pixels, and make it the
/// screen every later `gpu` render pass draws into. `false` when no window can
/// be opened (no display, a second window, a GPU that cannot present to it);
/// the reason is on stderr, and rendering stays offscreen.
pub fn open(title: &str, width: i64, height: i64) -> bool {
    HOST.with(|h| {
        let mut slot = h.borrow_mut();
        if slot.is_some() {
            eprintln!("[snaidhm/window] a window is already open; snaidhm drives one");
            return false;
        }
        let event_loop = match EventLoop::new() {
            Ok(el) => el,
            Err(e) => {
                eprintln!("[snaidhm/window] no event loop (is there a display?): {e}");
                return false;
            }
        };
        let attrs = Window::default_attributes()
            .with_title(title)
            .with_inner_size(LogicalSize::new(width.max(1) as f64, height.max(1) as f64));
        let mut host = Host { event_loop, app: App { pending: Some(attrs), ..App::default() } };
        let deadline = Instant::now() + OPEN_DEADLINE;
        while host.app.window.is_none() && !host.app.failed {
            if Instant::now() >= deadline {
                eprintln!("[snaidhm/window] the platform never let the window be created");
                return false;
            }
            host.event_loop.pump_app_events(Some(Duration::from_millis(10)), &mut host.app);
        }
        if host.app.failed {
            return false;
        }
        *slot = Some(host);
        true
    })
}

/// End the frame and process pending events without waiting. `false` once the
/// window has been asked to close. For animation: call once per frame.
pub fn pump() -> bool {
    with_host(false, |host| host.pump(Some(Duration::ZERO)))
}

/// End the frame and sleep until at least one event arrives. `false` once the
/// window has been asked to close. For a UI that only changes in response to
/// input: an idle window costs no CPU.
pub fn wait() -> bool {
    with_host(false, |host| host.pump(None))
}

/// `true` exactly once after each change of the window's size — the moment to
/// rebuild anything sized to it (a scene, the depth target).
pub fn resized() -> bool {
    with_host(false, |host| std::mem::take(&mut host.app.resized))
}

/// Width of the drawable area in physical pixels, the size render targets are.
pub fn width() -> i64 {
    with_host(0, |host| host.app.window.as_ref().map_or(0, |w| w.inner_size().width as i64))
}

/// Height of the drawable area in physical pixels.
pub fn height() -> i64 {
    with_host(0, |host| host.app.window.as_ref().map_or(0, |w| w.inner_size().height as i64))
}

/// Physical pixels per logical pixel (2.0 on a typical Retina display).
pub fn scale_factor() -> f64 {
    with_host(1.0, |host| host.app.window.as_ref().map_or(1.0, |w| w.scale_factor()))
}
