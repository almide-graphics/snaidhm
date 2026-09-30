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
//! ## Input
//!
//! Window events the program reacts to are queued as `Input`s, in logical
//! pixels (CSS pixels, the unit ceangal lays out in), and read one at a time
//! with `next_event` and its accessors. The kinds and key codes follow the
//! DOM, so a browser host and this one hand a UI the same events.
//!
//! ## State
//!
//! A winit event loop is `!Send`, so the window lives in a thread-local, not in
//! a global like the GPU runtime.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{Key, NamedKey};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Window, WindowAttributes, WindowId};

/// How long `open` waits for the platform to deliver the event that allows a
/// window to be created. Normally one pump; the bound only turns a platform
/// that never delivers it into an error instead of a hang.
const OPEN_DEADLINE: Duration = Duration::from_secs(5);

/// Event kinds, as `next_event` returns them (0: none left).
const MOUSE_MOVE: i64 = 1;
const MOUSE_DOWN: i64 = 2;
const MOUSE_UP: i64 = 3;
const WHEEL: i64 = 4;
const TEXT: i64 = 5;
const KEY: i64 = 6;

/// Pixels one wheel notch scrolls, for devices that report lines.
const LINE_PX: f64 = 40.0;

/// Events kept for a program that is not reading them.
const QUEUE_LIMIT: usize = 1024;

/// One queued input. `code` is the button (0 left, 1 right, 2 middle), the
/// typed code point, or the DOM key code, by kind.
#[derive(Clone, Copy, Default)]
struct Input {
    kind: i64,
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
    code: i64,
}

/// The DOM `keyCode` of the keys a UI handles as keys rather than text.
fn key_code(key: &NamedKey) -> Option<i64> {
    Some(match key {
        NamedKey::Backspace => 8,
        NamedKey::Tab => 9,
        NamedKey::Enter => 13,
        NamedKey::Escape => 27,
        NamedKey::ArrowLeft => 37,
        NamedKey::ArrowUp => 38,
        NamedKey::ArrowRight => 39,
        NamedKey::ArrowDown => 40,
        NamedKey::Delete => 46,
        _ => return None,
    })
}

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
    /// Last cursor position, logical pixels.
    cursor: (f64, f64),
    events: VecDeque<Input>,
    /// The event `next_event` last returned, read by the accessors.
    current: Input,
}

impl App {
    fn scale(&self) -> f64 {
        self.window.as_ref().map_or(1.0, |w| w.scale_factor())
    }

    fn push(&mut self, input: Input) {
        if self.events.len() >= QUEUE_LIMIT {
            self.events.pop_front();
        }
        self.events.push_back(input);
    }

    fn at_cursor(&self, kind: i64, code: i64) -> Input {
        Input { kind, x: self.cursor.0, y: self.cursor.1, code, ..Input::default() }
    }
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
            WindowEvent::CursorMoved { position, .. } => {
                let s = self.scale();
                self.cursor = (position.x / s, position.y / s);
                let input = self.at_cursor(MOUSE_MOVE, 0);
                self.push(input);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let code = match button {
                    MouseButton::Left => 0,
                    MouseButton::Right => 1,
                    MouseButton::Middle => 2,
                    _ => return,
                };
                let kind = if state == ElementState::Pressed { MOUSE_DOWN } else { MOUSE_UP };
                let input = self.at_cursor(kind, code);
                self.push(input);
            }
            // winit's positive delta moves the content right and down; the
            // DOM's `deltaY` is positive scrolling down, which moves it up.
            WindowEvent::MouseWheel { delta, .. } => {
                let (dx, dy) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (-f64::from(x) * LINE_PX, -f64::from(y) * LINE_PX),
                    MouseScrollDelta::PixelDelta(p) => {
                        let s = self.scale();
                        (-p.x / s, -p.y / s)
                    }
                };
                let input = Input { dx, dy, ..self.at_cursor(WHEEL, 0) };
                self.push(input);
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                // A key a UI handles as a key is only that: Enter also carries
                // "\r" as text, which would type a line break into a field.
                if let Key::Named(named) = &event.logical_key {
                    if let Some(code) = key_code(named) {
                        let input = self.at_cursor(KEY, code);
                        self.push(input);
                        return;
                    }
                }
                // Space arrives as `NamedKey::Space`, and on macOS without
                // `text`; the key's own text covers it.
                let text = event.text.as_deref().or_else(|| event.logical_key.to_text());
                if let Some(text) = text {
                    for ch in text.chars().filter(|c| !c.is_control()) {
                        let input = self.at_cursor(TEXT, i64::from(u32::from(ch)));
                        self.push(input);
                    }
                }
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

/// The kind of the next queued input, which the `event_*` accessors then
/// describe; 0 once the queue is empty. Inputs queue up across `pump` calls
/// until read.
pub fn next_event() -> i64 {
    with_host(0, |host| {
        let input = host.app.events.pop_front().unwrap_or_default();
        host.app.current = input;
        input.kind
    })
}

/// Pointer position of the current input, logical pixels.
pub fn event_x() -> f64 {
    with_host(0.0, |host| host.app.current.x)
}

pub fn event_y() -> f64 {
    with_host(0.0, |host| host.app.current.y)
}

/// Wheel distance of the current input, logical pixels, DOM signs.
pub fn event_dx() -> f64 {
    with_host(0.0, |host| host.app.current.dx)
}

pub fn event_dy() -> f64 {
    with_host(0.0, |host| host.app.current.dy)
}

/// Button, typed code point, or DOM key code of the current input.
pub fn event_code() -> i64 {
    with_host(0, |host| host.app.current.code)
}

/// Physical pixels per logical pixel (2.0 on a typical Retina display).
pub fn scale_factor() -> f64 {
    with_host(1.0, |host| host.app.window.as_ref().map_or(1.0, |w| w.scale_factor()))
}
