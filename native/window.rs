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
//! ## IME
//!
//! With `set_ime(true)` the platform's input method composes text: what it
//! is composing arrives as `COMPOSE` events (the whole composition each time,
//! empty when it ends) and what it commits as `TEXT` events, one per
//! character — the same events typing produces, so a field needs nothing
//! else to take committed text. `set_ime_area` tells the platform where the
//! caret is, for its candidate window. In the browser a DOM input element
//! does all of this itself.
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
use winit::dpi::{LogicalPosition, LogicalSize};
use winit::event::{ElementState, Ime, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Window, WindowAttributes, WindowId};

/// How long `open` waits for the platform to deliver the event that allows a
/// window to be created. Normally one pump; the bound only turns a platform
/// that never delivers it into an error instead of a hang.
const OPEN_DEADLINE: Duration = Duration::from_secs(5);

/// How long a window opened hidden waits for its first frame before it is
/// shown anyway.
const SHOW_DEADLINE: Duration = Duration::from_millis(500);

/// Event kinds, as `next_event` returns them (0: none left).
const MOUSE_MOVE: i64 = 1;
const MOUSE_DOWN: i64 = 2;
const MOUSE_UP: i64 = 3;
const WHEEL: i64 = 4;
const TEXT: i64 = 5;
const KEY: i64 = 6;
const COMPOSE: i64 = 7;

/// Modifier bits, as `event_mods` returns them.
const MOD_SHIFT: i64 = 1;
const MOD_CTRL: i64 = 2;
const MOD_ALT: i64 = 4;
const MOD_SUPER: i64 = 8;

/// Pixels one wheel notch scrolls, for devices that report lines.
const LINE_PX: f64 = 40.0;

/// Events kept for a program that is not reading them.
const QUEUE_LIMIT: usize = 1024;

/// One queued input. `code` is the button (0 left, 1 right, 2 middle), the
/// typed code point, or the DOM key code, by kind. A composition carries its
/// text, and in `code` .. `end` the part the input method marks (the clause
/// being converted, or an empty range at its caret), in characters; -1 when
/// it shows no caret. `mods` are the modifiers held when it happened.
#[derive(Clone, Default)]
struct Input {
    kind: i64,
    mods: i64,
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
    code: i64,
    end: i64,
    text: String,
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
        NamedKey::PageUp => 33,
        NamedKey::PageDown => 34,
        NamedKey::End => 35,
        NamedKey::Home => 36,
        NamedKey::Insert => 45,
        NamedKey::F1 => 112,
        NamedKey::F2 => 113,
        NamedKey::F3 => 114,
        NamedKey::F4 => 115,
        NamedKey::F5 => 116,
        NamedKey::F6 => 117,
        NamedKey::F7 => 118,
        NamedKey::F8 => 119,
        NamedKey::F9 => 120,
        NamedKey::F10 => 121,
        NamedKey::F11 => 122,
        NamedKey::F12 => 123,
        _ => return None,
    })
}

struct Host {
    event_loop: EventLoop<()>,
    app: App,
    /// When the window opened, until it is shown: it opens hidden and shows
    /// with its first frame, rather than empty while the program gets its
    /// GPU work ready.
    hidden_since: Option<Instant>,
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
    /// Modifiers held now, MOD_* bits.
    mods: i64,
}

fn mod_bits(m: ModifiersState) -> i64 {
    let mut bits = 0;
    if m.shift_key() { bits |= MOD_SHIFT; }
    if m.control_key() { bits |= MOD_CTRL; }
    if m.alt_key() { bits |= MOD_ALT; }
    if m.super_key() { bits |= MOD_SUPER; }
    bits
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

    fn typed(&mut self, text: &str) {
        for ch in text.chars().filter(|c| !c.is_control()) {
            let input = self.at_cursor(TEXT, i64::from(u32::from(ch)));
            self.push(input);
        }
    }

    fn compose(&mut self, text: String, marked: Option<(usize, usize)>) {
        // winit gives byte offsets; a UI counts characters.
        let chars = |byte: usize| text.get(..byte).map_or(0, |s| s.chars().count()) as i64;
        let (code, end) = marked.map_or((-1, -1), |(a, b)| (chars(a), chars(b)));
        let input = Input { end, text, ..self.at_cursor(COMPOSE, code) };
        self.push(input);
    }

    fn at_cursor(&self, kind: i64, code: i64) -> Input {
        Input { kind, mods: self.mods, x: self.cursor.0, y: self.cursor.1, code, ..Input::default() }
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
                // With Ctrl, Alt or Super held the text is a control char,
                // an Option-composed letter or nothing; a shortcut wants the
                // key itself, with `event_mods` saying what was held.
                if self.mods & (MOD_CTRL | MOD_ALT | MOD_SUPER) != 0 {
                    if let Some(text) = event.key_without_modifiers().to_text() {
                        let text = text.to_string();
                        self.typed(&text);
                    }
                    return;
                }
                // Space arrives as `NamedKey::Space`, and on macOS without
                // `text`; the key's own text covers it.
                let text = event.text.as_deref().or_else(|| event.logical_key.to_text());
                if let Some(text) = text {
                    let text = text.to_string();
                    self.typed(&text);
                }
            }
            // Keys the input method takes arrive as these, not as keys.
            WindowEvent::ModifiersChanged(m) => self.mods = mod_bits(m.state()),
            WindowEvent::Ime(Ime::Preedit(text, marked)) => self.compose(text, marked),
            WindowEvent::Ime(Ime::Commit(text)) => self.typed(&text),
            WindowEvent::Ime(Ime::Disabled) => self.compose(String::new(), None),
            WindowEvent::Ime(Ime::Enabled) => {}
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
        let presented = crate::gpu::present_frame();
        if let Some(since) = self.hidden_since {
            if presented || since.elapsed() >= SHOW_DEADLINE {
                if let Some(w) = &self.app.window {
                    w.set_visible(true);
                }
                self.hidden_since = None;
            }
        }
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
            .with_inner_size(LogicalSize::new(width.max(1) as f64, height.max(1) as f64))
            .with_visible(false);
        let _ = PROXY.set(event_loop.create_proxy());
        let mut host = Host { event_loop, app: App { pending: Some(attrs), ..App::default() }, hidden_since: Some(Instant::now()) };
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

/// End the frame and sleep until an event arrives, one of `fds` has something
/// to read, or `timeout_ms` passes (negative: no limit). `false` once the
/// window has been asked to close.
///
/// For a program that also waits on files — a terminal on its PTYs: it
/// sleeps through both at once, where polling each in turn would wake it
/// every few milliseconds for nothing.
pub fn wait_fds(fds: &[i64], timeout_ms: i64) -> bool {
    #[cfg(unix)]
    watch::arm(fds.iter().map(|&fd| fd as i32).collect());
    let timeout = if timeout_ms < 0 { None } else { Some(Duration::from_millis(timeout_ms as u64)) };
    let alive = with_host(false, |host| host.pump(timeout));
    #[cfg(unix)]
    watch::disarm();
    alive
}

/// Wakes the event loop from other threads.
static PROXY: std::sync::OnceLock<EventLoopProxy<()>> = std::sync::OnceLock::new();

/// The thread behind `wait_fds`: while armed it polls the fds, and when one is
/// readable it wakes the event loop and disarms, so a file nobody has read
/// yet can't keep it spinning. A pipe interrupts its poll when the set
/// changes.
#[cfg(unix)]
mod watch {
    use std::sync::{Condvar, Mutex, OnceLock};

    struct State { fds: Vec<i32>, armed: bool }

    struct Watch { state: Mutex<State>, changed: Condvar, pipe: (i32, i32) }

    static WATCH: OnceLock<&'static Watch> = OnceLock::new();

    fn get() -> &'static Watch {
        WATCH.get_or_init(|| {
            let mut p = [0i32; 2];
            unsafe { libc::pipe(p.as_mut_ptr()) };
            for fd in p {
                unsafe { libc::fcntl(fd, libc::F_SETFL, libc::fcntl(fd, libc::F_GETFL) | libc::O_NONBLOCK) };
            }
            let w: &'static Watch = Box::leak(Box::new(Watch {
                state: Mutex::new(State { fds: Vec::new(), armed: false }),
                changed: Condvar::new(),
                pipe: (p[0], p[1]),
            }));
            std::thread::spawn(move || run(w));
            w
        })
    }

    fn run(w: &'static Watch) {
        loop {
            let fds = {
                let mut st = w.state.lock().unwrap();
                while !st.armed {
                    st = w.changed.wait(st).unwrap();
                }
                st.fds.clone()
            };
            let mut pfds: Vec<libc::pollfd> = fds
                .iter()
                .map(|&fd| libc::pollfd { fd, events: libc::POLLIN, revents: 0 })
                .collect();
            pfds.push(libc::pollfd { fd: w.pipe.0, events: libc::POLLIN, revents: 0 });
            unsafe { libc::poll(pfds.as_mut_ptr(), pfds.len() as libc::nfds_t, -1) };
            let mut buf = [0u8; 64];
            while unsafe { libc::read(w.pipe.0, buf.as_mut_ptr() as *mut _, buf.len()) } > 0 {}
            let ready = pfds[..fds.len()].iter().any(|p| p.revents != 0);
            if ready {
                w.state.lock().unwrap().armed = false;
                if let Some(proxy) = super::PROXY.get() {
                    let _ = proxy.send_event(());
                }
            }
        }
    }

    pub fn arm(fds: Vec<i32>) {
        let w = get();
        let mut st = w.state.lock().unwrap();
        st.fds = fds;
        st.armed = true;
        w.changed.notify_one();
        drop(st);
        unsafe { libc::write(w.pipe.1, [1u8].as_ptr() as *const _, 1) };
    }

    pub fn disarm() {
        let w = get();
        w.state.lock().unwrap().armed = false;
        unsafe { libc::write(w.pipe.1, [1u8].as_ptr() as *const _, 1) };
    }
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
        let kind = input.kind;
        host.app.current = input;
        kind
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

/// End of the marked part of the current composition, characters.
pub fn event_end() -> i64 {
    with_host(0, |host| host.app.current.end)
}

/// Modifiers held when the current input happened: 1 Shift, 2 Ctrl, 4 Alt
/// (Option), 8 Super (Command).
pub fn event_mods() -> i64 {
    with_host(0, |host| host.app.current.mods)
}

/// Text of the current composition.
pub fn event_text() -> String {
    with_host(String::new(), |host| host.app.current.text.clone())
}

/// Let the platform's input method compose text in this window (`true`), or
/// take keys as they are (`false`, the default).
pub fn set_ime(allowed: bool) {
    with_host((), |host| {
        if let Some(w) = &host.app.window {
            w.set_ime_allowed(allowed);
        }
    })
}

/// Where the caret is — `x`, `y`, `w`, `h` in logical pixels — so the input
/// method can put its candidate window beside it rather than over it.
pub fn set_ime_area(x: f64, y: f64, w: f64, h: f64) {
    with_host((), |host| {
        if let Some(win) = &host.app.window {
            win.set_ime_cursor_area(LogicalPosition::new(x, y), LogicalSize::new(w.max(1.0), h.max(1.0)));
        }
    })
}
