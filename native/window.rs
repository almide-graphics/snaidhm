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
//! ## Windows
//!
//! `open` opens window 1, `open_window` more. The per-window calls — size,
//! title, focus, IME, pointer, full screen — are about the window `select`
//! chose, and `gpu` render passes draw into its surface; each input names the
//! window it happened in. One event loop serves them all.
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
use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

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
const DROP: i64 = 8;
const THEME: i64 = 9;
const FOCUS: i64 = 10;
/// A window's close button, while more than one is open (with one, `pump`
/// and the waits return `false` instead).
const CLOSE: i64 = 11;

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
    /// The window it happened in.
    win: i64,
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
    /// Quitting asks the program first (see `layer::ask_before_quit`).
    quit_routed: bool,
}

/// One open window.
struct Win {
    /// The program's name for it: 1 for the window `open` opened, then 2, 3, ...
    id: i64,
    window: Arc<Window>,
    /// Set by a resize, cleared by the `resized()` that reports it.
    resized: bool,
    /// The view's size, physical pixels: what the surface is configured to
    /// and the cursor's position is measured in. Kept from the last
    /// `Resized`, which reports the view's frame — `inner_size` on macOS is
    /// the window's content rect, which in full screen still leaves out a
    /// title bar that is no longer there, so the view is taller than it says.
    size: (u32, u32),
    /// Last cursor position over it, logical pixels.
    cursor: (f64, f64),
    /// Whether it has the keyboard focus.
    focused: bool,
    /// When it opened, until it is shown: a window opens hidden and shows
    /// with its first frame, rather than empty while the program gets its
    /// GPU work ready.
    hidden_since: Option<Instant>,
}

#[derive(Default)]
struct App {
    /// Windows to create, with their ids. winit only lets a window be
    /// created from inside the event loop.
    pending: Vec<(i64, WindowAttributes)>,
    /// Ids of windows that could not be created (or given a surface).
    failed: Vec<i64>,
    wins: Vec<Win>,
    /// The id the next window gets.
    next_id: i64,
    /// The window the per-window calls (size, title, IME, ...) and render
    /// passes are for: see `select`.
    current: i64,
    /// Set when the only window's close button was pressed, or the app was
    /// asked to quit.
    close_requested: bool,
    events: VecDeque<Input>,
    /// The event `next_event` last returned, read by the accessors.
    input: Input,
    /// Modifiers held now, MOD_* bits.
    mods: i64,
    /// Which Option keys are down: 1 the left, 2 the right.
    alt_keys: i64,
    /// Which Option keys type as Alt (see `set_option_as_alt`): 0 none,
    /// 1 both, 2 the left, 3 the right.
    option_as_alt: i64,
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
    fn win(&self, id: i64) -> Option<&Win> {
        self.wins.iter().find(|w| w.id == id)
    }

    fn win_mut(&mut self, id: i64) -> Option<&mut Win> {
        self.wins.iter_mut().find(|w| w.id == id)
    }

    /// The window `select` chose.
    fn cur(&self) -> Option<&Win> {
        self.win(self.current)
    }

    fn cur_window(&self) -> Option<&Arc<Window>> {
        self.cur().map(|w| &w.window)
    }

    fn scale_of(&self, id: i64) -> f64 {
        self.win(id).map_or(1.0, |w| w.window.scale_factor())
    }

    fn push(&mut self, input: Input) {
        if self.events.len() >= QUEUE_LIMIT {
            self.events.pop_front();
        }
        self.events.push_back(input);
    }

    fn typed(&mut self, win: i64, text: &str) {
        // An Option that composes, not Alt: the text is what it typed.
        let mods = if self.mods & MOD_ALT != 0 && !self.option_is_alt() { self.mods & !MOD_ALT } else { self.mods };
        for ch in text.chars().filter(|c| !c.is_control()) {
            let input = Input { mods, ..self.at_cursor(win, TEXT, i64::from(u32::from(ch))) };
            self.push(input);
        }
    }

    /// Whether the Option held now types as Alt. Off macOS, Alt always does.
    fn option_is_alt(&self) -> bool {
        if !cfg!(target_os = "macos") {
            return true;
        }
        match self.option_as_alt {
            1 => true,
            2 => self.alt_keys & 1 != 0,
            3 => self.alt_keys & 2 != 0,
            _ => false,
        }
    }

    fn compose(&mut self, win: i64, text: String, marked: Option<(usize, usize)>) {
        // winit gives byte offsets; a UI counts characters.
        let chars = |byte: usize| text.get(..byte).map_or(0, |s| s.chars().count()) as i64;
        let (code, end) = marked.map_or((-1, -1), |(a, b)| (chars(a), chars(b)));
        let input = Input { end, text, ..self.at_cursor(win, COMPOSE, code) };
        self.push(input);
    }

    fn at_cursor(&self, win: i64, kind: i64, code: i64) -> Input {
        let (x, y) = self.win(win).map_or((0.0, 0.0), |w| w.cursor);
        Input { win, kind, mods: self.mods, x, y, code, ..Input::default() }
    }

    /// Create the windows waiting to be.
    fn create_pending(&mut self, event_loop: &ActiveEventLoop) {
        for (id, attrs) in std::mem::take(&mut self.pending) {
            match create(event_loop, id, attrs) {
                Some(win) => self.wins.push(win),
                None => self.failed.push(id),
            }
        }
    }
}

/// Window `id`, with a surface the GPU draws into; none (the reason on
/// stderr) when it can't be had.
fn create(event_loop: &ActiveEventLoop, id: i64, attrs: WindowAttributes) -> Option<Win> {
    let window = match event_loop.create_window(attrs) {
        Ok(w) => Arc::new(w),
        Err(e) => {
            eprintln!("[snaidhm/window] cannot create the window: {e}");
            return None;
        }
    };
    let size = window.inner_size();
    if !crate::gpu::attach_surface(id, window.clone().into(), size.width, size.height) {
        return None;
    }
    #[cfg(target_os = "macos")]
    if let Some(view) = ns_view(&window) {
        layer::pin(view);
    }
    Some(Win {
        id,
        window,
        resized: false,
        size: (size.width, size.height),
        cursor: (0.0, 0.0),
        focused: false,
        hidden_since: Some(Instant::now()),
    })
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.create_pending(event_loop);
    }

    fn new_events(&mut self, event_loop: &ActiveEventLoop, _cause: winit::event::StartCause) {
        self.create_pending(event_loop);
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, wid: WindowId, event: WindowEvent) {
        let Some(id) = self.wins.iter().find(|w| w.window.id() == wid).map(|w| w.id) else { return };
        match event {
            WindowEvent::CloseRequested => {
                if self.wins.len() > 1 {
                    let input = self.at_cursor(id, CLOSE, 0);
                    self.push(input);
                } else {
                    self.close_requested = true;
                }
            }
            // A scale-factor change is followed by the `Resized` that carries
            // its new physical size, so this one arm covers both.
            WindowEvent::Resized(size) => {
                crate::gpu::resize_surface(id, size.width, size.height);
                if let Some(w) = self.win_mut(id) {
                    w.size = (size.width, size.height);
                    w.resized = true;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let s = self.scale_of(id);
                if let Some(w) = self.win_mut(id) {
                    w.cursor = (position.x / s, position.y / s);
                }
                let input = self.at_cursor(id, MOUSE_MOVE, 0);
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
                let input = self.at_cursor(id, kind, code);
                self.push(input);
            }
            // winit's positive delta moves the content right and down; the
            // DOM's `deltaY` is positive scrolling down, which moves it up.
            WindowEvent::MouseWheel { delta, .. } => {
                let (dx, dy) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (-f64::from(x) * LINE_PX, -f64::from(y) * LINE_PX),
                    MouseScrollDelta::PixelDelta(p) => {
                        let s = self.scale_of(id);
                        (-p.x / s, -p.y / s)
                    }
                };
                let input = Input { dx, dy, ..self.at_cursor(id, WHEEL, 0) };
                self.push(input);
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                // A key a UI handles as a key is only that: Enter also carries
                // "\r" as text, which would type a line break into a field.
                if let Key::Named(named) = &event.logical_key {
                    if let Some(code) = key_code(named) {
                        let input = self.at_cursor(id, KEY, code);
                        self.push(input);
                        return;
                    }
                }
                // With Ctrl, Super or an Option typing as Alt held, the text
                // is a control char, an Option-composed letter or nothing; a
                // shortcut wants the key itself, with `event_mods` saying
                // what was held. An Option not typing as Alt types what it
                // composes (Option+¥ is a backslash on a Japanese keyboard),
                // reported without Alt.
                if self.mods & (MOD_CTRL | MOD_SUPER) != 0 || (self.mods & MOD_ALT != 0 && self.option_is_alt()) {
                    if let Some(text) = event.key_without_modifiers().to_text() {
                        let text = text.to_string();
                        self.typed(id, &text);
                    }
                    return;
                }
                // Space arrives as `NamedKey::Space`, and on macOS without
                // `text`; the key's own text covers it.
                let text = event.text.as_deref().or_else(|| event.logical_key.to_text());
                if let Some(text) = text {
                    let text = text.to_string();
                    self.typed(id, &text);
                }
            }
            // Keys the input method takes arrive as these, not as keys.
            WindowEvent::ModifiersChanged(m) => {
                self.mods = mod_bits(m.state());
                use winit::keyboard::ModifiersKeyState::Pressed;
                self.alt_keys = (if m.lalt_state() == Pressed { 1 } else { 0 }) | (if m.ralt_state() == Pressed { 2 } else { 0 });
            }
            WindowEvent::Focused(on) => {
                if let Some(w) = self.win_mut(id) {
                    w.focused = on;
                }
                let input = self.at_cursor(id, FOCUS, if on { 1 } else { 0 });
                self.push(input);
            }
            WindowEvent::ThemeChanged(_) => {
                let input = self.at_cursor(id, THEME, 0);
                self.push(input);
            }
            WindowEvent::Ime(Ime::Preedit(text, marked)) => self.compose(id, text, marked),
            WindowEvent::Ime(Ime::Commit(text)) => self.typed(id, &text),
            // One event per file: a drop of several is several in a row.
            WindowEvent::DroppedFile(path) => {
                let input = Input { text: path.to_string_lossy().into_owned(), ..self.at_cursor(id, DROP, 0) };
                self.push(input);
            }
            WindowEvent::Ime(Ime::Disabled) => self.compose(id, String::new(), None),
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
        #[cfg(target_os = "macos")]
        if !self.quit_routed {
            self.quit_routed = layer::ask_before_quit();
        }
        let presented = crate::gpu::present_frame();
        for w in &mut self.app.wins {
            if let Some(since) = w.hidden_since {
                if presented.contains(&w.id) || since.elapsed() >= SHOW_DEADLINE {
                    w.window.set_visible(true);
                    w.hidden_since = None;
                }
            }
        }
        if let PumpStatus::Exit(_) = self.event_loop.pump_app_events(timeout, &mut self.app) {
            return false;
        }
        #[cfg(target_os = "macos")]
        if layer::QUIT_ASKED.swap(false, std::sync::atomic::Ordering::AcqRel) {
            self.app.close_requested = true;
        }
        // Once more without waiting, when the wait had one: on macOS winit
        // can hold an input that arrived while it woke for something else (a
        // PTY's output, the timeout) until the next event, and a key held down
        // showed up in pairs, one repeat late.
        if timeout != Some(Duration::ZERO) {
            if let PumpStatus::Exit(_) = self.event_loop.pump_app_events(Some(Duration::ZERO), &mut self.app) {
                return false;
            }
        }
        !self.app.close_requested
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Extern entry points — one `pub fn` per `@extern(rust, "crate::window", ...)`.
// ══════════════════════════════════════════════════════════════════════════

/// Open the window, `width` x `height` in logical pixels, and make it the
/// screen every later `gpu` render pass draws into; it is window 1 (see
/// `open_window` for more). `false` when no window can be opened (no
/// display, one already open, a GPU that cannot present to it); the reason is
/// on stderr, and rendering stays offscreen.
pub fn open(title: &str, width: i64, height: i64) -> bool {
    HOST.with(|h| {
        let mut slot = h.borrow_mut();
        if slot.is_some() {
            eprintln!("[snaidhm/window] a window is already open; open more with open_window");
            return false;
        }
        let event_loop = match EventLoop::new() {
            Ok(el) => el,
            Err(e) => {
                eprintln!("[snaidhm/window] no event loop (is there a display?): {e}");
                return false;
            }
        };
        let attrs = attributes(title, width, height);
        let _ = PROXY.set(event_loop.create_proxy());
        let app = App { pending: vec![(1, attrs)], next_id: 2, current: 1, ..App::default() };
        let mut host = Host { event_loop, app, quit_routed: false };
        if !host.created(1) {
            return false;
        }
        *slot = Some(host);
        true
    })
}

/// A window titled `title`, `width` x `height` logical pixels, hidden until
/// its first frame.
fn attributes(title: &str, width: i64, height: i64) -> WindowAttributes {
    Window::default_attributes()
        .with_title(title)
        .with_inner_size(LogicalSize::new(width.max(1) as f64, height.max(1) as f64))
        .with_visible(false)
}

impl Host {
    /// Pump until window `id` waiting to be created is, or couldn't be.
    fn created(&mut self, id: i64) -> bool {
        let deadline = Instant::now() + OPEN_DEADLINE;
        loop {
            if self.app.wins.iter().any(|w| w.id == id) {
                return true;
            }
            if self.app.failed.contains(&id) {
                return false;
            }
            if Instant::now() >= deadline {
                eprintln!("[snaidhm/window] the platform never let the window be created");
                self.app.pending.retain(|(pid, _)| *pid != id);
                return false;
            }
            self.event_loop.pump_app_events(Some(Duration::from_millis(10)), &mut self.app);
        }
    }
}

/// Open another window, `width` x `height` logical pixels, a little below and
/// right of the current one; its id, 0 when it can't be opened. It draws
/// nothing until `select`ed. Needs a window `open`ed first.
pub fn open_window(title: &str, width: i64, height: i64) -> i64 {
    with_host(0, |host| {
        let id = host.app.next_id;
        host.app.next_id += 1;
        let mut attrs = attributes(title, width, height);
        if let Some(w) = host.app.cur_window() {
            if let Ok(at) = w.outer_position() {
                let s = w.scale_factor();
                attrs = attrs.with_position(winit::dpi::LogicalPosition::new(at.x as f64 / s + 24.0, at.y as f64 / s + 24.0));
            }
        }
        host.app.pending.push((id, attrs));
        if let Some(proxy) = PROXY.get() {
            let _ = proxy.send_event(());
        }
        if host.created(id) { id } else { 0 }
    })
}

/// Close window `id`. The last one closing leaves no screen: render passes
/// draw offscreen.
pub fn close_window(id: i64) {
    with_host((), |host| {
        crate::gpu::detach_surface(id);
        host.app.wins.retain(|w| w.id != id);
        host.app.events.retain(|e| e.win != id);
        if host.app.current == id {
            host.app.current = host.app.wins.first().map_or(0, |w| w.id);
        }
    })
}

/// Make window `id` the one the per-window calls — its size, title, focus,
/// IME, pointer, ... — are about, and the screen render passes draw into.
pub fn select(id: i64) {
    with_host((), |host| {
        if host.app.win(id).is_some() {
            host.app.current = id;
            crate::gpu::select_surface(id);
        }
    })
}

/// The window `select` chose (1 until another is).
pub fn current() -> i64 {
    with_host(0, |host| host.app.current)
}

/// The window the current input happened in.
pub fn event_window() -> i64 {
    with_host(0, |host| host.app.input.win)
}

/// The id of the window with the keyboard focus, 0 when none has it.
pub fn focused_window() -> i64 {
    with_host(0, |host| host.app.wins.iter().find(|w| w.focused).map_or(0, |w| w.id))
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
    let timeout = if timeout_ms < 0 { None } else { Some(Duration::from_millis(timeout_ms as u64)) };
    // The watcher also keeps the time: on macOS winit's pump outlasts its
    // timeout until an event comes — a frame due in 10 ms waited for the next
    // key, and held keys showed a key late.
    #[cfg(unix)]
    watch::arm(fds.iter().map(|&fd| fd as i32).collect(), timeout.map(|t| Instant::now() + t));
    let alive = with_host(false, |host| host.pump(timeout));
    #[cfg(unix)]
    watch::disarm();
    alive
}

/// Wakes the event loop from other threads.
static PROXY: std::sync::OnceLock<EventLoopProxy<()>> = std::sync::OnceLock::new();

/// The thread behind `wait_fds`: while armed it polls the fds, and when one is
/// readable or the deadline passes it wakes the event loop and disarms, so a
/// file nobody has read yet can't keep it spinning. A pipe interrupts its poll
/// when the set changes.
#[cfg(unix)]
mod watch {
    use std::sync::{Condvar, Mutex, OnceLock};

    struct State { fds: Vec<i32>, deadline: Option<std::time::Instant>, armed: bool }

    struct Watch { state: Mutex<State>, changed: Condvar, pipe: (i32, i32) }

    static WATCH: OnceLock<&'static Watch> = OnceLock::new();

    fn get() -> &'static Watch {
        WATCH.get_or_init(|| {
            let mut p = [0i32; 2];
            unsafe { libc::pipe(p.as_mut_ptr()) };
            for fd in p {
                // Not inherited by the programs the window starts.
                unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
                unsafe { libc::fcntl(fd, libc::F_SETFL, libc::fcntl(fd, libc::F_GETFL) | libc::O_NONBLOCK) };
            }
            let w: &'static Watch = Box::leak(Box::new(Watch {
                state: Mutex::new(State { fds: Vec::new(), deadline: None, armed: false }),
                changed: Condvar::new(),
                pipe: (p[0], p[1]),
            }));
            std::thread::spawn(move || run(w));
            w
        })
    }

    fn run(w: &'static Watch) {
        loop {
            let (fds, deadline) = {
                let mut st = w.state.lock().unwrap();
                while !st.armed {
                    st = w.changed.wait(st).unwrap();
                }
                (st.fds.clone(), st.deadline)
            };
            // Rounded up: waking a moment early would only arm it again.
            let wait_ms = deadline.map_or(-1, |d| {
                let left = d.saturating_duration_since(std::time::Instant::now());
                left.as_micros().div_ceil(1000).min(i32::MAX as u128) as i32
            });
            let mut pfds: Vec<libc::pollfd> = fds
                .iter()
                .map(|&fd| libc::pollfd { fd, events: libc::POLLIN, revents: 0 })
                .collect();
            pfds.push(libc::pollfd { fd: w.pipe.0, events: libc::POLLIN, revents: 0 });
            unsafe { libc::poll(pfds.as_mut_ptr(), pfds.len() as libc::nfds_t, wait_ms) };
            let mut buf = [0u8; 64];
            while unsafe { libc::read(w.pipe.0, buf.as_mut_ptr() as *mut _, buf.len()) } > 0 {}
            let ready = pfds[..fds.len()].iter().any(|p| p.revents != 0);
            let due = deadline.is_some_and(|d| std::time::Instant::now() >= d);
            // Still armed for the same wait: a disarm or a new arm meanwhile
            // changed the pipe's state and the loop goes round again.
            let current = { let st = w.state.lock().unwrap(); st.armed && st.deadline == deadline };
            if (ready || due) && current {
                w.state.lock().unwrap().armed = false;
                if let Some(proxy) = super::PROXY.get() {
                    let _ = proxy.send_event(());
                }
            }
        }
    }

    pub fn arm(fds: Vec<i32>, deadline: Option<std::time::Instant>) {
        let w = get();
        let mut st = w.state.lock().unwrap();
        st.fds = fds;
        st.deadline = deadline;
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

/// Which Option keys type as Alt — sending ESC before the key, as a
/// terminal's Meta — rather than composing characters as macOS does
/// (Option+¥ a backslash on a Japanese keyboard, Option+e an accent):
/// 0 neither (the default), 1 both, 2 the left, 3 the right. macOS only;
/// elsewhere Alt is always Alt.
pub fn set_option_as_alt(mode: i64) {
    with_host((), |host| {
        host.app.option_as_alt = mode;
        #[cfg(target_os = "macos")]
        for w in &host.app.wins {
            use winit::platform::macos::{OptionAsAlt, WindowExtMacOS};
            w.window.set_option_as_alt(match mode {
                1 => OptionAsAlt::Both,
                2 => OptionAsAlt::OnlyLeft,
                3 => OptionAsAlt::OnlyRight,
                _ => OptionAsAlt::None,
            });
        }
    });
}

/// Whether the window has the keyboard focus; FOCUS events follow changes.
pub fn focused() -> bool {
    with_host(false, |host| host.app.cur().is_some_and(|w| w.focused))
}

/// Whether the system shows its dark appearance (macOS's, or the desktop's
/// where the platform tells). A THEME event follows each change.
pub fn dark() -> bool {
    with_host(false, |host| {
        host.app.wins.first().and_then(|w| w.window.theme()).map_or(false, |t| t == winit::window::Theme::Dark)
    })
}

/// Asked, when the app is told to quit, whether the program must be asked
/// first. Answering is the program's native code (it knows, say, whether a
/// shell is running something); asked synchronously on the main thread, as
/// the quit can't wait for the program's loop — a logout or restart waits
/// on the answer. Without one, every quit is asked about.
static QUIT_CHECK: std::sync::OnceLock<fn() -> bool> = std::sync::OnceLock::new();

/// Register the quit check (see `QUIT_CHECK`); the first registered stays.
pub fn set_quit_check(needs_asking: fn() -> bool) {
    let _ = QUIT_CHECK.set(needs_asking);
}

/// Take back a close the user asked for (`pump` and the waits returned
/// `false`): after asking whether to close and hearing no. The window stays.
pub fn keep_open() {
    with_host((), |host| host.app.close_requested = false)
}

/// Put the window in the Dock (minimize it).
pub fn minimize() {
    with_host((), |host| {
        if let Some(w) = host.app.cur_window() {
            w.set_minimized(true);
        }
    });
}

/// Where the window is on the screen: the top left of its frame, title bar
/// included, in logical pixels from the top left of the main display (x,
/// then y). (0, 0) when the platform doesn't say (Wayland).
pub fn x() -> f64 {
    with_host(0.0, |host| frame_at(host).0)
}

pub fn y() -> f64 {
    with_host(0.0, |host| frame_at(host).1)
}

fn frame_at(host: &Host) -> (f64, f64) {
    host.app.cur_window().and_then(|w| {
        let s = w.scale_factor();
        w.outer_position().ok().map(|p| (p.x as f64 / s, p.y as f64 / s))
    }).unwrap_or((0.0, 0.0))
}

/// Move the window so the top left of its frame is at (`x`, `y`), logical
/// pixels as `x` and `y` give them. Where nothing of it would show on a
/// display, it stays where it is.
pub fn set_position(x: f64, y: f64) {
    with_host((), |host| {
        if let Some(w) = host.app.cur_window() {
            let screens = displays(w);
            // A title bar's grip on some display: not off the edge of one
            // that is no longer there.
            let visible = screens.is_empty() || screens.iter().any(|&(sx, sy, sw, sh)| {
                x + 80.0 > sx && x < sx + sw - 80.0 && y >= sy - 1.0 && y < sy + sh - 40.0
            });
            if visible {
                w.set_outer_position(LogicalPosition::new(x, y));
            }
        }
    })
}

/// The displays as (x, y, width, height), logical pixels from the top left
/// of the main display; none when the platform doesn't say.
fn displays(w: &Window) -> Vec<(f64, f64, f64, f64)> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let _ = w;
        layer::screens()
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        w.available_monitors().map(|m| {
            let s = m.scale_factor();
            (m.position().x as f64 / s, m.position().y as f64 / s, m.size().width as f64 / s, m.size().height as f64 / s)
        }).collect()
    }
}

/// Size the window's drawable area to `width` x `height` logical pixels.
/// `resized()` reports it, as for a resize by the user.
pub fn set_size(width: f64, height: f64) {
    with_host((), |host| {
        let id = host.app.current;
        let Some(w) = host.app.cur_window().cloned() else { return };
        let asked = w.request_inner_size(LogicalSize::new(width.max(1.0), height.max(1.0)));
        // Done at once on some platforms, without a `Resized` to say so
        // (macOS): the size it has now is the one to draw at.
        let size = asked.unwrap_or_else(|| w.inner_size());
        crate::gpu::resize_surface(id, size.width, size.height);
        if let Some(win) = host.app.win_mut(id) {
            win.size = (size.width, size.height);
            win.resized = true;
        }
    })
}

/// Set the window's title.
pub fn set_title(title: &str) {
    with_host((), |host| {
        if let Some(w) = host.app.cur_window() {
            w.set_title(title);
        }
    });
}

/// Ask for the user's attention while the window is in the background: on
/// macOS the Dock icon bounces once, elsewhere the window is marked urgent.
/// Nothing while it has the focus.
pub fn request_attention() {
    with_host((), |host| {
        if let Some(w) = host.app.cur_window() {
            if !w.has_focus() {
                w.request_user_attention(Some(winit::window::UserAttentionType::Informational));
            }
        }
    });
}

/// The pointer's shape over the window: 0 the arrow, 1 the text I-beam, 2
/// the hand of a link. Set only when it changes.
pub fn set_pointer(kind: i64) {
    with_host((), |host| {
        if let Some(w) = host.app.cur_window() {
            use winit::window::CursorIcon;
            w.set_cursor(match kind {
                1 => CursorIcon::Text,
                2 => CursorIcon::Pointer,
                _ => CursorIcon::Default,
            });
        }
    });
}

/// Fill the screen with the window, or give it back its frame when it does:
/// on macOS a full-screen space of its own, as the green button makes.
pub fn toggle_fullscreen() {
    with_host((), |host| {
        if let Some(w) = host.app.cur_window() {
            w.set_fullscreen(if w.fullscreen().is_some() { None } else { Some(Fullscreen::Borderless(None)) });
        }
    });
}

/// Fill what a frame doesn't cover with this colour (0..1 each) — the edge a
/// window being resized uncovers before the program draws for the new size.
/// macOS only; elsewhere nothing happens.
pub fn set_background(r: f64, g: f64, b: f64) {
    #[cfg(target_os = "macos")]
    with_host((), |host| {
        if let Some(view) = host.app.cur_window().and_then(|w| ns_view(w)) {
            layer::background(view, r, g, b);
        }
    });
    #[cfg(not(target_os = "macos"))]
    let _ = (r, g, b);
}

#[cfg(target_os = "macos")]
fn ns_view(window: &Window) -> Option<*mut std::ffi::c_void> {
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    match window.window_handle().ok()?.as_raw() {
        RawWindowHandle::AppKit(h) => Some(h.ns_view.as_ptr()),
        _ => None,
    }
}

/// The Metal layer wgpu draws into, set up for resizing. macOS draws a
/// window being resized itself while the program waits (the resize is a
/// modal loop), with the last frame stretched to the new size by default:
/// the text swells and shrinks as the edge moves. Pinned to the top left and
/// unscaled, the frame stays as it was and the uncovered edge takes the
/// background colour, until the program draws for the size it settles at.
#[cfg(target_os = "macos")]
mod layer {
    use std::ffi::{c_void, CStr};

    type Id = *mut c_void;

    #[link(name = "objc")]
    extern "C" {
        fn objc_getClass(name: *const std::ffi::c_char) -> Id;
        fn sel_registerName(name: *const std::ffi::c_char) -> Id;
        fn objc_msgSend();
        fn object_getClass(obj: Id) -> Id;
        fn class_replaceMethod(class: Id, name: Id, imp: *const c_void, types: *const std::ffi::c_char) -> *const c_void;
    }
    #[link(name = "QuartzCore", kind = "framework")]
    extern "C" {
        static kCAGravityTopLeft: Id;
    }
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGColorCreateSRGB(r: f64, g: f64, b: f64, a: f64) -> Id;
        fn CGColorRelease(color: Id);
    }

    unsafe fn sel(name: &CStr) -> Id {
        unsafe { sel_registerName(name.as_ptr()) }
    }
    unsafe fn send(obj: Id, name: &CStr) -> Id {
        let f: unsafe extern "C" fn(Id, Id) -> Id = unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel(name)) }
    }
    unsafe fn send_id(obj: Id, name: &CStr, arg: Id) -> Id {
        let f: unsafe extern "C" fn(Id, Id, Id) -> Id = unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel(name), arg) }
    }
    unsafe fn send_index(obj: Id, name: &CStr, i: usize) -> Id {
        let f: unsafe extern "C" fn(Id, Id, usize) -> Id = unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel(name), i) }
    }
    unsafe fn count(obj: Id) -> usize {
        let f: unsafe extern "C" fn(Id, Id) -> usize = unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel(c"count")) }
    }
    unsafe fn is_kind(obj: Id, class: Id) -> bool {
        let f: unsafe extern "C" fn(Id, Id, Id) -> bool = unsafe { std::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel(c"isKindOfClass:"), class) }
    }

    /// The view's layer, then the Metal layers under it.
    fn layers(view: Id) -> Vec<Id> {
        unsafe {
            let root = send(view, c"layer");
            if root.is_null() {
                return Vec::new();
            }
            let metal = objc_getClass(c"CAMetalLayer".as_ptr());
            let subs = send(root, c"sublayers");
            let mut out = vec![root];
            if !subs.is_null() {
                for i in 0..count(subs) {
                    let l = send_index(subs, c"objectAtIndex:", i);
                    if is_kind(l, metal) {
                        out.push(l);
                    }
                }
            }
            out
        }
    }

    pub fn pin(view: Id) {
        for l in layers(view).into_iter().skip(1) {
            unsafe { send_id(l, c"setContentsGravity:", kCAGravityTopLeft) };
        }
    }

    /// Set when the app was asked to quit, for the next pump to report as a
    /// close request.
    pub static QUIT_ASKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    /// NSTerminateCancel, NSTerminateNow.
    const TERMINATE_CANCEL: usize = 0;
    const TERMINATE_NOW: usize = 1;

    unsafe extern "C" fn should_terminate(_this: Id, _cmd: Id, _sender: Id) -> usize {
        // Nothing to ask about: quit at once, so a logout or restart goes on.
        if super::QUIT_CHECK.get().is_some_and(|needs_asking| !needs_asking()) {
            return TERMINATE_NOW;
        }
        QUIT_ASKED.store(true, std::sync::atomic::Ordering::Release);
        if let Some(proxy) = super::PROXY.get() {
            let _ = proxy.send_event(());
        }
        TERMINATE_CANCEL
    }

    /// Make quitting — Cmd+Q, the Dock's Quit, a quit from another app —
    /// a close request the program hears like the close button's: one it
    /// can ask about and take back (`keep_open`), ending the program when it
    /// lets the window close. The app's delegate is told to answer "not
    /// now" to every quit and pass it on. `true` once done; the delegate is
    /// there once the app has finished launching.
    pub fn ask_before_quit() -> bool {
        unsafe {
            let app = send(objc_getClass(c"NSApplication".as_ptr()), c"sharedApplication");
            let delegate = send(app, c"delegate");
            if delegate.is_null() {
                return false;
            }
            let imp: unsafe extern "C" fn(Id, Id, Id) -> usize = should_terminate;
            class_replaceMethod(object_getClass(delegate), sel(c"applicationShouldTerminate:"), imp as *const c_void, c"Q@:@".as_ptr());
            true
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Rect { x: f64, y: f64, w: f64, h: f64 }

    /// Every display's frame, from `NSScreen.screens`, turned to top-left
    /// coordinates: AppKit's run up from the bottom of the main display (the
    /// first). A rect comes back in registers on arm64, so plain
    /// `objc_msgSend` returns it.
    #[cfg(target_arch = "aarch64")]
    pub fn screens() -> Vec<(f64, f64, f64, f64)> {
        unsafe {
            let all = send(objc_getClass(c"NSScreen".as_ptr()), c"screens");
            if all.is_null() {
                return Vec::new();
            }
            let frame: unsafe extern "C" fn(Id, Id) -> Rect = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            let rects: Vec<Rect> = (0..count(all)).map(|i| frame(send_index(all, c"objectAtIndex:", i), sel(c"frame"))).collect();
            let Some(main) = rects.first().copied() else { return Vec::new() };
            rects.iter().map(|r| (r.x, main.h - (r.y + r.h), r.w, r.h)).collect()
        }
    }

    pub fn background(view: Id, r: f64, g: f64, b: f64) {
        unsafe {
            let color = CGColorCreateSRGB(r, g, b, 1.0);
            for l in layers(view) {
                send_id(l, c"setBackgroundColor:", color);
            }
            CGColorRelease(color);
        }
    }
}

/// `true` exactly once after each change of the window's size — the moment to
/// rebuild anything sized to it (a scene, the depth target).
pub fn resized() -> bool {
    with_host(false, |host| {
        let id = host.app.current;
        host.app.win_mut(id).is_some_and(|w| std::mem::take(&mut w.resized))
    })
}

/// Width of the drawable area in physical pixels, the size render targets are.
pub fn width() -> i64 {
    with_host(0, |host| host.app.cur().map_or(0, |w| w.size.0 as i64))
}

/// Height of the drawable area in physical pixels.
pub fn height() -> i64 {
    with_host(0, |host| host.app.cur().map_or(0, |w| w.size.1 as i64))
}

/// The kind of the next queued input, which the `event_*` accessors then
/// describe; 0 once the queue is empty. Inputs queue up across `pump` calls
/// until read.
pub fn next_event() -> i64 {
    with_host(0, |host| {
        let input = host.app.events.pop_front().unwrap_or_default();
        let kind = input.kind;
        host.app.input = input;
        kind
    })
}

/// Pointer position of the current input, logical pixels.
pub fn event_x() -> f64 {
    with_host(0.0, |host| host.app.input.x)
}

pub fn event_y() -> f64 {
    with_host(0.0, |host| host.app.input.y)
}

/// Wheel distance of the current input, logical pixels, DOM signs.
pub fn event_dx() -> f64 {
    with_host(0.0, |host| host.app.input.dx)
}

pub fn event_dy() -> f64 {
    with_host(0.0, |host| host.app.input.dy)
}

/// Button, typed code point, or DOM key code of the current input.
pub fn event_code() -> i64 {
    with_host(0, |host| host.app.input.code)
}

/// Physical pixels per logical pixel (2.0 on a typical Retina display).
pub fn scale_factor() -> f64 {
    with_host(1.0, |host| host.app.cur_window().map_or(1.0, |w| w.scale_factor()))
}

/// End of the marked part of the current composition, characters.
pub fn event_end() -> i64 {
    with_host(0, |host| host.app.input.end)
}

/// Modifiers held when the current input happened: 1 Shift, 2 Ctrl, 4 Alt
/// (Option), 8 Super (Command).
pub fn event_mods() -> i64 {
    with_host(0, |host| host.app.input.mods)
}

/// Text of the current composition.
pub fn event_text() -> String {
    with_host(String::new(), |host| host.app.input.text.clone())
}

/// Let the platform's input method compose text in this window (`true`), or
/// take keys as they are (`false`, the default).
pub fn set_ime(allowed: bool) {
    with_host((), |host| {
        if let Some(w) = host.app.cur_window() {
            w.set_ime_allowed(allowed);
        }
    })
}

/// Where the caret is — `x`, `y`, `w`, `h` in logical pixels — so the input
/// method can put its candidate window beside it rather than over it.
pub fn set_ime_area(x: f64, y: f64, w: f64, h: f64) {
    with_host((), |host| {
        if let Some(win) = host.app.cur_window() {
            win.set_ime_cursor_area(LogicalPosition::new(x, y), LogicalSize::new(w.max(1.0), h.max(1.0)));
        }
    })
}
