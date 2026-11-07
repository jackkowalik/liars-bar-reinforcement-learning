//! Terminal styling for the live advisor: ANSI colours and filled percentage bars.

pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[2m";
pub const RED: &str = "\x1b[91m";
pub const GREEN: &str = "\x1b[92m";
pub const YELLOW: &str = "\x1b[93m";
pub const CYAN: &str = "\x1b[96m";
pub const MAGENTA: &str = "\x1b[95m";

/// Windows consoles only render ANSI colours once virtual-terminal processing is switched on.
#[cfg(windows)]
pub fn enable_colors() {
    type Handle = *mut core::ffi::c_void;
    extern "system" {
        fn GetStdHandle(n: u32) -> Handle;
        fn GetConsoleMode(h: Handle, mode: *mut u32) -> i32;
        fn SetConsoleMode(h: Handle, mode: u32) -> i32;
    }
    const STD_OUTPUT_HANDLE: u32 = -11i32 as u32;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
    // SAFETY: plain Win32 console calls on this process's own stdout handle.
    unsafe {
        let h = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut mode = 0u32;
        if GetConsoleMode(h, &mut mode) != 0 {
            SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }
}

#[cfg(not(windows))]
pub fn enable_colors() {}

/// A `width`-cell bar filled to `p` (0..1), in `color`.
pub fn bar(p: f64, width: usize, color: &str) -> String {
    let filled = ((p.clamp(0.0, 1.0) * width as f64).round() as usize).min(width);
    format!("{color}{}{DIM}{}{RESET}", "█".repeat(filled), "░".repeat(width - filled))
}

/// "  label  ███████░░░  62%" with the percentage right-aligned.
pub fn bar_line(label: &str, p: f64, color: &str) -> String {
    format!("  {label}  {}  {BOLD}{:>3.0}%{RESET}", bar(p, 24, color), p * 100.0)
}

pub fn rule(title: &str) -> String {
    if title.is_empty() {
        return format!("{CYAN}{}{RESET}", "━".repeat(50));
    }
    let pad = 46usize.saturating_sub(title.chars().count() + 2);
    format!("{CYAN}{BOLD}━━━ {title} {}{RESET}", "━".repeat(pad))
}
