"""
Liar's Bar AFK farm watchdog.
Monitors the game process and heartbeat file. Restarts the game if:
  - The game process dies/crashes
  - The heartbeat file stops updating for 90 seconds
  - The restart signal file appears

Usage:  python watchdog.py
        Ctrl+C to stop.
"""
import subprocess, time, os, sys

GAME_EXE = r"C:\Program Files (x86)\Steam\steamapps\common\Liar's Bar\Liar's Bar.exe"
STEAM_EXE = r"C:\Program Files (x86)\Steam\steam.exe"
STEAM_APPID = "3097560"
HEARTBEAT = r"C:\liars\aq_heartbeat.txt"
RESTART   = r"C:\liars\aq_restart.txt"
AUTOSTART = r"C:\liars\aq_autostart.txt"
HEARTBEAT_TIMEOUT = 300  # seconds without heartbeat update = stuck
# 2026-11-03: the advisor (liars.exe watch --auto) died at 18:39 during a swap and nobody noticed for 40 min.
# The watchdog now relaunches it from this bat when no such process exists. Swap advisors by changing the bat.
ADVISOR_BAT = "C:/liars/run_current_advisor.bat"
ADVISOR_COOLDOWN = 60  # seconds between relaunch attempts
CHECK_INTERVAL = 5

def game_running():
    """Check if the actual game is running (not the advisor)."""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath -like '*Liar*Bar*' } | Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True
    )
    return any(line.strip().isdigit() for line in result.stdout.splitlines())

def advisor_running():
    """A liars.exe whose command line is the advisor in auto mode."""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"name='liars.exe'\" | Where-Object { $_.CommandLine -like '* watch *' -and $_.CommandLine -like '*--auto*' } | Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True
    )
    return any(line.strip().isdigit() for line in result.stdout.splitlines())

def start_advisor():
    print(f"[watchdog] advisor not running, starting {ADVISOR_BAT}")
    subprocess.Popen(["powershell", "-NoProfile", "-Command",
                      f'Start-Process -FilePath "{ADVISOR_BAT}" -WorkingDirectory "C:/liars/rust" -WindowStyle Minimized'])

def heartbeat_age():
    """Returns (age_seconds, exists). age is None if file missing."""
    try:
        mtime = os.path.getmtime(HEARTBEAT)
        return time.time() - mtime, True
    except FileNotFoundError:
        return None, False

def start_game():
    print(f"[watchdog] starting game via Steam...")
    # Write autostart flag so the mod enables auto-queue + farm mode on load
    with open(AUTOSTART, "w") as f:
        f.write("go")
    subprocess.Popen(["powershell", "-NoProfile", "-Command", 'Start-Process "steam://rungameid/3097560"'])

def kill_game():
    print(f"[watchdog] killing game process...")
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath -like '*Liar*Bar*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
        capture_output=True
    )
    time.sleep(3)

def cleanup():
    for f in [RESTART, HEARTBEAT]:
        try: os.remove(f)
        except FileNotFoundError: pass

def main():
    print("[watchdog] Liar's Bar AFK farm watchdog started")
    print(f"  heartbeat: {HEARTBEAT}")
    print(f"  timeout:   {HEARTBEAT_TIMEOUT}s")
    print(f"  check:     every {CHECK_INTERVAL}s")
    print()

    # If game isn't running, start it
    if not game_running():
        cleanup()
        start_game()
        time.sleep(120)  # give game time to fully load + reach menu

    last_advisor_start = 0.0
    while True:
        try:
            time.sleep(CHECK_INTERVAL)

            # Advisor alive? (independent of the game)
            if os.path.exists(ADVISOR_BAT) and time.time() - last_advisor_start > ADVISOR_COOLDOWN and not advisor_running():
                last_advisor_start = time.time()
                start_advisor()

            # Check for restart signal from the mod
            if os.path.exists(RESTART):
                print("[watchdog] restart signal received")
                cleanup()
                time.sleep(5)  # let the game close
                if game_running():
                    kill_game()
                start_game()
                time.sleep(120)  # give game time to fully load + reach menu
                continue

            # Check if game is running
            if not game_running():
                print("[watchdog] game not running, restarting...")
                cleanup()
                start_game()
                time.sleep(120)  # give game time to fully load + reach menu
                continue

            # Check heartbeat freshness (only if heartbeat file exists)
            age, hb_exists = heartbeat_age()
            if hb_exists and age > HEARTBEAT_TIMEOUT:
                print(f"[watchdog] heartbeat stale ({age:.0f}s old), killing zombies and restarting...")
                kill_game()
                cleanup()
                time.sleep(5)
                start_game()
                time.sleep(120)  # give game time to fully load + reach menu
                continue

            # Status
            if hb_exists:
                try:
                    with open(HEARTBEAT) as f:
                        state = f.read().split("|", 1)[-1].strip()
                except:
                    state = "?"
                print(f"[watchdog] ok — game running, heartbeat {age:.0f}s ago, state: {state}")

        except KeyboardInterrupt:
            print("\n[watchdog] stopped")
            sys.exit(0)
        except Exception as e:
            print(f"[watchdog] error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
