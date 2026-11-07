# Deploy the built mod DLL into the game's Mods folder. A loaded DLL cannot be overwritten but CAN be renamed on Windows, so:
# rename the live DLL aside, copy the new one in (no need to stop anything), then at the next idle menu state stop the game and
# let the watchdog relaunch it with the new mod. The advisor process is never touched.
# 2026-11-07: stop-then-copy lost the race twice (file locked ~30 s after the kill); the first rename version of this script was
# generated with mangled backslashes and left Mods without a DLL for a minute. Keep this file hand-written.
$log  = 'C:\liars\research\deploy_dll_idle.log'
function Log($m) { Add-Content -Path $log -Value "$(Get-Date -Format 'HH:mm:ss') $m" }
$mods = "C:\Program Files (x86)\Steam\steamapps\common\Liar's Bar\Mods"
$src  = 'C:\liars\tells\bin\Release\net6.0\LiarsTells.dll'
$dst  = Join-Path $mods 'LiarsTells.dll'
if (-not (Test-Path $src)) { Log "source DLL missing: $src"; exit 1 }
Log "started (source $((Get-Item $src).Length) bytes, $((Get-Item $src).LastWriteTime))"
$aside = Join-Path $mods ("LiarsTells.dll.old_" + (Get-Date -Format 'MMdd_HHmm'))
try { Move-Item $dst $aside -ErrorAction Stop; Log "renamed live DLL aside -> $aside" } catch { Log "rename failed: $($_.Exception.Message)"; exit 1 }
try { Copy-Item $src $dst -ErrorAction Stop; Log "copied new DLL ($((Get-Item $dst).Length) bytes)" }
catch { Log "copy failed: $($_.Exception.Message); restoring the old DLL"; Move-Item $aside $dst -Force; exit 1 }
$deadline = (Get-Date).AddHours(2)
while ((Get-Date) -lt $deadline) {
  $st = ((Get-Content C:\liars\aq_heartbeat.txt -ErrorAction SilentlyContinue) -split '\|' | Select-Object -Last 1).Trim()
  if ($st -in @('OpenLobbyList','WaitMainMenu','ClickEnterBar','ClickMatchmaking','ClickPlay')) {
    Log "idle state ${st}: stopping the game; the watchdog relaunches it with the new mod"
    Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath -like "*Liar's Bar*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Log "done"
    exit 0
  }
  Start-Sleep -Seconds 5
}
Log "gave up after 2 h without an idle state (new DLL is in place; it loads at the next game start)"
