$log = 'C:\liars\restart_when_idle.log'
$src = 'C:\liars\tells\bin\Release\net6.0\LiarsTells.dll'
$dst = "C:\Program Files (x86)\Steam\steamapps\common\Liar's Bar\Mods\LiarsTells.dll"
$deadline = (Get-Date).AddHours(2)
while ((Get-Date) -lt $deadline) {
  $st = (Get-Content C:\liars\aq_heartbeat.txt -ErrorAction SilentlyContinue) -split '\|' | Select-Object -Last 1
  # WaitingForGame is NOT idle in ranked (match found, game loading; 2026-11-06 14:08 this killed a game at round 1)
  if ($st -in @('OpenLobbyList','WaitMainMenu','ClickEnterBar','ClickMatchmaking','ClickPlay')) {
    "$(Get-Date -Format HH:mm:ss) idle state $st, killing game to swap DLL" | Out-File $log -Append
    Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath -like '*Liar*Bar*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    $ok = $false
    for ($i = 0; $i -lt 40; $i++) {
      Start-Sleep -Milliseconds 500
      try { Copy-Item $src $dst -Force -ErrorAction Stop; $ok = $true; break } catch {}
    }
    "$(Get-Date -Format HH:mm:ss) dll copied=$ok; watchdog will relaunch" | Out-File $log -Append
    break
  }
  Start-Sleep -Seconds 5
}
