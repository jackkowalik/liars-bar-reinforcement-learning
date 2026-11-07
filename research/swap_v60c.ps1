# Proper swap: between games, stop the advisor AND its cmd.exe launcher window (cmd re-reads the bat from disk
# and relaunched the binary itself on the first attempt), then let the watchdog start run_current_advisor.bat fresh.
$log = "C:\liars\research\swap_v60c.log"
function Log($m) { $line = "$(Get-Date -Format 'HH:mm:ss') $m"; Add-Content -Path $log -Value $line; Write-Output $line }
Log "swap_v60c started (revert to v60: v62 guard premise wrong, 1v1 forced call); waiting for a between-games state"
$deadline = (Get-Date).AddHours(6)
while ((Get-Date) -lt $deadline) {
    $hb = Get-Content "C:\liars\aq_heartbeat.txt" -ErrorAction SilentlyContinue
    $state = ""
    if ($hb) { $state = ($hb -split '\|')[-1].Trim() }
    if ($state -ne "" -and $state -ne "InGame" -and $state -ne "StuckWaiting" -and $state -ne "InLobby" -and $state -ne "WaitingForGame") {
        Log "state=$state -> swapping"
        Copy-Item "C:\liars\run_v60_ranked.bat" "C:\liars\run_current_advisor.bat" -Force
        $adv = Get-CimInstance Win32_Process -Filter "Name = 'liars.exe'" | Where-Object { $_.CommandLine -like "*watch*--auto*" }
        $parents = @($adv | ForEach-Object { $_.ParentProcessId })
        foreach ($p in $adv) { Log "stopping advisor pid $($p.ProcessId)"; Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
        foreach ($pid in $parents) {
            $c = Get-CimInstance Win32_Process -Filter "ProcessId = $pid" -ErrorAction SilentlyContinue
            if ($c -and $c.Name -eq "cmd.exe") { Log "stopping launcher cmd pid $pid ($($c.CommandLine))"; Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue }
        }
        # any other stale launcher windows of the advisor bat
        Get-CimInstance Win32_Process -Filter "Name = 'cmd.exe'" | Where-Object { $_.CommandLine -like "*run_current_advisor.bat*" -or $_.CommandLine -like "*run_v6*_*.bat*" } | ForEach-Object { Log "stopping stale launcher pid $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        for ($i = 0; $i -lt 18; $i++) {
            Start-Sleep -Seconds 5
            $new = Get-CimInstance Win32_Process -Filter "Name = 'liars.exe'" | Where-Object { $_.CommandLine -like "*watch*--auto*" }
            if ($new) { break }
        }
        foreach ($p in $new) {
            $par = Get-CimInstance Win32_Process -Filter "ProcessId = $($p.ParentProcessId)" -ErrorAction SilentlyContinue
            $title = (Get-Process -Id $p.ParentProcessId -ErrorAction SilentlyContinue).MainWindowTitle
            Log "advisor running pid $($p.ProcessId) started $($p.CreationDate); launcher pid $($p.ParentProcessId) started $($par.CreationDate); window title: $title"
        }
        if (-not $new) { Log "WARNING: no advisor process 90 s after the stop" }
        Log "done"
        exit 0
    }
    Start-Sleep -Seconds 5
}
Log "gave up after 6 h without a between-games state"
