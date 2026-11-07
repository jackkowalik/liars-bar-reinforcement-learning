@echo off
title Liar's Bar advisor - raw network, min 3.5 s think time
rem Run this from the folder that holds liars.exe and weights\ (the release zip layout). The mod writes the game log to
rem C:\liars\logs and reads the chosen action from C:\liars\aq_action.txt; the advisor tails the first and writes the second.
cd /d %~dp0
set LIARS_NET=%~dp0weights\basic_ppo2
set LIARS_DEVIL_NET=%~dp0weights\devil_mlp
set LIARS_ORDER=0
set LIARS_CALL_RULE=0
set LIARS_OPEN_ONE=0
set LIARS_DELAY_MIN=4.5
set LIARS_DELAY_MAX=7
rem LIARS_TRUTH_FIRST=1 enables the truth-first variant (simulator +5 to +10 points, not yet trialled live).
if not exist C:\liars\logs mkdir C:\liars\logs
liars.exe watch C:\liars\logs --auto
pause
