@echo off
title Advisor v60t - raw network (v60) + min 3.5 s think time, QuickEdit off
cd /d C:\liars\rust
rem v60 (2026-11-06): research baseline. The simulator (research/RESEARCH.md, harness results 2-3) says the v54-v59
rem wrapper costs ~18 points against a population fitted to the logs: ordering rules ~17, lie-detector call override ~12
rem (overlapping). This runs the Devil network as trained: no ordering rules, its own call probability, no lie detector.
rem Pre-registered metrics over >= 200 games: win rate (death events), died-first rate, my pulls/game by cause.
set LIARS_ORDER=0
set LIARS_CALL_RULE=0
set LIARS_OPEN_ONE=0
set LIARS_DELAY_MIN=4.5
set LIARS_DELAY_MAX=7
C:\liars\rust\target\v64\release\liars.exe watch C:\liars\logs --auto --net C:/liars/nn/basic_ppo2 --net-devil C:/liars/nn/devil_tnet_r1v1
pause
