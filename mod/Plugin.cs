using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Il2Cpp;
using Il2CppInterop.Runtime;
using MelonLoader;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using Il2CppSteamworks;
using Il2CppTMPro;
using R = Il2CppSystem.Reflection;
using System.Runtime.InteropServices;
using System.Net;
using System.Text.RegularExpressions;
using System.Threading;

[assembly: MelonInfo(typeof(LiarsTells.Plugin), "LiarsTells", "0.1.0", "liars")]
[assembly: MelonGame("Curve Animation", "Liar's Bar")]

namespace LiarsTells
{
    /// <summary>
    /// Read-only game logger for Liar's Deck (basic and matchmaking modes).
    ///
    /// Records only what anyone at the table can see: my own hand, the table card, how many
    /// cards each player holds and throws, whose turn it is, revolver pulls, deaths, every
    /// player's head pose, and the thrown cards once a call has revealed them.
    /// It never reads other players' card values, any revolver's bullet position, or the
    /// thrown cards before a reveal, and it writes nothing back to the game.
    ///
    /// F10  dump field names of the game classes (to find head / look fields)
    /// F11  pause / resume logging
    /// Logs go to C:\liars\logs\game_*.jsonl, one JSON object per line.
    /// </summary>
    public class Plugin : MelonMod
    {
        const float PoseInterval = 0.05f; // 20 Hz
        const float DealSettle = 0.75f;   // wait for every hand to be dealt before logging a round
        static readonly string LogDir = @"C:\liars\logs";
        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        class Seat
        {
            public GameObject Go;
            public string Name;
            public bool Dead, Turn;
            public int Chamber, Hand;
            public int MinHand = 99; // fewest cards held this round (0 = hidden, ignored)
            public Transform Head;
            public string HeadSource;
            public int Slot;
            public int Id;
            public bool Aiming;
            // Tamper detection. Hidden values are only compared with their previous value and are
            // never written to the log or used for advice.
            public int Bullet = -1;
            public string CardSig = "";
            public float YawPrev = float.NaN, YawCum, YawMin, YawMax;
            public bool HeadFlagged;
            /// Died this game. The game can later reset a dead (often departed) player's seat, so
            /// they are left out of tamper checks until a new game starts.
            public bool EverDead;
            public int Aim = -1;
            public bool Looking;
            /// Their player object vanished from the scene mid-game (left or dropped); logged once.
            public bool Left;
        }

        // Public facts about my own connection, for working out why games end without a result:
        // am I the lobby host, and is my client still connected when the log closes.
        static bool IsHost() { try { return Il2CppMirror.NetworkServer.active; } catch { return false; } }
        static bool IsConnected() { try { return Il2CppMirror.NetworkClient.isConnected; } catch { return false; } }

        // My own ladder numbers as the game client holds them (DatabaseManager): logged at the start and
        // end of every game so the cost of a win, a death and a dropped game can be measured.
        void WriteMmr(string when)
        {
            try
            {
                var d = DatabaseManager.instance;
                if (d == null) return;
                Write(new J("mmr").F("t", Time.time).S("when", when).I("deck_mmr", (int)d.DeckMMR5).I("deck_wins", (int)d.DeckWins)
                    .I("deck_losses", (int)d.DeckLoses).I("wins", (int)d.CurrentWins).I("losses", (int)d.CurrentLoses).I("level", (int)d.Level));
            }
            catch (Exception ex) { MelonLogger.Msg("mmr read failed: " + ex.Message); }
        }

        // The lobby-side player object carries each seat's rank tier and whether they are the host.
        static (int rank, bool host) LobbyInfo(GameObject go, string name)
        {
            try
            {
                foreach (var poc in UnityEngine.Object.FindObjectsOfType<PlayerObjectController>())
                {
                    bool match = false;
                    try { match = poc.PlayerObject != null && go != null && poc.PlayerObject.GetInstanceID() == go.GetInstanceID(); } catch { }
                    if (!match) { try { match = !string.IsNullOrEmpty(name) && poc.PlayerName == name; } catch { } }
                    if (match) return ((int)poc.Rank, poc.isnewHost);
                }
            }
            catch { }
            return (-1, false);
        }

        readonly List<Seat> seats = new List<Seat>();
        GameObject me;
        string mode;
        /// I died in the game being logged (the log then follows the rest as a spectator).
        bool iDied;
        StreamWriter log;
        bool paused;
        float nextPose;
        float roundAt = -1f;      // time a deal was seen; the round event is written after DealSettle
        int round;
        string myHandSig = "";   // my hand at the last "hand" event
        int lastThrower = -1;     // seat of the most recent play (the player a call would accuse)
        int callerCandidate = -1; // first seat to get the turn after that play
        int turnSeat = -1;
        float turnStart;
        bool pulledThisRound;                         // a call happened: no more plays this round
        readonly HashSet<int> pulledSeats = new HashSet<int>(); // a revealed Devil makes several seats pull
        int callAccused, callCaller;
        // Chaos manager flags: a Liar call, a Chaos throw, a Master shot, a caught liar.
        bool fLiar, fChaos, fMaster, fCaught;
        bool callLie;
        List<int> callRevealed;

        bool farmMode;
        float farmNextCheck;   // the menu scene re-applies the saved display settings; undo that while farming

        public override void OnInitializeMelon() => instance = this;

        void SetFarmMode(bool on, bool quiet = false)
        {
            farmMode = on;
            // Keep ticking when the window is unfocused or minimised: auto-play no longer needs focus.
            Application.runInBackground = true;
            if (on)
            {
                Application.targetFrameRate = 10;
                QualitySettings.SetQualityLevel(0, true);
                QualitySettings.shadows = ShadowQuality.Disable;
                QualitySettings.antiAliasing = 0;
                QualitySettings.vSyncCount = 0;
                QualitySettings.lodBias = 0.1f;
                QualitySettings.masterTextureLimit = 4;
                Screen.SetResolution(640, 480, FullScreenMode.Windowed);
                if (!quiet) MelonLogger.Msg("FARM MODE ON — 640x480, 10fps, min quality");
            }
            else
            {
                Application.targetFrameRate = -1;
                QualitySettings.SetQualityLevel(QualitySettings.names.Length - 1, true);
                QualitySettings.vSyncCount = 1;
                Screen.SetResolution(1920, 1080, true);
                MelonLogger.Msg("FARM MODE OFF — restored defaults");
            }
        }

        // ── Auto-queue state machine ─────────────────────────────────────────
        //
        // F8 toggles auto-queue. When active it cycles:
        //   MainMenu → open lobby list → find best deck lobby → join → ready →
        //   play game (auto-play via advisor) → on kick/end → back to MainMenu.

        bool autoQueue;
        enum AQState { Idle, WaitMainMenu, ClickEnterBar, ClickMatchmaking, ClickPlay, WaitingForGame, InGame, StuckWaiting, OpenLobbyList, JoiningLobby, InLobby }
        // Lobby mode (default): ENTER THE BAR -> LOBBY LIST -> join the fullest open public lobby -> ready up ->
        // wait for the host. Never touches the MATCHMAKING / Find Match (ranked) queue. Creating
        // C:\liars\aq_ranked.txt switches back to the old matchmaking flow.
        const string AQ_RANKED_FLAG = @"C:\liars\aq_ranked.txt";
        bool AQRankedMode() { try { return File.Exists(AQ_RANKED_FLAG); } catch { return false; } }
        float aqLobbyEnteredAt, aqLobbyAloneSince, aqLobbyNextReady, aqLobbyListOpenedAt, aqLobbyNextRefresh;
        bool aqLobbyModeLogged;
        AQState aqState = AQState.Idle;
        float aqTimer;
        float aqLobbyJoinedAt;
        float aqMatchFoundAt;            // when the ranked screen first said a match was found (0 = not yet)
        bool aqConnectedAtSearch;        // Mirror already claimed "connected" when the search started (stale client)
        const string AQ_WEDGE_MARK = @"C:\liars\aq_wedge_restart.txt";
        float aqLastStateChange;
        float aqPopupScanAt;             // connection-quality popup: last scan time
        float aqPopupFirstAt;            // first popup in the current 15-minute window
        int aqPopupCount;                // popups in that window; 3 -> restart the game
        float aqWrongModeAt;
        float aqLastGameActivity;
        int aqRetries;
        const string AQ_HEARTBEAT = @"C:\liars\aq_heartbeat.txt";
        const string AQ_RESTART   = @"C:\liars\aq_restart.txt";
        const string AQ_AUTOSTART = @"C:\liars\aq_autostart.txt";
        const float  AQ_WATCHDOG  = 600f;
        readonly Dictionary<string, float> aqBlacklist = new Dictionary<string, float>();
        readonly string[] aqPermaBan = { "45699" };
        string aqCurrentLobby = "";
        CSteamID aqJoinedLobbyId;        // the lobby we last asked Steam to join (for an explicit LeaveLobby)
        float aqSceneStuckSince;         // WaitMainMenu: when we first noticed we are not in the menu scene
        const float AQ_BLACKLIST = 180f; // avoid same lobby for 3 min

        // ── Auto-play ────────────────────────────────────────────────────────
        System.Reflection.MethodInfo apThrowMethod;
        System.Reflection.MethodInfo apCallMethod;
        bool apMethodsSearched;
        float apTurnDelay;
        bool apActedThisTurn;
        bool apStaleLogged;
        bool apNoCardsLogged;

        /// True when our HaveTurn flag cannot be trusted: another live seat also holds the turn, or we were the
        /// most recent thrower and other players are still alive.
        bool APTurnIsStale(out string why)
        {
            why = null;
            try
            {
                int mine = -1, others = 0, alive = 0;
                for (int i = 0; i < seats.Count; i++)
                {
                    var g = seats[i].Go; if (g == null) continue;
                    bool isMe = IsMe(g), dead = DeadOf(g);
                    if (isMe) mine = i;
                    if (!dead) alive++;
                    if (!isMe && !dead && TurnOf(g)) others++;
                }
                if (others > 0) { why = others + " other live seat(s) hold the turn"; return true; }
                if (mine >= 0 && lastThrower == mine && alive > 1) { why = "we were the last to throw"; return true; }
            }
            catch { }
            return false;
        }
        /// When this turn began, and the advisor's action line once read (null = not read yet).
        float apTurnStart;
        string apPendingLine;
        const string AP_ACTION = @"C:\liars\aq_action.txt";
        /// ThrowCards()/RequestCallLiar() send a Mirror command; the play shows up for everyone ~0.9s later
        /// (measured 2026-11-02: selected -> hand shrank = 0.90s on 30/30 throws). Fire that much early so the
        /// advisor's "delay=" lands when it says, not a second late.
        const float AP_EXEC_LEAD = 1.0f;

        void ToggleAutoQueue()
        {
            autoQueue = !autoQueue;
            if (autoQueue)
            {
                AQChangeState(AQState.WaitMainMenu);
                aqTimer = 0f;
                aqRetries = 0;
                MelonLogger.Msg("AUTO-QUEUE ON — scene='" + SceneManager.GetActiveScene().name + "'");
                AQHeartbeat("started");
            }
            else
            {
                aqState = AQState.Idle;
                MelonLogger.Msg("AUTO-QUEUE OFF");
                try { File.Delete(AQ_HEARTBEAT); } catch { }
            }
        }

        void AQHeartbeat(string state)
        {
            try { File.WriteAllText(AQ_HEARTBEAT, DateTime.UtcNow.ToString("o") + "|" + state); } catch { }
        }

        void AQRequestRestart()
        {
            MelonLogger.Msg("[AQ] requesting game restart...");
            try { File.WriteAllText(AQ_RESTART, DateTime.UtcNow.ToString("o")); } catch { }
            Application.Quit();
        }

        static void AQCall(Component c, string method)
        {
            var m = c.GetIl2CppType().GetMethod(method, R.BindingFlags.Public | R.BindingFlags.NonPublic | R.BindingFlags.Instance);
            if (m != null) m.Invoke(c, null);
        }

        // Detect where we are
        bool AQHasVLC() => AQLobby() != null;
        bool aqDeckFilterApplied;
        float aqDeckWrongSince;      // when the lobby's deck first read as non-Devil (0 = fine)
        float aqReadyClickedAt;      // last time we pressed Ready (0 = never)
        bool aqReadyConfirmed;
        bool aqReadyWaitLogged;
        float aqDeadSince;
        float aqDisconnectedSince;   // when NetworkClient.isConnected first read false during a game (0 = connected)               // when the local player was first seen dead this game (0 = alive)
        const float AQ_DEAD_LEAVE_DELAY = 6f;  // let the death animation / game log finish before leaving
        const float AQ_DECK_GRACE = 6f;  // let the synced deck mode arrive before judging it
        bool AQAtMainScene() => SceneManager.GetActiveScene().name == "SteamTest";
        void AQSetLobbyPanel(bool active)
        {
            foreach (var t in Resources.FindObjectsOfTypeAll<Transform>())
            {
                try
                {
                    if (t.name == "substate_lOBBY")
                    {
                        t.gameObject.SetActive(active);
                        return;
                    }
                }
                catch { }
            }
        }
        int AQGetGameMode()
        {
            try
            {
                var mgr = Manager.Instance;
                if (mgr == null) return -1;
                var prop = mgr.GetIl2CppType().GetProperty("GameMode");
                if (prop != null) return (int)prop.GetGetMethod().Invoke(mgr, null).Unbox<int>();
            }
            catch { }
            return -1;
        }
        int AQGetDeckMode()
        {
            try
            {
                var mgr = Manager.Instance;
                if (mgr == null) return -1;
                var prop = mgr.GetIl2CppType().GetProperty("DeckMode");
                if (prop != null) return (int)prop.GetGetMethod().Invoke(mgr, null).Unbox<int>();
            }
            catch { }
            return -1;
        }
        bool AQIsWrongMode()
        {
            int dm = AQGetDeckMode();
            if (dm > 0) { MelonLogger.Msg("[AQ] DeckMode=" + dm + ", wrong mode"); return true; }

            foreach (var go in UnityEngine.Object.FindObjectsOfType<GameObject>())
            {
                if (!go.activeInHierarchy) continue;
                string n = go.name;
                string lo = n.ToLowerInvariant();
                if (lo.Contains("poker") || lo.Contains("dice") || lo.Contains("spin")
                    || lo.Contains("chaos") || lo.Contains("slot"))
                {
                    if (lo.Contains("mode") || lo.Contains("button") || lo.Contains("image")
                        || lo.Contains("parent") || lo.Contains("toggle") || lo.Contains("panel"))
                    {
                        MelonLogger.Msg("[AQ] wrong mode object: " + n);
                        return true;
                    }
                }
            }

            // Also check lobby text for mode names
            foreach (var txt in UnityEngine.Object.FindObjectsOfType<UnityEngine.UI.Text>())
            {
                if (txt == null || !txt.gameObject.activeInHierarchy) continue;
                string t = txt.text.ToLowerInvariant();
                if (t.Contains("poker") || t.Contains("dice") || t.Contains("spin")
                    || t.Contains("slot") || t.Contains("chaos"))
                {
                    MelonLogger.Msg("[AQ] wrong mode text: '" + txt.text + "' on " + txt.gameObject.name);
                    return true;
                }
            }
            return false;
        }

        bool AQInGame()
        {
            try { if (DeckGamePlayManager.Instance != null) return true; } catch { }
            try { if (ChaosDeckGamePlayManager.Instance != null) return true; } catch { }
            var mgr = Manager.Instance;
            if (mgr == null) return false;
            var lp = mgr.GetLocalPlayer();
            if (lp == null) return false;
            var go = lp.gameObject;
            return go.GetComponent<DeckGameplay>() != null
                || go.GetComponent<BlorfGamePlayMatchMaking>() != null
                || go.GetComponent<BlorfGamePlay>() != null
                || go.GetComponent<ChaosDeckGameplay>() != null;
        }
        bool AQIsDead()
        {
            var mgr = Manager.Instance;
            if (mgr == null) return false;
            var lp = mgr.GetLocalPlayer();
            if (lp == null) return false;
            var ps = lp.gameObject.GetComponent<PlayerStats>();
            return ps != null && ps.Dead;
        }

        bool AQClickButton(string nameContains)
        {
            foreach (var btn in UnityEngine.Object.FindObjectsOfType<Button>())
            {
                if (btn.gameObject.name.ToLowerInvariant().Contains(nameContains.ToLowerInvariant())
                    && btn.gameObject.activeInHierarchy && btn.interactable)
                {
                    MelonLogger.Msg("[AQ] clicking button: " + btn.gameObject.name);
                    btn.onClick.Invoke();
                    return true;
                }
            }
            return false;
        }

        /// Ranked queue through the game's own MatchmakerTicketer (the "Find Match Button" has no Button
        /// component, so the old path moved the real mouse). OnFindMatch() is what that click runs.
        MatchmakerTicketer AQTicketer()
        {
            try { return UnityEngine.Object.FindObjectOfType<MatchmakerTicketer>(); } catch { return null; }
        }
        bool AQFindMatch()
        {
            var t = AQTicketer();
            if (t == null) return false;
            try { t.OnFindMatch(); MelonLogger.Msg("[AQ] MatchmakerTicketer.OnFindMatch() (direct, no mouse)"); return true; }
            catch (Exception ex) { MelonLogger.Msg("[AQ] OnFindMatch failed: " + ex.GetType().Name + ": " + ex.Message); return false; }
        }
        /// What the ranked screen is telling the player right now (the ticketer's InfoText), "" if none.
        string AQTicketerInfo()
        {
            try
            {
                var t = AQTicketer();
                if (t == null) return "";
                string a = "", b = "";
                try { var it = t.InfoText; if (it != null) a = it.text ?? ""; } catch { }
                try { var ip = t.InfoPaneText; if (ip != null) b = ip.text ?? ""; } catch { }
                return (a + " | " + b).Trim(' ', '|');
            }
            catch { return ""; }
        }
        static bool AQTextSaysMatchFound(string info)
        {
            if (string.IsNullOrEmpty(info)) return false;
            string u = info.ToUpperInvariant();
            return u.Contains("FOUND") || u.Contains("CONNECTING") || u.Contains("MATCHED") || u.Contains("ALLOCAT");
        }
        static string AQMirrorState()
        {
            try
            {
                return "active=" + Il2CppMirror.NetworkClient.active + " connecting=" + Il2CppMirror.NetworkClient.isConnecting
                     + " connected=" + Il2CppMirror.NetworkClient.isConnected;
            }
            catch (Exception ex) { return "mirror state unreadable: " + ex.Message; }
        }
        static bool AQMirrorConnected()
        {
            try { return Il2CppMirror.NetworkClient.isConnected; } catch { return false; }
        }
        /// The game's Mirror client is wedged: it reports connected while nothing is connected, and every
        /// later match "connects" instantly to nothing (2026-11-03 03:10: Telepathy receive-thread NRE on a
        /// retried connect, then 9 found matches in a row lost). Only a process restart clears it. Guarded
        /// so a wrong guess costs one restart per 10 min, not a loop.
        void AQRestartForWedge(string why)
        {
            try
            {
                if (File.Exists(AQ_WEDGE_MARK))
                {
                    var last = DateTime.Parse(File.ReadAllText(AQ_WEDGE_MARK).Trim(), null, DateTimeStyles.RoundtripKind);
                    if ((DateTime.UtcNow - last).TotalMinutes < 10)
                    {
                        MelonLogger.Msg("[AQ] wedge suspected (" + why + ") but the last wedge restart was "
                            + (int)(DateTime.UtcNow - last).TotalMinutes + " min ago; not restarting again yet");
                        return;
                    }
                }
                File.WriteAllText(AQ_WEDGE_MARK, DateTime.UtcNow.ToString("o"));
            }
            catch { }
            MelonLogger.Msg("[AQ] Mirror client wedged (" + why + "; " + AQMirrorState() + "): restarting the game");
            AQRequestRestart();
        }

        /// Cancel a running search and go back to the main screen (the ticketer's own ReturnMainMenu loads it).
        void AQCancelMatchmaking(string why)
        {
            var t = AQTicketer();
            MelonLogger.Msg("[AQ] cancelling matchmaking: " + why);
            if (t != null)
            {
                try { t.CancelActiveSearchAsync(true); } catch (Exception ex) { MelonLogger.Msg("[AQ]   cancel failed: " + ex.Message); }
                try { t.ReturnMainMenu(); MelonLogger.Msg("[AQ]   MatchmakerTicketer.ReturnMainMenu()"); return; } catch (Exception ex) { MelonLogger.Msg("[AQ]   ReturnMainMenu failed: " + ex.Message); }
            }
            AQGoToMenu();
        }

        bool AQClickGameObjectButton(string goName)
        {
            var go = GameObject.Find(goName);
            if (go == null) return false;
            var btn = go.GetComponent<Button>();
            if (btn == null) btn = go.GetComponentInChildren<Button>();
            if (btn == null) btn = go.GetComponentInParent<Button>();
            if (btn != null)
            {
                MelonLogger.Msg("[AQ] clicking GO button: " + go.name + " (Button on " + btn.gameObject.name + ")");
                btn.onClick.Invoke();
                return true;
            }
            // No Button component — simulate a real mouse click at screen position
            var rt = go.GetComponent<RectTransform>();
            if (rt == null) return false;
            APEnsureFocus();
            // For Screen Space - Overlay canvas, rt.position is already in screen pixels (0,0 = bottom-left)
            int cx = (int)rt.position.x;
            int cy = Screen.height - (int)rt.position.y;
            // Scale if Unity resolution differs from actual window client area
            if (apGameHwnd != IntPtr.Zero)
            {
                GetClientRect(apGameHwnd, out RECT cr);
                int clientW = cr.Right - cr.Left;
                int clientH = cr.Bottom - cr.Top;
                if (clientW > 0 && clientH > 0)
                {
                    cx = (int)(cx * (float)clientW / Screen.width);
                    cy = (int)(cy * (float)clientH / Screen.height);
                }
                // Convert client coords to absolute desktop coords
                var pt = new POINT { X = cx, Y = cy };
                ClientToScreen(apGameHwnd, ref pt);
                MelonLogger.Msg("[AQ] mouse-clicking '" + go.name + "' desktop=(" + pt.X + "," + pt.Y + ") client=(" + cx + "," + cy + ") unity=(" + (int)rt.position.x + "," + (int)rt.position.y + ") screen=" + Screen.width + "x" + Screen.height);
                SetCursorPos(pt.X, pt.Y);
            }
            else
            {
                MelonLogger.Msg("[AQ] mouse-clicking '" + go.name + "' raw=(" + cx + "," + cy + ") no hwnd");
                SetCursorPos(cx, cy);
            }
            System.Threading.Thread.Sleep(100);
            mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, IntPtr.Zero);
            System.Threading.Thread.Sleep(80);
            mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, IntPtr.Zero);
            return true;
        }

        bool AQClickButtonByText(string textContains)
        {
            string upper = textContains.ToUpperInvariant();
            foreach (var btn in UnityEngine.Object.FindObjectsOfType<Button>())
            {
                if (!btn.gameObject.activeInHierarchy || !btn.interactable) continue;
                try
                {
                    var tmpLabels = btn.GetComponentsInChildren<TMP_Text>();
                    for (int i = 0; i < tmpLabels.Length; i++)
                    {
                        if (tmpLabels[i] != null && tmpLabels[i].text.ToUpperInvariant().Contains(upper))
                        {
                            MelonLogger.Msg("[AQ] clicking button by TMP text: '" + tmpLabels[i].text + "' on " + btn.gameObject.name);
                            btn.onClick.Invoke();
                            return true;
                        }
                    }
                }
                catch { }
                try
                {
                    var labels = btn.GetComponentsInChildren<UnityEngine.UI.Text>();
                    for (int i = 0; i < labels.Length; i++)
                    {
                        if (labels[i] != null && labels[i].text.ToUpperInvariant().Contains(upper))
                        {
                            MelonLogger.Msg("[AQ] clicking button by text: '" + labels[i].text + "' on " + btn.gameObject.name);
                            btn.onClick.Invoke();
                            return true;
                        }
                    }
                }
                catch { }
                if (btn.gameObject.name.ToUpperInvariant().Contains(upper))
                {
                    MelonLogger.Msg("[AQ] clicking button by name: " + btn.gameObject.name);
                    btn.onClick.Invoke();
                    return true;
                }
            }
            return false;
        }

        void AQDumpButtons()
        {
            var buttons = UnityEngine.Object.FindObjectsOfType<Button>();
            MelonLogger.Msg("[AQ] === " + buttons.Length + " buttons in scene ===");
            for (int i = 0; i < buttons.Length; i++)
            {
                var b = buttons[i];
                MelonLogger.Msg("[AQ]   btn[" + i + "] name='" + b.gameObject.name
                    + "' active=" + b.gameObject.activeInHierarchy
                    + " interactable=" + b.interactable);
            }
        }

        void AQDumpAllButtons()
        {
            var buttons = UnityEngine.Object.FindObjectsOfType<Button>();
            MelonLogger.Msg("[AQ] === BUTTON DUMP: " + buttons.Length + " buttons ===");
            for (int i = 0; i < buttons.Length; i++)
            {
                var b = buttons[i];
                string label = "";
                try { var t = b.GetComponentInChildren<TMP_Text>(); if (t != null) label = t.text; } catch { }
                if (string.IsNullOrEmpty(label))
                    try { var t = b.GetComponentInChildren<UnityEngine.UI.Text>(); if (t != null) label = t.text; } catch { }
                MelonLogger.Msg("[AQ]   [" + i + "] name='" + b.gameObject.name + "' text='" + label
                    + "' active=" + b.gameObject.activeInHierarchy + " interact=" + b.interactable);
            }
            // Also dump ALL text in scene to find PLAY
            MelonLogger.Msg("[AQ] === ALL TMP TEXT IN SCENE ===");
            foreach (var t in UnityEngine.Object.FindObjectsOfType<TMP_Text>())
            {
                if (t == null || !t.gameObject.activeInHierarchy) continue;
                string txt = t.text;
                if (string.IsNullOrWhiteSpace(txt)) continue;
                MelonLogger.Msg("[AQ]   tmp='" + txt + "' on " + t.gameObject.name + " parent=" + (t.transform.parent != null ? t.transform.parent.gameObject.name : "none"));
            }
        }

        void AQGoToLobby()
        {
            // Try clicking "Enter the Bar" or similar button on main menu
            if (AQClickButton("enter") || AQClickButton("play") || AQClickButton("bar")
                || AQClickButton("lobby") || AQClickButton("start"))
            {
                return;
            }
            // Fallback: call LoadLobbyScene via reflection
            foreach (var mb in UnityEngine.Object.FindObjectsOfType<MonoBehaviour>())
            {
                try { AQCall(mb, "LoadLobbyScene"); MelonLogger.Msg("[AQ] LoadLobbyScene called"); return; }
                catch { }
            }
            MelonLogger.Msg("[AQ] could not find Enter button or LoadLobbyScene!");
        }

        void AQShowMenu()
        {
            foreach (var t in Resources.FindObjectsOfTypeAll<Transform>())
            {
                try
                {
                    if (t.name == "state_menu" && !t.gameObject.activeInHierarchy)
                        t.gameObject.SetActive(true);
                    else if (t.name == "substate_lOBBY" && t.gameObject.activeInHierarchy)
                        t.gameObject.SetActive(false);
                }
                catch { }
            }
        }

        void AQGoToMenu()
        {
            if (AQAtMainScene() && !AQHasVLC())
            {
                if (AQClickButton("btn_return")) MelonLogger.Msg("[AQ] clicked back (btn_return)");
                AQShowMenu();
                return;
            }
            try
            {
                var mgr = Manager.Instance;
                if (mgr != null) { MelonLogger.Msg("[AQ] Manager.ReturnMenu() (game scene)"); mgr.ReturnMenu(); return; }
            }
            catch (Exception ex) { MelonLogger.Msg("[AQ] Manager.ReturnMenu failed: " + ex.Message); }
            if (!AQAtMainScene())
            {
                AQForceMenu("scene '" + AQSceneName() + "' with no game Manager and no lobby");
                return;
            }
            foreach (var mb in UnityEngine.Object.FindObjectsOfType<MonoBehaviour>())
            {
                try { AQCall(mb, "ReturnMainMenu"); MelonLogger.Msg("[AQ] ReturnMainMenu called"); return; }
                catch { }
            }
            foreach (var mb in UnityEngine.Object.FindObjectsOfType<MonoBehaviour>())
            {
                try { AQCall(mb, "ReturnMenu"); MelonLogger.Msg("[AQ] ReturnMenu called"); return; }
                catch { }
            }
            AQShowMenu();
        }

        static string AQSceneName()
        {
            try { return SceneManager.GetActiveScene().name; } catch { return "?"; }
        }

        /// The same three steps Manager.ReturnMenu takes, for scenes that have no Manager (the MatchMaking
        /// scene after a lobby join that never connected): stop the Mirror client, leave the Steam lobby we
        /// asked to join, load the menu scene. Each step is independent; a failure in one is only logged.
        void AQForceMenu(string why)
        {
            MelonLogger.Msg("[AQ] forcing return to menu (" + why + ")");
            try
            {
                var nm = Il2CppMirror.NetworkManager.singleton;
                if (nm != null && (Il2CppMirror.NetworkClient.active || Il2CppMirror.NetworkClient.isConnecting))
                {
                    nm.StopClient();
                    MelonLogger.Msg("[AQ]   NetworkManager.StopClient()");
                }
            }
            catch (Exception ex) { MelonLogger.Msg("[AQ]   StopClient failed: " + ex.GetType().Name + ": " + ex.Message); }
            try
            {
                if (aqJoinedLobbyId.m_SteamID != 0)
                {
                    SteamMatchmaking.LeaveLobby(aqJoinedLobbyId);
                    MelonLogger.Msg("[AQ]   SteamMatchmaking.LeaveLobby(" + aqJoinedLobbyId.m_SteamID + ")");
                    aqJoinedLobbyId = new CSteamID(0);
                }
            }
            catch (Exception ex) { MelonLogger.Msg("[AQ]   LeaveLobby failed: " + ex.GetType().Name + ": " + ex.Message); }
            try
            {
                SceneManager.LoadScene("SteamTest");
                MelonLogger.Msg("[AQ]   SceneManager.LoadScene(SteamTest)");
            }
            catch (Exception ex) { MelonLogger.Msg("[AQ]   LoadScene failed: " + ex.GetType().Name + ": " + ex.Message); }
        }

        /// A game scene is up (the in-game Manager exists) but we are not seated in it with a DeckGameplay.
        bool AQInRunningGameScene()
        {
            try { return Manager.Instance != null && !AQHasVLC(); } catch { return false; }
        }

        void AQLeaveGameNow()
        {
            try
            {
                var mgr = Manager.Instance;
                if (mgr != null) { MelonLogger.Msg("[AQ] leaving game: Manager.ReturnMenu()"); mgr.ReturnMenu(); return; }
            }
            catch (Exception ex) { MelonLogger.Msg("[AQ] Manager.ReturnMenu failed: " + ex.GetType().Name + ": " + ex.Message); }
            AQGoToMenu();
        }

        void AQLeaveLobby()
        {
            foreach (var mb in UnityEngine.Object.FindObjectsOfType<MonoBehaviour>())
            {
                try { AQCall(mb, "LeaveLobbyScene"); MelonLogger.Msg("[AQ] LeaveLobbyScene called"); return; }
                catch { }
            }
            AQGoToMenu();
        }

        // 2026-11-07: the game shows "Connection quality check failed. Please try again." (and similar) as a modal over the
        // menu; nothing in the queue flow sees it, so the bot sat behind it. Scan the visible UI text in menu states, click the
        // popup's dismiss/retry button, restart the flow from the main menu. Never restarts the game from here (v1 looped).
        bool AQHandleConnectionPopup(float now)
        {
            if (now - aqPopupScanAt < 2f) return false;
            aqPopupScanAt = now;
            Transform hit = null; string hitText = null;
            try
            {
                foreach (var t in UnityEngine.Object.FindObjectsOfType<TMP_Text>())
                {
                    if (t == null || !t.gameObject.activeInHierarchy) continue;
                    string u = (t.text ?? "").ToUpperInvariant();
                    if (u.Contains("QUALITY CHECK FAILED") || u.Contains("PLEASE TRY AGAIN") || u.Contains("CONNECTION FAILED"))
                    { hit = t.transform; hitText = t.text; break; }
                }
                if (hit == null)
                    foreach (var t in UnityEngine.Object.FindObjectsOfType<UnityEngine.UI.Text>())
                    {
                        if (t == null || !t.gameObject.activeInHierarchy) continue;
                        string u = (t.text ?? "").ToUpperInvariant();
                        if (u.Contains("QUALITY CHECK FAILED") || u.Contains("PLEASE TRY AGAIN") || u.Contains("CONNECTION FAILED"))
                        { hit = t.transform; hitText = t.text; break; }
                    }
            }
            catch (Exception ex) { MelonLogger.Msg("[AQ] popup scan error: " + ex.Message); return false; }
            if (hit == null) return false;

            if (now - aqPopupFirstAt < 60f) return false;   // handled one less than a minute ago; let the flow run
            aqPopupFirstAt = now; aqPopupCount++;
            MelonLogger.Warning("[AQ] popup detected (#" + aqPopupCount + " this session): '" + hitText + "' on " + hit.gameObject.name
                + " in state " + aqState);
            // Click only a button that is clearly the popup's dismiss/retry button (2026-11-07 06:31: an unlabelled match was
            // the Devil help button). Search the popup's own subtree first, then the scene, by label or object name.
            bool clicked = false;
            string[] okWords = { "OK", "RETRY", "TRY AGAIN", "CLOSE", "CONFIRM", "YES", "TAMAM", "KAPAT" };
            Transform root = hit;
            for (int up = 0; up < 4 && root.parent != null; up++) root = root.parent;
            try
            {
                var btns = root.GetComponentsInChildren<Button>(false);
                for (int i = 0; i < btns.Length && !clicked; i++)
                {
                    var b = btns[i];
                    if (b == null || !b.gameObject.activeInHierarchy || !b.interactable) continue;
                    string label = "";
                    try { var tl = b.GetComponentInChildren<TMP_Text>(); if (tl != null) label = tl.text ?? ""; } catch { }
                    string lu = label.ToUpperInvariant(); string nu = b.gameObject.name.ToUpperInvariant();
                    bool ok = false;
                    foreach (var w in okWords) if (lu.Contains(w) || nu.Contains(w)) ok = true;
                    MelonLogger.Msg("[AQ] popup button candidate: '" + b.gameObject.name + "' text='" + label + "'" + (ok ? " -> clicking" : " (skipped)"));
                    if (!ok) continue;
                    b.onClick.Invoke();
                    clicked = true;
                }
            }
            catch (Exception ex) { MelonLogger.Msg("[AQ] popup button search error: " + ex.Message); }
            if (!clicked)
                foreach (var w in okWords) { if (AQClickButtonByText(w)) { clicked = true; break; } }
            if (!clicked)
            {
                MelonLogger.Msg("[AQ] popup has no recognisable button; dumping buttons and forcing the main menu");
                AQDumpAllButtons();
                AQForceMenu("connection popup without a recognisable button");
            }
            AQChangeState(AQState.WaitMainMenu);
            aqTimer = now + 4f;
            return true;
        }

        void AQChangeState(AQState s) { aqState = s; aqLastStateChange = Time.time; }

        void AutoQueueTick()
        {
            if (!autoQueue) return;
            float now = Time.time;

            // Watchdog: if stuck in any non-game state for too long, restart the game
            if (aqState != AQState.InGame && aqState != AQState.StuckWaiting && aqState != AQState.Idle
                && aqState != AQState.ClickPlay
                && now - aqLastStateChange > AQ_WATCHDOG)
            {
                MelonLogger.Warning("[AQ] WATCHDOG: stuck in " + aqState + " for " + AQ_WATCHDOG + "s, restarting game...");
                AQRequestRestart();
                return;
            }

            // Write heartbeat every tick
            AQHeartbeat(aqState.ToString());

            if (now < aqTimer) return;

            try
            {
                if (aqState != AQState.InGame && aqState != AQState.StuckWaiting && aqState != AQState.InLobby
                    && aqState != AQState.Idle && AQHandleConnectionPopup(now))
                    return;
                switch (aqState)
                {
                    case AQState.WaitMainMenu:
                    {
                        if (AQInGame())
                        {
                            MelonLogger.Msg("[AQ] already in game, switching to InGame state");
                            AQChangeState(AQState.InGame);
                            aqLastGameActivity = now;
                            break;
                        }
                        if (AQHasVLC()) { AQEnterLobbyState(now); break; }
                        if (!AQAtMainScene())
                        {
                            if (aqSceneStuckSince == 0f) aqSceneStuckSince = now;
                            if (now - aqSceneStuckSince > 20f)
                            {
                                AQForceMenu("WaitMainMenu has been in scene '" + AQSceneName() + "' for 20s");
                                aqSceneStuckSince = 0f;
                            }
                            aqTimer = now + 2f;
                            break;
                        }
                        aqSceneStuckSince = 0f;
                        // Try clicking "ENTER THE BAR"
                        if (AQClickButtonByText("ENTER THE BAR") || AQClickButton("enter"))
                        {
                            MelonLogger.Msg("[AQ] clicked Enter the Bar");
                            AQChangeState(AQState.ClickEnterBar);
                            aqTimer = now + 2f;
                        }
                        else
                        {
                            if (aqRetries == 0) AQDumpAllButtons();
                            aqRetries++;
                            aqTimer = now + 2f;
                        }
                        break;
                    }

                    case AQState.ClickEnterBar:
                    {
                        if (AQInGame())
                        {
                            AQChangeState(AQState.InGame);
                            break;
                        }
                        if (!AQRankedMode())
                        {
                            if (AQHasVLC()) { AQEnterLobbyState(now); break; }
                            if (AQLobbyListOpen())
                            {
                                MelonLogger.Msg("[AQ] lobby list already open (lobby mode, no ranked queue)");
                                AQChangeState(AQState.OpenLobbyList);
                                aqDeckFilterApplied = false;
                                aqLobbyListOpenedAt = now;
                                aqLobbyNextRefresh = now + 15f;
                                aqRetries = 0;
                                aqTimer = now + 2f;
                                break;
                            }
                            if (AQClickButtonByText("LOBBY LIST") || AQClickButton("loadgame"))
                            {
                                MelonLogger.Msg("[AQ] clicked Lobby List (lobby mode, no ranked queue)");
                                AQChangeState(AQState.OpenLobbyList);
                                aqDeckFilterApplied = false;
                                aqLobbyListOpenedAt = now;
                                aqLobbyNextRefresh = now + 15f;
                                aqRetries = 0;
                                aqTimer = now + 3f;
                            }
                            else
                            {
                                aqRetries++;
                                if (aqRetries == 1) AQDumpAllButtons();
                                if (aqRetries > 10)
                                {
                                    MelonLogger.Msg("[AQ] can't find Lobby List button, going back...");
                                    AQGoToMenu();
                                    AQChangeState(AQState.WaitMainMenu);
                                    aqTimer = now + 3f;
                                    aqRetries = 0;
                                }
                                else aqTimer = now + 1f;
                            }
                            break;
                        }
                        if (AQClickButtonByText("MATCHMAKING") || AQClickButton("matchmak"))
                        {
                            MelonLogger.Msg("[AQ] clicked Matchmaking, waiting for connection...");
                            AQChangeState(AQState.ClickMatchmaking);
                            aqRetries = 0;
                            aqTimer = now + 15f;
                        }
                        else
                        {
                            aqRetries++;
                            if (aqRetries > 10)
                            {
                                MelonLogger.Msg("[AQ] can't find Matchmaking button, going back...");
                                AQGoToMenu();
                                AQChangeState(AQState.WaitMainMenu);
                                aqTimer = now + 3f;
                                aqRetries = 0;
                            }
                            else
                            {
                                aqTimer = now + 1f;
                            }
                        }
                        break;
                    }

                    case AQState.ClickMatchmaking:
                    {
                        if (AQInGame())
                        {
                            AQChangeState(AQState.InGame);
                            break;
                        }
                        aqConnectedAtSearch = AQMirrorConnected();
                        MelonLogger.Msg("[AQ] mirror before search: " + AQMirrorState());
                        if (aqConnectedAtSearch && !AQHasVLC() && !AQInGame())
                        {
                            AQRestartForWedge("client says connected at the main menu with no lobby and no game");
                            break;
                        }
                        aqMatchFoundAt = 0f;
                        if (AQFindMatch() || AQClickButtonByText("PLAY") || AQClickButton("play") || AQClickGameObjectButton("Find Match Button"))
                        {
                            MelonLogger.Msg("[AQ] searching for a match, waiting...");
                            AQChangeState(AQState.ClickPlay);
                            aqTimer = now + 2f;
                            aqLobbyJoinedAt = now;
                            aqRetries = 0;
                        }
                        else
                        {
                            aqRetries++;
                            if (aqRetries == 1 || aqRetries == 10 || aqRetries == 30) AQDumpAllButtons();
                            if (aqRetries > 90)
                            {
                                MelonLogger.Msg("[AQ] can't find Play button after 90s, restarting...");
                                AQRequestRestart();
                            }
                            else
                            {
                                aqTimer = now + 1f;
                            }
                        }
                        break;
                    }

                    case AQState.ClickPlay:
                    {
                        if (AQInGame())
                        {
                            MelonLogger.Msg("[AQ] game started via matchmaking!");
                            AQChangeState(AQState.InGame);
                            apSelectIdx = 0; apCursorReset = true;
                            aqRetries = 0;
                            aqLastGameActivity = now;
                            break;
                        }
                        // Match found: the game drops us in a lobby that fills and starts on its own.
                        if (AQHasVLC())
                        {
                            MelonLogger.Msg("[AQ] match found, in the fill lobby; waiting for it to start");
                            AQEnterLobbyState(now);
                            break;
                        }
                        // Match found on screen (or the client went "connected") but no lobby ever comes: the
                        // Mirror client is wedged, and nothing short of a restart fixes it.
                        {
                            string info = AQTicketerInfo();
                            bool found = AQTextSaysMatchFound(info) || (!aqConnectedAtSearch && AQMirrorConnected());
                            if (found && aqMatchFoundAt == 0f)
                            {
                                aqMatchFoundAt = now;
                                MelonLogger.Msg("[AQ] ranked screen: '" + info + "' (" + AQMirrorState() + "), waiting for the lobby");
                            }
                            else if (found && now - aqMatchFoundAt > 60f)
                            {
                                AQRestartForWedge("match found 60s ago, still no lobby; screen: '" + info + "'");
                                aqMatchFoundAt = now;  // if the guard refused, try again in another 60 s
                                break;
                            }
                            else if (!found && aqMatchFoundAt != 0f)
                            {
                                MelonLogger.Msg("[AQ] ranked screen no longer says match found ('" + info + "'), back to waiting");
                                aqMatchFoundAt = 0f;
                            }
                        }
                        // No match in 5 minutes: cancel the search, back to the main screen, start over.
                        if (now - aqLobbyJoinedAt > 300f)
                        {
                            AQCancelMatchmaking("no match in 5 min");
                            AQChangeState(AQState.WaitMainMenu);
                            aqTimer = now + 4f;
                        }
                        else
                        {
                            aqTimer = now + 1f;
                        }
                        break;
                    }

                    case AQState.OpenLobbyList:
                    {
                        if (AQInGame()) { AQChangeState(AQState.InGame); aqLastGameActivity = now; break; }
                        if (AQHasVLC()) { AQEnterLobbyState(now); break; }
                        if (!aqDeckFilterApplied) { aqDeckFilterApplied = AQSetDeckFilter(); aqTimer = now + 2f; break; }
                        if (TryJoinBestLobby())
                        {
                            AQChangeState(AQState.JoiningLobby);
                            aqTimer = now + 2f;
                            break;
                        }
                        if (now >= aqLobbyNextRefresh)
                        {
                            AQRefreshLobbyList();
                            aqLobbyNextRefresh = now + 15f;
                        }
                        if (now - aqLobbyListOpenedAt > 150f)
                        {
                            MelonLogger.Msg("[AQ] no joinable deck lobby for 150s, back to menu and retry");
                            AQGoToMenu();
                            AQChangeState(AQState.WaitMainMenu);
                            aqTimer = now + 3f;
                            break;
                        }
                        aqTimer = now + 3f;
                        break;
                    }

                    case AQState.JoiningLobby:
                    {
                        if (AQInGame()) { AQChangeState(AQState.InGame); aqLastGameActivity = now; break; }
                        if (AQHasVLC()) { AQEnterLobbyState(now); break; }
                        if (!AQAtMainScene() && now - aqLastStateChange > 12f && AQInRunningGameScene())
                        {
                            MelonLogger.Msg("[AQ] joined '" + aqCurrentLobby + "' after its game had started (spectating), leaving");
                            AQBlacklistLobby();
                            AQLeaveGameNow();
                            AQChangeState(AQState.WaitingForGame);
                            aqTimer = now + 4f;
                            break;
                        }
                        if (now - aqLastStateChange > 40f)
                        {
                            MelonLogger.Msg("[AQ] lobby join did not land in 40s (scene '" + AQSceneName() + "'), blacklisting and retrying");
                            AQBlacklistLobby();
                            AQGoToMenu();
                            AQChangeState(AQState.WaitMainMenu);
                            aqTimer = now + 3f;
                            break;
                        }
                        aqTimer = now + 1f;
                        break;
                    }

                    case AQState.InLobby:
                    {
                        if (AQInGame())
                        {
                            MelonLogger.Msg("[AQ] lobby game started!");
                            AQChangeState(AQState.InGame);
                            apSelectIdx = 0; apCursorReset = true;
                            aqRetries = 0;
                            aqLastGameActivity = now;
                            break;
                        }
                        if (!AQHasVLC())
                        {
                            // Kicked, host left, or scene changing. Give the scene a moment, then re-evaluate.
                            if (now - aqLastStateChange > 8f)
                            {
                                MelonLogger.Msg("[AQ] lobby gone, restarting flow");
                                AQChangeState(AQState.WaitingForGame);
                                aqTimer = now + 2f;
                            }
                            else aqTimer = now + 1f;
                            break;
                        }
                        if (AQRankedMode())
                        {
                            // Matchmaking fill lobby: fixed mode, starts by itself when full. Only bail if it never does.
                            if (now - aqLobbyEnteredAt > 480f)
                            {
                                MelonLogger.Msg("[AQ] fill lobby has not started in 8 min, leaving and searching again");
                                AQLeaveCurrentLobby();
                                AQChangeState(AQState.WaitMainMenu);
                                aqTimer = now + 4f;
                                break;
                            }
                            if (!aqLobbyModeLogged)
                            {
                                aqLobbyModeLogged = true;
                                MelonLogger.Msg("[AQ] ranked fill lobby: " + AQLobbyPlayerCount() + " players so far, waiting (no checks, no ready)");
                            }
                            aqTimer = now + 2f;
                            break;
                        }
                        string why;
                        if (AQLobbyIsWrongMode(out why))
                        {
                            MelonLogger.Msg("[AQ] leaving lobby: " + why);
                            AQBlacklistLobby();
                            AQLeaveCurrentLobby();
                            AQChangeState(AQState.WaitingForGame);
                            aqTimer = now + 4f;
                            break;
                        }
                        // Deck lobby but not the Devil deck: give the synced value a few seconds, then leave.
                        var lref = AQLobby();
                        bool wantReady = lref != null && lref.IsDevil;
                        if (lref != null && !lref.IsDevil)
                        {
                            if (aqDeckWrongSince == 0f) aqDeckWrongSince = now;
                            if (now - aqDeckWrongSince > AQ_DECK_GRACE)
                            {
                                MelonLogger.Msg("[AQ] leaving lobby: deck mode " + lref.DeckName + " (want Devil)");
                                AQBlacklistLobby();
                                AQLeaveCurrentLobby();
                                AQChangeState(AQState.WaitingForGame);
                                aqTimer = now + 4f;
                                break;
                            }
                        }
                        else aqDeckWrongSince = 0f;
                        int players = AQLobbyPlayerCount();
                        if (players <= 1) { if (aqLobbyAloneSince == 0f) aqLobbyAloneSince = now; }
                        else aqLobbyAloneSince = 0f;
                        if (aqLobbyAloneSince > 0f && now - aqLobbyAloneSince > 180f)
                        {
                            MelonLogger.Msg("[AQ] alone in lobby for 3 min, leaving");
                            AQBlacklistLobby();
                            AQLeaveCurrentLobby();
                            AQChangeState(AQState.WaitingForGame);
                            aqTimer = now + 4f;
                            break;
                        }
                        if (now - aqLobbyEnteredAt > 480f)
                        {
                            MelonLogger.Msg("[AQ] lobby has not started in 8 min, leaving");
                            AQBlacklistLobby();
                            AQLeaveCurrentLobby();
                            AQChangeState(AQState.WaitingForGame);
                            aqTimer = now + 4f;
                            break;
                        }
                        if (now >= aqLobbyNextReady)
                        {
                            AQEnsureReady(now, wantReady);
                            aqLobbyNextReady = now + 1f;
                        }
                        aqTimer = now + 1f;
                        break;
                    }

                    case AQState.WaitingForGame:
                    {
                        if (AQInGame())
                        {
                            MelonLogger.Msg("[AQ] game started!");
                            AQChangeState(AQState.InGame);
                            apSelectIdx = 0; apCursorReset = true;
                            aqRetries = 0;
                            aqLastGameActivity = now;
                            break;
                        }
                        if (!AQRankedMode())
                        {
                            // Lobby mode: after a game the room comes back; otherwise rebuild from the menu.
                            if (AQHasVLC()) { AQEnterLobbyState(now); break; }
                            if (AQAtMainScene())
                            {
                                MelonLogger.Msg("[AQ] back at main menu, restarting lobby flow...");
                                AQChangeState(AQState.WaitMainMenu);
                                aqTimer = now + 2f;
                                break;
                            }
                            if (now - aqLastStateChange > 120f)
                            {
                                MelonLogger.Msg("[AQ] neither lobby nor menu for 2 minutes, going to menu");
                                AQGoToMenu();
                                AQChangeState(AQState.WaitMainMenu);
                                aqTimer = now + 3f;
                            }
                            else aqTimer = now + 1f;
                            break;
                        }
                        // Try Play button again (might be back at play screen after kick)
                        if (AQFindMatch() || AQClickButtonByText("PLAY") || AQClickButton("play") || AQClickGameObjectButton("Find Match Button"))
                        {
                            MelonLogger.Msg("[AQ] searching again");
                            AQChangeState(AQState.ClickPlay);
                            aqLobbyJoinedAt = now;
                            aqTimer = now + 2f;
                        }
                        // Or we're back at main menu
                        else if (AQAtMainScene())
                        {
                            MelonLogger.Msg("[AQ] back at main menu, restarting flow...");
                            AQChangeState(AQState.WaitMainMenu);
                            aqTimer = now + 2f;
                        }
                        else
                        {
                            // Might be on loading/score screen — give it plenty of time
                            if (now - aqLastStateChange > 120f)
                            {
                                MelonLogger.Msg("[AQ] stuck for 2 minutes, restarting game...");
                                AQRequestRestart();
                            }
                            else
                            {
                                aqTimer = now + 1f;
                            }
                        }
                        break;
                    }

                    case AQState.InGame:
                    {
                        // Dead? In ranked matchmaking wait for the game to end naturally (leaving costs rating).
                        // In lobby mode there is nothing to gain by spectating, so leave and find the next lobby.
                        if (AQIsDead())  // 2026-11-03: leave on death in ranked too; games per hour matter more than rating
                        {
                            if (aqDeadSince == 0f)
                            {
                                aqDeadSince = now;
                                MelonLogger.Msg("[AQ] we died, leaving the game in " + AQ_DEAD_LEAVE_DELAY + "s");
                            }
                            else if (now - aqDeadSince > AQ_DEAD_LEAVE_DELAY)
                            {
                                aqDeadSince = 0f;
                                AQLeaveGameNow();
                                AQChangeState(AQState.WaitingForGame);
                                aqTimer = now + 3f;
                                break;
                            }
                        }
                        else aqDeadSince = 0f;

                        AutoPlayTick(now);
                        APKeyboardTick(now);

                        // 2026-11-06: a dropped server connection (Mirror OnClientDisconnect, "server baglantisi
                        // koptu") is terminal in ranked; the loss is booked by the server at the drop and the
                        // 120 s frozen-table wait only cost two idle minutes per incident. Leave 5 s after the
                        // flag goes false and stays false.
                        if (!IsConnected())
                        {
                            if (aqDisconnectedSince == 0f)
                            {
                                aqDisconnectedSince = now;
                                MelonLogger.Msg("[AQ] connection to the match server lost; leaving in 5s if it stays down");
                            }
                            else if (now - aqDisconnectedSince > 5f)
                            {
                                aqDisconnectedSince = 0f;
                                MelonLogger.Msg("[AQ] connection still down, leaving the dead game now");
                                AQLeaveGameNow();
                                AQChangeState(AQState.WaitingForGame);
                                aqTimer = now + 3f;
                                break;
                            }
                        }
                        else aqDisconnectedSince = 0f;

                        // Game ended
                        if (!AQInGame())
                        {
                            MelonLogger.Msg("[AQ] game ended, looking for Play button...");
                            AQChangeState(AQState.WaitingForGame);
                            aqTimer = now + 2f;
                            break;
                        }

                        // Stuck detection: no game activity for 120s = broken state
                        if (aqLastGameActivity > 0 && now - aqLastGameActivity > 120f)
                        {
                            // 2026-11-03 10:29: left a ranked game 0 pulls vs 5 because the opponent froze on their turn.
                            // The server times stuck players out by itself; while we are still connected in ranked, wait.
                            if (!(AQRankedMode() && IsConnected()))
                            {
                                MelonLogger.Msg("[AQ] table frozen (no play for 120s, connected=" + IsConnected()
                                    + "), leaving the game (lobby mode)");
                                AQLeaveGameNow();
                                AQChangeState(AQState.WaitingForGame);
                                aqTimer = now + 3f;
                                break;
                            }
                            MelonLogger.Msg("[AQ] game appears stuck (no activity for 120s), waiting it out...");
                            AQChangeState(AQState.StuckWaiting);
                            aqTimer = now + 5f;
                        }
                        break;
                    }

                    case AQState.StuckWaiting:
                    {
                        AutoPlayTick(now);
                        APKeyboardTick(now);
                        // Game stuck — don't leave (lose rating). Wait for others to leave.
                        if (!AQInGame())
                        {
                            MelonLogger.Msg("[AQ] stuck game resolved, back to matchmaking...");
                            AQChangeState(AQState.WaitingForGame);
                            aqTimer = now + 2f;
                            break;
                        }
                        // If activity resumes, go back to InGame
                        if (aqLastGameActivity > 0 && now - aqLastGameActivity < 10f)
                        {
                            MelonLogger.Msg("[AQ] game activity resumed!");
                            AQChangeState(AQState.InGame);
                            break;
                        }
                        // Connection gone while waiting: nothing to wait for (same rule as InGame).
                        if (!IsConnected())
                        {
                            if (aqDisconnectedSince == 0f) aqDisconnectedSince = now;
                            else if (now - aqDisconnectedSince > 5f)
                            {
                                aqDisconnectedSince = 0f;
                                MelonLogger.Msg("[AQ] connection down while stuck-waiting, leaving the dead game now");
                                AQLeaveGameNow();
                                AQChangeState(AQState.WaitingForGame);
                                aqTimer = now + 3f;
                                break;
                            }
                        }
                        else aqDisconnectedSince = 0f;
                        // After 10 minutes of being stuck (20 in ranked, where leaving costs rating), give up and restart
                        float stuckCap = AQRankedMode() ? 1200f : 600f;
                        if (now - aqLastStateChange > stuckCap)
                        {
                            MelonLogger.Msg("[AQ] stuck for " + (int)(stuckCap / 60f) + " minutes, restarting...");
                            AQRequestRestart();
                            break;
                        }
                        AQHeartbeat("StuckWaiting");
                        aqTimer = now + 5f;
                        break;
                    }
                }
            }
            catch (Exception ex)
            {
                MelonLogger.Warning("[AQ] error in " + aqState + ": " + ex.Message);
                AQChangeState(AQState.WaitMainMenu);
                aqTimer = Time.time + 5f;
            }
        }

        int AQPlayerCount()
        {
            try { var ps = UnityEngine.Object.FindObjectsOfType<PlayerStats>(); return ps != null ? ps.Length : 0; } catch { return 0; }
        }

        void AQBlacklistLobby()
        {
            if (!string.IsNullOrEmpty(aqCurrentLobby) && aqCurrentLobby != "?")
            {
                aqBlacklist[aqCurrentLobby] = Time.time + AQ_BLACKLIST;
                MelonLogger.Msg("[AQ] blacklisted '" + aqCurrentLobby + "' for " + AQ_BLACKLIST + "s");
            }
            aqCurrentLobby = "";
        }

        // ── Lobby mode helpers ───────────────────────────────────────────────
        void AQEnterLobbyState(float now)
        {
            MelonLogger.Msg("[AQ] in lobby '" + aqCurrentLobby + "', waiting for the host (ready-up on)");
            AQChangeState(AQState.InLobby);
            aqLobbyEnteredAt = now;
            aqLobbyAloneSince = 0f;
            aqLobbyNextReady = now + 1f;
            aqLobbyModeLogged = false;
            aqDeckWrongSince = 0f;
            aqReadyClickedAt = 0f;
            aqReadyConfirmed = false;
            aqReadyWaitLogged = false;
            aqTimer = now + 1f;
        }

        /// Liar's Deck / Dice / Poker / Spin lobbies load the "SteamLobby" prefab, whose script is LobbyController;
        /// only Velvet Room lobbies use VelvetLobbyController. Both expose the same members, so wrap them.
        class LobbyRef
        {
            public LobbyController lc; public VelvetLobbyController vlc;
            public string ModeText
            {
                get
                {
                    try { var t = lc != null ? lc.gamemodetext : vlc.gamemodetext; return t != null ? (t.text ?? "") : ""; }
                    catch { return ""; }
                }
            }
            public string Mode
            {
                get
                {
                    try { return lc != null ? lc.NetworkMode.ToString() : vlc.NetworkMode.ToString(); } catch { }
                    try { return lc != null ? lc.Mode.ToString() : vlc.Mode.ToString(); } catch { return ""; }
                }
            }
            public string Deck
            {
                get { try { return lc != null ? lc.NetworkDeckMode.ToString() : vlc.NetworkDeckMode.ToString(); } catch { return ""; } }
            }
            public int PlayerCount { get { try { return lc != null ? lc.playercount : vlc.playercount; } catch { return 0; } } }
            public GameObject ReadyButton { get { try { return lc != null ? lc.ReadyButton : vlc.ReadyButton; } catch { return null; } } }
            public GameObject LeaveButton { get { try { return lc != null ? lc.leavebutton : vlc.leavebutton; } catch { return null; } } }
            public GameObject NotReadyButton { get { try { return lc != null ? lc.NotReadyButton : vlc.NotReadyButton; } catch { return null; } } }
            /// deckmode enum: 0 Basic, 1 Devil, 2 Deck2. Il2Cpp may print the number or the name.
            public string DeckName
            {
                get
                {
                    string d = Deck;
                    if (d == "0") return "Basic"; if (d == "1") return "Devil"; if (d == "2") return "Deck2";
                    return d;
                }
            }
            public bool IsDevil { get { string d = Deck; return d == "1" || d.IndexOf("Devil", StringComparison.OrdinalIgnoreCase) >= 0; } }
            public void SetReady() { if (lc != null) lc.setReady(); else vlc.setReady(); }
            public bool HasLocalPlayer
            {
                get { try { return lc != null ? lc.LocalPlayerController != null : vlc.LocalPlayerController != null; } catch { return false; } }
            }
            public bool LocalReady
            {
                get { try { return lc != null ? lc.LocalPlayerController.Ready : vlc.LocalPlayerController.Ready; } catch { return false; } }
            }
            /// PlayerObjectController.SetReady(bool) sends the SetReadyCMD command: sets ready explicitly (the lobby's
            /// own setReady() only toggles, so it would un-ready us if called twice).
            public void SetLocalReady(bool on) { if (lc != null) lc.LocalPlayerController.SetReady(on); else vlc.LocalPlayerController.SetReady(on); }
            public void Close() { if (lc != null) lc.ClooseLooby(); else vlc.ClooseLooby(); }
        }

        LobbyRef AQLobby()
        {
            var r = new LobbyRef();
            try { r.lc = UnityEngine.Object.FindObjectOfType<LobbyController>(); } catch { }
            if (r.lc == null) { try { r.vlc = UnityEngine.Object.FindObjectOfType<VelvetLobbyController>(); } catch { } }
            return (r.lc != null || r.vlc != null) ? r : null;
        }

        /// A lobby is wrong when its game is not Liar's Deck. The synced NetworkMode (GameMode enum: LiarsDeck,
        /// LiarsDice, LiarsChaos, LiarsPoker, VelvetRoom, LiarsTexas, LiarsSpin, Arena) is authoritative; the
        /// on-screen mode text is the fallback. Devil / Basic / Deck2 decks are all fine.
        bool AQLobbyIsWrongMode(out string why)
        {
            why = null;
            var l = AQLobby();
            if (l == null) return false;
            string modeText = l.ModeText, mode = l.Mode, deck = l.Deck;
            if (!aqLobbyModeLogged)
            {
                MelonLogger.Msg("[AQ] lobby mode text='" + modeText + "' mode=" + mode + " deck=" + deck
                    + " script=" + (l.lc != null ? "LobbyController" : "VelvetLobbyController"));
                aqLobbyModeLogged = true;
            }
            if (mode != "")
            {
                if (mode.Contains("Deck") || mode == "0") return false;
                why = "mode " + mode;
                return true;
            }
            string lo = modeText.ToLowerInvariant();
            if (lo.Contains("dice") || lo.Contains("poker") || lo.Contains("spin") || lo.Contains("slot")
                || lo.Contains("texas") || lo.Contains("velvet") || lo.Contains("arena"))
            {
                why = "mode text '" + modeText + "'";
                return true;
            }
            return false;
        }

        int AQLobbyPlayerCount()
        {
            var l = AQLobby();
            if (l != null) { int pc = l.PlayerCount; if (pc > 0) return pc; }
            return AQPlayerCount();
        }

        /// Keep the local player's synced Ready flag equal to wantReady. Uses PlayerObjectController.SetReady(bool)
        /// (explicit, idempotent) rather than the lobby's toggle. Re-sends every 2s until the synced flag agrees, and
        /// un-readies if the host switches the lobby to a deck we do not want.
        void AQEnsureReady(float now, bool wantReady)
        {
            var l = AQLobby();
            if (l == null) return;
            try
            {
                if (!l.HasLocalPlayer)
                {
                    if (!aqReadyWaitLogged) { MelonLogger.Msg("[AQ] waiting for local player object before ready-up"); aqReadyWaitLogged = true; }
                    return;
                }
                bool ready = l.LocalReady;
                if (ready == wantReady)
                {
                    if (ready && !aqReadyConfirmed) { MelonLogger.Msg("[AQ] ready confirmed (player.Ready=true)"); aqReadyConfirmed = true; }
                    if (!ready) aqReadyConfirmed = false;
                    return;
                }
                aqReadyConfirmed = false;
                if (aqReadyClickedAt > 0f && now - aqReadyClickedAt < 2f) return;   // give the command time to sync back
                MelonLogger.Msg(wantReady ? "[AQ] sending SetReady(true)" : "[AQ] un-readying: SetReady(false)");
                l.SetLocalReady(wantReady);
                aqReadyClickedAt = now;
            }
            catch (Exception ex) { MelonLogger.Msg("[AQ] ready failed: " + ex.GetType().Name + ": " + ex.Message); }
        }

        void AQLeaveCurrentLobby()
        {
            var l = AQLobby();
            try
            {
                var lb = l != null ? l.LeaveButton : null;
                if (lb != null)
                {
                    var btn = lb.GetComponent<Button>() ?? lb.GetComponentInChildren<Button>();
                    if (btn != null) { MelonLogger.Msg("[AQ] clicking Leave"); btn.onClick.Invoke(); return; }
                }
            }
            catch (Exception ex) { MelonLogger.Msg("[AQ] leave button failed: " + ex.Message); }
            try { if (l != null) { MelonLogger.Msg("[AQ] calling ClooseLooby()"); l.Close(); return; } } catch { }
            AQLeaveLobby();
        }

        /// True while the Lobby List panel is showing (its Refresh / Quick Join buttons are live).
        bool AQLobbyListOpen()
        {
            try
            {
                foreach (var btn in UnityEngine.Object.FindObjectsOfType<Button>())
                {
                    string n = btn.gameObject.name;
                    if ((n == "RefreshButton" || n == "QuickJoin") && btn.gameObject.activeInHierarchy) return true;
                }
            }
            catch { }
            return false;
        }

        /// Tick the Lobby List's DECK filter and clear DICE / POKER / SLOTS so the game itself hides other modes.
        /// The per-entry Steam "Mode" check in TryJoinBestLobby is the hard guarantee; this just tidies the list.
        bool AQSetDeckFilter()
        {
            try
            {
                var lm = UnityEngine.Object.FindObjectOfType<LobbyListManager>();
                if (lm == null) return false;
                bool changed = false;
                changed |= AQSetToggle(lm.DeckFilter, true, "DECK");
                changed |= AQSetToggle(lm.DiceFilter, false, "DICE");
                changed |= AQSetToggle(lm.PokerFilter, false, "POKER");
                changed |= AQSetToggle(lm.SpinFilter, false, "SLOTS");
                MelonLogger.Msg(changed ? "[AQ] lobby list filter set to DECK only" : "[AQ] lobby list filter already DECK only");
                return true;
            }
            catch (Exception ex) { MelonLogger.Msg("[AQ] filter toggles failed: " + ex.Message); return true; }
        }

        bool AQSetToggle(Toggle t, bool on, string label)
        {
            if (t == null) return false;
            if (t.isOn == on) return false;
            t.isOn = on;
            MelonLogger.Msg("[AQ]   filter " + label + " -> " + (on ? "on" : "off"));
            return true;
        }

        void AQRefreshLobbyList()
        {
            try
            {
                var steam = SteamLobby.Instance ?? UnityEngine.Object.FindObjectOfType<SteamLobby>();
                if (steam != null) { steam.GetLobbiesList(); MelonLogger.Msg("[AQ] refreshed lobby list"); return; }
            }
            catch (Exception ex) { MelonLogger.Msg("[AQ] GetLobbiesList failed: " + ex.Message); }
            if (AQClickButtonByText("LOBBY LIST") || AQClickButton("loadgame")) MelonLogger.Msg("[AQ] re-clicked Lobby List");
        }

        bool TryJoinBestLobby()
        {
            try
            {
                var entries = UnityEngine.Object.FindObjectsOfType<LobbyDataEntry>();
                MelonLogger.Msg("[AQ] found " + (entries != null ? entries.Length : 0) + " lobby entries");
                if (entries == null || entries.Length == 0) return false;

                float now = Time.time;
                LobbyDataEntry best = null;
                int bestCount = 0;
                string bestName = "?";

                for (int i = 0; i < entries.Length; i++)
                {
                    var entry = entries[i];
                    if (entry == null) continue;
                    string name = "?";
                    try { name = entry.LobbyName; } catch { }
                    if (aqBlacklist.TryGetValue(name, out float exp) && now < exp)
                    {
                        MelonLogger.Msg("[AQ]   skip '" + name + "' (blacklisted " + (int)(exp - now) + "s left)");
                        continue;
                    }
                    bool banned = false;
                    foreach (var b in aqPermaBan) { if (name.Contains(b)) { banned = true; break; } }
                    if (banned) { MelonLogger.Msg("[AQ]   skip '" + name + "' (permaban)"); continue; }
                    // Hosts publish GameMode as the "Mode" lobby value: 0 = LiarsDeck, 1 = LiarsDice, 5 = LiarsTexas
                    // (POKER), 6 = LiarsSpin (SLOTS). Only ever join Liar's Deck.
                    string lmode = "";
                    try { lmode = SteamMatchmaking.GetLobbyData(entry.lobbyID, "Mode") ?? ""; } catch (Exception ex) { lmode = "?" + ex.GetType().Name; }
                    if (lmode != "0")
                    {
                        MelonLogger.Msg("[AQ]   skip '" + name + "' (Mode='" + lmode + "', not Liar's Deck)");
                        continue;
                    }
                    // The list is refreshed every 15s; a host can start in between. Joining then lands us as a
                    // spectator in a running game, so re-check the lobby phase right before joining.
                    string phase = "";
                    try { phase = SteamMatchmaking.GetLobbyData(entry.lobbyID, "InGame") ?? ""; } catch { }
                    if (phase != "" && phase != "lobby")
                    {
                        MelonLogger.Msg("[AQ]   skip '" + name + "' (InGame='" + phase + "', already playing)");
                        continue;
                    }
                    int pc = entry.PlayerCount;
                    int maxPc = 4;
                    try { if (entry.MaxPlayerCount > 0) maxPc = entry.MaxPlayerCount; } catch { }
                    if (pc > bestCount && pc < maxPc)
                    {
                        bestCount = pc;
                        best = entry;
                        bestName = name;
                    }
                }

                if (best == null) return false;

                MelonLogger.Msg("[AQ] joining '" + bestName + "' (" + bestCount + " players, Mode=0 Liar's Deck)");
                aqCurrentLobby = bestName;
                try { aqJoinedLobbyId = best.lobbyID; } catch { }

                // The entry's own join is what its Join button runs; fall back to SteamLobby.
                try { best.JoinLobby(); return true; }
                catch (Exception ex) { MelonLogger.Msg("[AQ] entry.JoinLobby failed: " + ex.Message + ", trying SteamLobby"); }

                var steam = UnityEngine.Object.FindObjectOfType<SteamLobby>();
                if (steam != null)
                {
                    steam.JoinLobby(best.lobbyID);
                    return true;
                }
                return false;
            }
            catch (Exception ex)
            {
                MelonLogger.Warning("[AQ] join failed: " + ex.Message);
                return false;
            }
        }

        // ── Auto-play logic ──────────────────────────────────────────────────

        void AutoPlayTick(float now)
        {
            var mgr = Manager.Instance;
            if (mgr == null) return;
            var lp = mgr.GetLocalPlayer();
            if (lp == null) return;
            var go = lp.gameObject;

            var dg = go.GetComponent<DeckGameplay>();
            if (dg == null) return;

            var ps = go.GetComponent<PlayerStats>();
            if (ps == null || ps.NetworkDead) return;

            if (!apMethodsSearched) { APFindMethods(); apMethodsSearched = true; }

            if (!ps.NetworkHaveTurn)
            {
                apActedThisTurn = false;
                apTurnDelay = 0f;
                apNoCardsLogged = false;
                return;
            }

            if (apActedThisTurn) return;

            // 2026-11-06 (game_20261106_135014 round 8): when a dead player left mid-round the server re-raised our
            // HaveTurn flag and left it stale while the next player held the real turn; the bot threw a second card
            // out of turn and the server accepted it. Never act while another live seat holds the turn, or while
            // we are still the last to have thrown (someone else must play before it can be our turn again).
            string stale;
            if (APTurnIsStale(out stale))
            {
                apTurnDelay = 0f; apPendingLine = null;
                if (!apStaleLogged) { apStaleLogged = true; MelonLogger.Msg("[AP] HaveTurn set but not our turn (" + stale + "), ignoring"); }
                return;
            }
            apStaleLogged = false;

            if (apTurnDelay == 0f)
            {
                apTurnStart = now;
                apTurnDelay = now + 1.2f + UnityEngine.Random.Range(0f, 0.8f);
                apPendingLine = null;
                return;
            }
            if (now < apTurnDelay) return;

            // Read the advisor's line once. It may carry "delay=S": act S seconds after the turn
            // began (timing is part of the play), capped to stay under the game's turn timer.
            if (apPendingLine == null)
            {
                apPendingLine = "";
                try
                {
                    if (File.Exists(AP_ACTION))
                    {
                        string raw = File.ReadAllText(AP_ACTION).Trim();
                        File.Delete(AP_ACTION);
                        float delay = 0f;
                        var toks = new List<string>();
                        foreach (var t in raw.Split(' '))
                        {
                            if (t.StartsWith("delay=")) float.TryParse(t.Substring(6), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out delay);
                            else if (t.Length > 0) toks.Add(t);
                        }
                        apPendingLine = string.Join(" ", toks);
                        if (delay > 0f)
                        {
                            float fireAt = Math.Max(0f, Math.Min(delay, 24f) - AP_EXEC_LEAD);
                            apTurnDelay = apTurnStart + fireAt;
                            MelonLogger.Msg("[AP] advisor: '" + apPendingLine + "' after " + delay.ToString("0.0")
                                + "s (firing at " + fireAt.ToString("0.0") + "s, lead " + AP_EXEC_LEAD + "s)");
                            if (now < apTurnDelay) return;
                        }
                    }
                }
                catch { }
            }

            var cards = CardObjects(me);
            int visibleCards = 0;
            if (cards != null)
                for (int i = 0; i < cards.Count; i++)
                    if (cards[i] != null && cards[i].activeInHierarchy && cards[i].GetComponent<Card>() != null) visibleCards++;
            if (visibleCards == 0)
            {
                // No card rendered yet (deal animation late or lost). Do not give the turn up: retry every tick, and
                // after 20 s act anyway so the server does not drop us at the 30 s timer.
                if (!apNoCardsLogged)
                {
                    apNoCardsLogged = true;
                    string diag = "list=" + (cards == null ? "null" : cards.Count.ToString());
                    try
                    {
                        if (cards != null)
                        {
                            var parts = new List<string>();
                            for (int i = 0; i < cards.Count; i++)
                            {
                                var c = cards[i];
                                if (c == null) { parts.Add("null"); continue; }
                                var cc = c.GetComponent<Card>();
                                parts.Add((c.activeSelf ? "a" : "i") + (c.activeInHierarchy ? "H" : "h") + (cc != null ? ":t" + cc.cardtype : ":-"));
                            }
                            diag += " cards=[" + string.Join(",", parts) + "]";
                        }
                        diag += " haveCards=" + dg.NetworkHaveCards;
                        var mig = dg.GetMigrationCardTypes();
                        if (mig != null) { var ml = new List<string>(); for (int i = 0; i < mig.Count; i++) ml.Add(mig[i].ToString()); diag += " migTypes=[" + string.Join(",", ml) + "]"; }
                    }
                    catch (Exception ex) { diag += " diagErr=" + ex.GetType().Name; }
                    MelonLogger.Msg("[AP] my turn but no card is rendered (" + diag + "); waiting for the deal");
                }
                if (now - apTurnStart < 20f) return;
                apActedThisTurn = true;
                if (lastThrower >= 0)
                {
                    MelonLogger.Msg("[AP] still no cards after 20s: calling liar rather than being dropped at the timer");
                    APCallLiar(dg);
                    return;
                }
                MelonLogger.Msg("[AP] still no cards after 20s and nothing to call: trying the game's own auto-throw");
                try { dg.TryAutoThrowOnTimer(); } catch (Exception ex) { MelonLogger.Msg("[AP] TryAutoThrowOnTimer failed: " + ex.Message); }
                return;
            }
            if (apNoCardsLogged) MelonLogger.Msg("[AP] cards rendered " + (now - apTurnStart).ToString("0.0") + "s into the turn, acting");

            apActedThisTurn = true;

            if (cards == null) return;
            int table = TableCard();
            if (table <= 0) return;

            var idx = new List<int>();
            var types = new List<int>();
            int vis = 0;
            for (int i = 0; i < cards.Count; i++)
            {
                var c = cards[i];
                if (c == null || !c.activeInHierarchy) continue;
                var cc = c.GetComponent<Card>();
                if (cc != null) { idx.Add(vis); types.Add(cc.cardtype); vis++; }
            }
            if (idx.Count == 0) return;

            int wantReal = 1;
            int wantFake = 0;
            bool doCall = false;
            try
            {
                if (!string.IsNullOrEmpty(apPendingLine))
                {
                    string line = apPendingLine;
                    if (line.StartsWith("call")) { doCall = true; MelonLogger.Msg("[AP] advisor: call liar"); }
                    else if (line == "play devil") { wantReal = 0; wantFake = 0; MelonLogger.Msg("[AP] advisor: play devil bait"); }
                    else if (line.StartsWith("play "))
                    {
                        var parts = line.Substring(5).Trim().Split(' ');
                        if (parts.Length >= 2)
                        {
                            int.TryParse(parts[0], out wantReal);
                            int.TryParse(parts[1], out wantFake);
                        }
                        else if (parts.Length == 1)
                        {
                            int.TryParse(parts[0], out wantReal);
                            wantFake = 0;
                        }
                        if (wantReal + wantFake < 1) { wantReal = 1; wantFake = 0; }
                        MelonLogger.Msg("[AP] advisor: play real=" + wantReal + " fake=" + wantFake);
                    }
                }
            }
            catch { }

            if (doCall)
            {
                if (lastThrower >= 0)
                {
                    APCallLiar(dg);
                    return;
                }
                MelonLogger.Msg("[AP] advisor said call but lastThrower=-1, skipping turn");
                return;
            }

            var exactMatch = new List<int>();
            var jokers = new List<int>();
            var normalNon = new List<int>();
            var devils = new List<int>();
            for (int i = 0; i < types.Count; i++)
            {
                if (types[i] == table)
                    exactMatch.Add(idx[i]);
                else if (types[i] == 4)
                    jokers.Add(idx[i]);
                else if (types[i] == -1)
                    devils.Add(idx[i]);
                else
                    normalNon.Add(idx[i]);
            }
            // Which hand slots a player throws from is visible to the whole table. Always taking the
            // leftmost card of the kind wanted made bluffs come from slot 0 half the time and truths a
            // quarter of the time (2026-11-01 logs), so pick among equal cards at random.
            var apRng = new System.Random();
            void Shuffle(List<int> l) { for (int i = l.Count - 1; i > 0; i--) { int j = apRng.Next(i + 1); (l[i], l[j]) = (l[j], l[i]); } }
            Shuffle(exactMatch); Shuffle(jokers); Shuffle(normalNon);
            var matching = new List<int>(exactMatch);
            matching.AddRange(jokers);
            var nonMatching = new List<int>(normalNon);
            nonMatching.AddRange(devils);

            // Devil bait: play the Devil card alone
            if (wantReal == 0 && wantFake == 0 && devils.Count > 0)
            {
                MelonLogger.Msg("[AP] throwing Devil bait, table=" + table
                    + " hand=" + string.Join(",", MyHand()));
                APThrowCards(dg, new List<int> { devils[0] }, cards);
                return;
            }

            int realCount = Math.Min(wantReal, matching.Count);
            int fakeCount = Math.Min(wantFake, nonMatching.Count);
            int total = realCount + fakeCount;
            if (total < 1) { realCount = 1; total = 1; }
            total = Math.Min(total, 3);

            var throwIdx = new List<int>();
            for (int i = 0; i < exactMatch.Count && throwIdx.Count < realCount; i++)
                throwIdx.Add(exactMatch[i]);
            for (int i = 0; i < jokers.Count && throwIdx.Count < realCount; i++)
                throwIdx.Add(jokers[i]);
            for (int i = 0; i < normalNon.Count && throwIdx.Count < total; i++)
                throwIdx.Add(normalNon[i]);
            for (int i = 0; i < devils.Count && throwIdx.Count < total; i++)
                throwIdx.Add(devils[i]);
            for (int i = 0; i < matching.Count && throwIdx.Count < total; i++)
                if (!throwIdx.Contains(matching[i])) throwIdx.Add(matching[i]);

            MelonLogger.Msg("[AP] throwing " + throwIdx.Count + ", table=" + table
                + " hand=" + string.Join(",", MyHand()) + " match=" + matching.Count);
            APThrowCards(dg, throwIdx, cards);
        }

        void APFindMethods()
        {
            var flags = System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;

            // Prefer CmdThrowCards (Mirror Cmd) > RequestThrowCards (game API)
            // Never use UserCode_ methods — those are server-side handlers
            string[] throwNames = { "CmdThrowCards", "RequestThrowCards" };
            string[] callNames = { "RequestCallLiar", "CmdCallLiar" };

            foreach (var n in throwNames)
            {
                apThrowMethod = typeof(DeckGameplay).GetMethod(n, flags);
                if (apThrowMethod != null) break;
            }
            foreach (var n in callNames)
            {
                apCallMethod = typeof(DeckGameplay).GetMethod(n, flags);
                if (apCallMethod != null) break;
            }

            MelonLogger.Msg("[AP] throw: " + (apThrowMethod != null ? apThrowMethod.Name : "NOT FOUND"));
            MelonLogger.Msg("[AP] call:  " + (apCallMethod != null ? apCallMethod.Name : "NOT FOUND"));
        }

        [DllImport("user32.dll")] static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, IntPtr dwExtraInfo);
        [DllImport("user32.dll")] static extern bool SetForegroundWindow(IntPtr hWnd);
        [DllImport("user32.dll")] static extern bool PostMessageA(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);
        const uint WM_KEYDOWN = 0x0100, WM_KEYUP = 0x0101;
        [DllImport("user32.dll")] static extern bool SetCursorPos(int X, int Y);
        [DllImport("user32.dll")] static extern void mouse_event(uint dwFlags, int dx, int dy, uint dwData, IntPtr dwExtraInfo);
        [DllImport("user32.dll")] static extern bool ClientToScreen(IntPtr hWnd, ref POINT lpPoint);
        [DllImport("user32.dll")] static extern bool GetClientRect(IntPtr hWnd, out RECT lpRect);
        [DllImport("user32.dll", CharSet = CharSet.Auto)] static extern int GetClassName(IntPtr hWnd, System.Text.StringBuilder lpClassName, int nMaxCount);
        [StructLayout(LayoutKind.Sequential)] struct POINT { public int X, Y; }
        [StructLayout(LayoutKind.Sequential)] struct RECT { public int Left, Top, Right, Bottom; }
        const uint MOUSEEVENTF_LEFTDOWN = 0x0002, MOUSEEVENTF_LEFTUP = 0x0004;
        [DllImport("kernel32.dll")] static extern uint GetCurrentProcessId();
        [DllImport("user32.dll")] static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
        [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);
        [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr hWnd);
        delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
        const byte VK_A = 0x41, VK_D = 0x44, VK_E = 0x45, VK_X = 0x58, VK_SPACE = 0x20;

        IntPtr apGameHwnd = IntPtr.Zero;
        void APEnsureHwnd()
        {
            if (apGameHwnd == IntPtr.Zero)
            {
                uint pid = GetCurrentProcessId();
                EnumWindows((hWnd, _) =>
                {
                    GetWindowThreadProcessId(hWnd, out uint wp);
                    if (wp != pid || !IsWindowVisible(hWnd)) return true;
                    var sb = new System.Text.StringBuilder(256);
                    GetClassName(hWnd, sb, 256);
                    if (sb.ToString() == "UnityWndClass") { apGameHwnd = hWnd; return false; }
                    return true;
                }, IntPtr.Zero);
            }
        }
        void APEnsureFocus()
        {
            APEnsureHwnd();
            if (apGameHwnd != IntPtr.Zero) SetForegroundWindow(apGameHwnd);
        }

        void APPressKey(byte vk, int holdMs = 50)
        {
            APEnsureFocus();
            keybd_event(vk, 0, 0, IntPtr.Zero);
            System.Threading.Thread.Sleep(holdMs);
            keybd_event(vk, 0, 2, IntPtr.Zero);
        }

        int apSelectIdx = 0;
        int apThrowPhase = 0; // 0=idle, 1=selecting, 2=throwing
        List<int> apThrowTargets;
        int apThrowNext;
        int apLastHandSize;
        bool apCursorReset = true;
        float apNextKeyAt;

        // ── Direct input (no keyboard, no window focus) ──────────────────────
        // The game's A/D/Space/E handlers (HandleCardNavigation / HandleSelectToggle /
        // HandleTurnActionsInput, decoded 2026-11-02) just flip Card.Selected on Cards[currentCard],
        // keep the same object in selectedCards, and on E call ThrowCards(), which sends
        // RequestThrowCards -> CmdThrowCards to the host. X calls RequestCallLiar(). Doing those
        // steps ourselves produces the same network traffic without Input.GetKeyDown, so the window
        // can sit unfocused and minimised. Creating C:\liars\ap_keyboard.txt forces the old keypress
        // path; a direct throw that has not shrunk the hand after AP_DIRECT_TIMEOUT also falls back.
        const string AP_KEYBOARD_FLAG = @"C:\liars\ap_keyboard.txt";
        const float AP_DIRECT_TIMEOUT = 2.0f;
        bool apDirectPending;
        float apDirectDeadline;
        int apDirectExpectHand;
        DeckGameplay apDirectDg;
        List<int> apDirectIndices;
        Il2CppSystem.Collections.Generic.List<GameObject> apDirectCards;
        int apDirectOk, apDirectFallbacks;

        bool APUseKeyboard() { try { return File.Exists(AP_KEYBOARD_FLAG); } catch { return false; } }

        /// Map "visible" indices (position among active cards, what the keyboard cursor walks) to
        /// real indices into dg.Cards.
        static List<int> APRealIndices(Il2CppSystem.Collections.Generic.List<GameObject> cards, List<int> visible)
        {
            var map = new List<int>();
            for (int i = 0; i < cards.Count; i++)
            {
                var c = cards[i];
                if (c == null || !c.activeInHierarchy) continue;
                if (c.GetComponent<Card>() == null) continue;
                map.Add(i);
            }
            var real = new List<int>();
            foreach (var v in visible) if (v >= 0 && v < map.Count) real.Add(map[v]);
            return real;
        }

        void APClearSelection(DeckGameplay dg, Il2CppSystem.Collections.Generic.List<GameObject> cards)
        {
            try
            {
                for (int i = 0; i < cards.Count; i++)
                {
                    var c = cards[i];
                    if (c == null) continue;
                    var cc = c.GetComponent<Card>();
                    if (cc != null && cc.Selected) cc.Selected = false;
                }
                if (dg.selectedCards != null) dg.selectedCards.Clear();
            }
            catch (Exception ex) { MelonLogger.Msg("[AP] clear selection failed: " + ex.Message); }
        }

        /// Select the cards the way Space does, then throw the way E does. Returns false if anything
        /// threw, in which case the selection is cleared and nothing was sent.
        bool APThrowDirect(DeckGameplay dg, List<int> indices, Il2CppSystem.Collections.Generic.List<GameObject> cards)
        {
            var real = APRealIndices(cards, indices);
            if (real.Count != indices.Count || real.Count == 0)
            {
                MelonLogger.Msg("[AP] direct: could not map indices " + string.Join(",", indices) + " onto hand");
                return false;
            }
            try
            {
                if (dg.throwed) { MelonLogger.Msg("[AP] direct: throwed flag still set, not throwing"); return false; }
                APClearSelection(dg, cards);
                var picked = new List<string>();
                foreach (var ri in real)
                {
                    var go = cards[ri];
                    var cc = go.GetComponent<Card>();
                    cc.Selected = true;
                    if (dg.selectedCards != null && !dg.selectedCards.Contains(go)) dg.selectedCards.Add(go);
                    picked.Add(ri + ":t" + cc.cardtype);
                }
                dg.currentCard = real[real.Count - 1];
                MelonLogger.Msg("[AP] direct: selected [" + string.Join(" ", picked) + "] -> ThrowCards()");
                dg.ThrowCards();
                return true;
            }
            catch (Exception ex)
            {
                MelonLogger.Msg("[AP] direct throw failed: " + ex.GetType().Name + ": " + ex.Message);
                APClearSelection(dg, cards);
                return false;
            }
        }

        /// After a direct throw: if the hand did not shrink in time, fall back to the keyboard path once.
        void APDirectTick(float now)
        {
            if (!apDirectPending) return;
            int hand = me != null ? HandCount(me) : -1;
            if (hand == apDirectExpectHand)
            {
                apDirectPending = false; apDirectOk++;
                MelonLogger.Msg("[AP] direct: throw confirmed, hand=" + hand + " (ok " + apDirectOk + ", fallbacks " + apDirectFallbacks + ")");
                return;
            }
            if (now < apDirectDeadline) return;
            apDirectPending = false; apDirectFallbacks++;
            MelonLogger.Msg("[AP] direct: hand still " + hand + " after " + AP_DIRECT_TIMEOUT + "s, falling back to keyboard");
            try { if (apDirectDg != null && apDirectCards != null) APClearSelection(apDirectDg, apDirectCards); } catch { }
            if (apDirectDg != null && apDirectIndices != null && apDirectCards != null)
                APThrowCardsKeyboard(apDirectDg, apDirectIndices, apDirectCards);
        }

        void APThrowCards(DeckGameplay dg, List<int> indices, Il2CppSystem.Collections.Generic.List<GameObject> cards)
        {
            if (APUseKeyboard())
            {
                APThrowCardsKeyboard(dg, indices, cards);
                return;
            }
            int before = HandCount(me);
            if (APThrowDirect(dg, indices, cards))
            {
                apDirectPending = true;
                apDirectDeadline = Time.time + AP_DIRECT_TIMEOUT;
                apDirectExpectHand = Math.Max(0, before - indices.Count);
                apDirectDg = dg; apDirectIndices = new List<int>(indices); apDirectCards = cards;
                return;
            }
            APThrowCardsKeyboard(dg, indices, cards);
        }

        void APThrowCardsKeyboard(DeckGameplay dg, List<int> indices, Il2CppSystem.Collections.Generic.List<GameObject> cards)
        {
            int handSize = 0;
            var handTypes = new List<string>();
            var posLog = new List<string>();
            for (int i = 0; i < cards.Count; i++)
            {
                var c = cards[i];
                if (c == null || !c.activeInHierarchy) continue;
                var cc = c.GetComponent<Card>();
                if (cc != null)
                {
                    handTypes.Add(cc.cardtype.ToString());
                    var p = c.transform.position;
                    posLog.Add(handSize + ":t" + cc.cardtype + " x=" + p.x.ToString("F2") + " z=" + p.z.ToString("F2"));
                    handSize++;
                }
            }
            MelonLogger.Msg("[AP] hand=[" + string.Join(",", handTypes) + "] target=[" + string.Join(",", indices) + "]");
            MelonLogger.Msg("[AP] positions: " + string.Join(" | ", posLog));

            apLastHandSize = handSize;
            apThrowTargets = new List<int>(indices);
            apThrowTargets.Sort();
            apThrowNext = 0;
            // The game puts the card cursor back on the first card every turn, so start from 0 on every
            // throw. Carrying the cursor over from the last throw (2026-10-31 22:36 build) threw the card
            // one to the left of the intended one on every second throw of a round (2026-11-01 01:06: asked
            // for a Q from K A Q Q, threw the A). The 16:32 build reset to 0 and threw right 276/276 times.
            apSelectIdx = 0; apCursorReset = false;
            MelonLogger.Msg("[AP] cursor at " + apSelectIdx + ", hand=" + handSize);
            apThrowPhase = 1;
            apNextKeyAt = Time.time + 0.2f;
        }

        void APKeyboardTick(float now)
        {
            APDirectTick(now);
            if (apThrowPhase == 0 || now < apNextKeyAt) return;

            if (apThrowPhase == 1)
            {
                if (apThrowNext >= apThrowTargets.Count)
                {
                    apThrowPhase = 2;
                    apNextKeyAt = now + 0.15f;
                    return;
                }
                int target = apThrowTargets[apThrowNext];
                if (apSelectIdx < target)
                {
                    APPressKey(VK_D, 40);
                    apSelectIdx++;
                    apNextKeyAt = now + 0.12f;
                }
                else if (apSelectIdx > target)
                {
                    APPressKey(VK_A, 40);
                    apSelectIdx--;
                    apNextKeyAt = now + 0.12f;
                }
                else
                {
                    MelonLogger.Msg("[AP] select " + target);
                    APPressKey(VK_SPACE, 40);
                    apThrowNext++;
                    apNextKeyAt = now + 0.15f;
                }
            }
            else if (apThrowPhase == 2)
            {
                MelonLogger.Msg("[AP] throw E");
                APPressKey(VK_E, 40);
                int newSize = apLastHandSize - apThrowTargets.Count;
                int lastSelected = apThrowTargets.Count > 0 ? apThrowTargets[apThrowTargets.Count - 1] : 0;
                apSelectIdx = Math.Min(lastSelected, Math.Max(0, newSize - 1));
                MelonLogger.Msg("[AP] post-throw cursor=" + apSelectIdx + " newHand=" + newSize);
                apThrowPhase = 0;
            }
        }

        void APCallLiar(DeckGameplay dg)
        {
            if (!APUseKeyboard())
            {
                try
                {
                    MelonLogger.Msg("[AP] calling liar via RequestCallLiar()");
                    dg.RequestCallLiar();
                    return;
                }
                catch (Exception ex)
                {
                    MelonLogger.Msg("[AP] direct call failed: " + ex.GetType().Name + ": " + ex.Message + " — using X key");
                }
            }
            MelonLogger.Msg("[AP] calling liar via X key");
            APPressKey(VK_X);
        }

        float aqAutoStartDelay = 10f; // wait 10s after game loads before autostarting
        /// The main menu (and some scene loads) restore the player's saved resolution / fullscreen / quality.
        /// While farm mode is on, put the small window back whenever that happens.
        void FarmModeTick()
        {
            if (!farmMode) return;
            float now = Time.time;
            if (now < farmNextCheck) return;
            farmNextCheck = now + 1f;
            bool reset = Screen.fullScreen || Screen.width > 800 || Application.targetFrameRate != 10;
            if (!reset) return;
            MelonLogger.Msg("[FARM] game restored " + Screen.width + "x" + Screen.height + (Screen.fullScreen ? " fullscreen" : "")
                + ", re-applying farm mode");
            SetFarmMode(true);
        }

        public override void OnUpdate()
        {
            FarmModeTick();
            // Autostart: if aq_autostart.txt exists and game has been up long enough, start auto-queue
            if (!autoQueue && aqAutoStartDelay > 0f)
            {
                aqAutoStartDelay -= Time.deltaTime;
                if (aqAutoStartDelay <= 0f && File.Exists(AQ_AUTOSTART))
                {
                    MelonLogger.Msg("[AQ] autostart flag detected, enabling auto-queue + farm mode");
                    try { File.Delete(AQ_AUTOSTART); } catch { }
                    SetFarmMode(true);
                    ToggleAutoQueue();
                }
            }

            if (Input.GetKeyDown(KeyCode.F8)) ToggleAutoQueue();
            if (Input.GetKeyDown(KeyCode.F9)) SetFarmMode(!farmMode);
            if (Input.GetKeyDown(KeyCode.F10)) DumpFields();
            if (Input.GetKeyDown(KeyCode.F11))
            {
                paused = !paused;
                MelonLogger.Msg(paused ? "logging paused" : "logging resumed");
            }
            try { AutoQueueTick(); }
            catch (Exception ex) { MelonLogger.Warning("aq tick: " + ex.Message); }
            try { Tick(); }
            catch (Exception ex) { MelonLogger.Warning("tick: " + ex.Message); }
            try { TranslationTick(); }
            catch { }
            try { LeaderboardTick(); }
            catch { }
        }

        // ── Leaderboard snapshots ───────────────────────────────────────────
        // Whenever a leaderboard panel is on screen, record its visible rows (what every player sees)
        // to C:\liars\data\leaderboard.jsonl, once per distinct view. Score trajectories over days.
        float lbNextAt; string lbLast = "";
        const string LB_OUT = @"C:\liars\data\leaderboard.jsonl";
        static string HierPath(Transform t) { var s = t.name; while (t.parent != null) { t = t.parent; s = t.name + "/" + s; } return s; }
        void LeaderboardTick()
        {
            float now = Time.time;
            if (now < lbNextAt) return;
            lbNextAt = now + 2f;
            var rows = new List<(string path, string text)>();
            foreach (var t in UnityEngine.Object.FindObjectsOfType<TMP_Text>())
            {
                if (t == null || !t.gameObject.activeInHierarchy) continue;
                string path;
                try { path = HierPath(t.transform); } catch { continue; }
                if (path.IndexOf("eaderboard", StringComparison.OrdinalIgnoreCase) < 0 && path.IndexOf("Rank", StringComparison.Ordinal) < 0) continue;
                string txt = t.text;
                if (string.IsNullOrWhiteSpace(txt)) continue;
                rows.Add((path, txt.Trim()));
            }
            if (rows.Count < 5) return;
            rows.Sort((a, b) => string.CompareOrdinal(a.path, b.path));
            var snap = string.Join("\n", rows.ConvertAll(r => r.text));
            if (snap == lbLast) return;
            lbLast = snap;
            try
            {
                var j = new J("leaderboard").S("at", DateTime.UtcNow.ToString("o")).S("scene", SceneManager.GetActiveScene().name);
                j.L("rows", rows.ConvertAll(r => r.text));
                Directory.CreateDirectory(Path.GetDirectoryName(LB_OUT));
                File.AppendAllText(LB_OUT, j.Done() + "\n");
                MelonLogger.Msg("leaderboard snapshot: " + rows.Count + " rows");
            }
            catch (Exception ex) { MelonLogger.Msg("leaderboard snapshot failed: " + ex.Message); }
        }

        void Tick()
        {
            var mgr = Manager.Instance;
            if (mgr == null || paused) { CloseLog(); return; }
            var lp = mgr.GetLocalPlayer();
            var meNow = lp != null ? lp.gameObject : null;
            string m = meNow == null ? null
                : meNow.GetComponent<ChaosDeckGameplay>() != null ? "chaos"
                : meNow.GetComponent<DeckGameplay>() != null ? "deckgameplay"
                : meNow.GetComponent<BlorfGamePlayMatchMaking>() != null ? "deck_matchmaking"
                : meNow.GetComponent<BlorfGamePlay>() != null ? "deck" : null;
            if (m == null)
            {
                // After I die the game drops my player (or its gameplay part) while the others
                // play on: keep following the table as a spectator until everyone has left.
                if (log == null || mode == null || !iDied || AllPlayers(mgr).Count == 0 || NeedsRescan(mgr)) { CloseLog(); return; }
                m = mode;
                me = meNow;
            }
            else
            {
                // Alive again with a log from a game I died in: a new game, so a new log.
                if (iDied && log != null && !DeadOf(meNow)) CloseLog();
                me = meNow;
            }
            if (m != mode || NeedsRescan(mgr)) { mode = m; Rescan(mgr); }
            if (log == null) OpenLog();

            float now = Time.time;
            PollAll(now);
            if (dealAt >= 0f && now - dealAt > 0.3f) { WriteDealCards(now); dealAt = -1f; }

            if (roundAt >= 0f && now - roundAt >= DealSettle) { WriteRound(); roundAt = -1f; }
            if (now >= nextPose) { WritePoses(now); nextPose = now + PoseInterval; }
        }

        // ── Seats ────────────────────────────────────────────────────────────

        bool NeedsRescan(Manager mgr)
        {
            // Compare instance IDs: scene searches return fresh wrapper objects every frame.
            foreach (var go in AllPlayers(mgr))
                if (!seats.Exists(x => x.Id == go.GetInstanceID())) return true;
            return false;
        }

        static List<GameObject> AllPlayers(Manager mgr)
        {
            var list = new List<GameObject>();
            try
            {
                if (mgr.Players != null)
                    for (int i = 0; i < mgr.Players.Count; i++)
                    {
                        try { var go = mgr.Players[i].gameObject; if (go != null && !list.Exists(g => g.GetInstanceID() == go.GetInstanceID())) list.Add(go); } catch { }
                    }
            }
            catch { }
            // Manager.Players is empty or drops dead players on some frames; always merge in every
            // PlayerStats in the scene so the seat list is stable.
            {
                try
                {
                    foreach (var ps in UnityEngine.Object.FindObjectsOfType<PlayerStats>())
                        if (ps != null && ps.gameObject != null && !list.Exists(g => g.GetInstanceID() == ps.gameObject.GetInstanceID())) list.Add(ps.gameObject);
                }
                catch { }
            }
            list.Sort((x, y) => SlotOf(x).CompareTo(SlotOf(y)));
            return list;
        }

        void Rescan(Manager mgr)
        {
            // Keep tracked state for players already known; only new players start fresh.
            // Seats are only ever added, never dropped or reordered, so seat numbers in the log
            // stay fixed for the whole game.
            int before = seats.Count;
            foreach (var go in AllPlayers(mgr))
            {
                int id = go.GetInstanceID();
                if (seats.Exists(x => x.Id == id)) continue;
                var s = new Seat { Go = go, Id = id, Name = NameOf(go), Chamber = ChamberOf(go), Hand = HandCount(go), Dead = DeadOf(go), Slot = SlotOf(go), Looking = LookingOf(go) };
                FindHead(s);
                seats.Add(s);
            }
            if (seats.Count != before) MelonLogger.Msg("seats found: " + seats.Count);
            // Dump CharController properties once per game to discover rank fields
            if (seats.Count > before && seats.Count > 0)
            {
                try
                {
                    var cc0 = seats[0].Go.GetComponent<CharController>();
                    if (cc0 != null)
                    {
                        var rt = cc0.GetIl2CppType();
                        MelonLogger.Msg("[DUMP] CharController properties:");
                        foreach (var p in rt.GetProperties(R.BindingFlags.Public | R.BindingFlags.Instance))
                        {
                            string val = "?";
                            try { val = p.GetValue(cc0)?.ToString() ?? "null"; } catch { }
                            MelonLogger.Msg("[DUMP]   prop " + p.Name + " (" + p.PropertyType.Name + ") = " + val);
                        }
                        MelonLogger.Msg("[DUMP] CharController fields:");
                        foreach (var f in rt.GetFields(R.BindingFlags.Public | R.BindingFlags.Instance))
                        {
                            string val = "?";
                            try { val = f.GetValue(cc0)?.ToString() ?? "null"; } catch { }
                            MelonLogger.Msg("[DUMP]   field " + f.Name + " (" + f.FieldType.Name + ") = " + val);
                        }
                    }
                }
                catch (Exception dex) { MelonLogger.Warning("[DUMP] error: " + dex.Message); }
            }
            if (log == null) return;
            for (int i = 0; i < seats.Count; i++)
            {
                var s = seats[i];
                int pRank = -1, pLevel = -1;
                try
                {
                    var cc = s.Go.GetComponent<CharController>();
                    if (cc != null)
                    {
                        var rt = cc.GetIl2CppType();
                        var rf = rt.GetProperty("NetworkRank", R.BindingFlags.Public | R.BindingFlags.Instance);
                        if (rf != null) pRank = rf.GetValue(cc).Unbox<int>();
                        var lf = rt.GetProperty("Networklevel", R.BindingFlags.Public | R.BindingFlags.Instance);
                        if (lf != null) pLevel = lf.GetValue(cc).Unbox<int>();
                    }
                }
                catch { }
                Write(new J("seat").I("p", i).S("name", s.Name).B("me", IsMe(s.Go)).S("head", s.HeadSource).I("slot", s.Slot)
                    .V("pos", s.Go.transform.position).Q("rot", s.Go.transform.rotation).B("ranked", IsRanked(s.Go))
                    .I("rank", pRank).I("level", pLevel).B("host", IsHost()).I("tier", LobbyInfo(s.Go, s.Name).rank).B("is_host", LobbyInfo(s.Go, s.Name).host));
            }
        }

        static void FindHeadIn(Transform t, ref Transform best)
        {
            if (best != null) return;
            string n = t.name.ToLowerInvariant();
            if (n.Contains("head") && !n.Contains("top") && !n.Contains("end")) { best = t; return; }
            for (int i = 0; i < t.childCount; i++) FindHeadIn(t.GetChild(i), ref best);
        }

        // Ranked queue (CharController.ranked; public, the player chose the queue).
        static bool IsRanked(GameObject go)
        {
            try { var cc = go.GetComponent<CharController>(); return cc != null && cc.ranked; }
            catch { return false; }
        }

        static void FindHead(Seat s)
        {
            // HeadPivot is what mouse-look turns; prefer it over the skeleton's head bone.
            try
            {
                var cc = s.Go.GetComponent<CharController>();
                if (cc != null && cc.HeadPivot != null) { s.Head = cc.HeadPivot; s.HeadSource = "HeadPivot"; return; }
            }
            catch { }
            var anim = s.Go.GetComponentInChildren<Animator>();
            if (anim != null && anim.isHuman)
            {
                var h = anim.GetBoneTransform(HumanBodyBones.Head);
                if (h != null) { s.Head = h; s.HeadSource = "humanoid:" + h.name; return; }
            }
            Transform best = null;
            FindHeadIn(s.Go.transform, ref best);
            s.Head = best;
            s.HeadSource = best != null ? "name:" + best.name : "none";
        }

        // ── Public game state accessors ──────────────────────────────────────

        bool IsMe(GameObject go) => me != null && go != null && go.GetInstanceID() == me.GetInstanceID();

        static string NameOf(GameObject go)
        {
            try { var ps = go.GetComponent<PlayerStats>(); return ps != null ? ps.NetworkPlayerName : "?"; } catch { return "?"; }
        }

        static bool DeadOf(GameObject go)
        {
            try { var ps = go.GetComponent<PlayerStats>(); return ps != null && ps.NetworkDead; } catch { return false; }
        }

        static bool TurnOf(GameObject go)
        {
            try { var ps = go.GetComponent<PlayerStats>(); return ps != null && ps.NetworkHaveTurn; } catch { return false; }
        }

        // [removed for release: read other players' revolver/card objects; see README 'Scope and conduct']

        // [removed for release: read other players' revolver/card objects; see README 'Scope and conduct']

        // [removed for release: read other players' revolver/card objects; see README 'Scope and conduct']

        // [removed for release: read other players' revolver/card objects; see README 'Scope and conduct']

        static int SlotOf(GameObject go)
        {
            try { var ps = go.GetComponent<PlayerStats>(); return ps != null ? ps.NetworkSlot : -1; } catch { return -1; }
        }

        /// The game's own synced "looking at cards" flag.
        static bool LookingOf(GameObject go)
        {
            try
            {
                var c = go.GetComponent<ChaosDeckGameplay>();
                if (c != null) return c.NetworkLooking;
                var d = go.GetComponent<DeckGameplay>();
                return d != null && d.NetworkLooking;
            }
            catch { return false; }
        }

        /// Chaos: whether the player is aiming, and at which seat slot (-1 when not aiming).
        static (bool, int) AimOf(GameObject go)
        {
            try
            {
                var c = go.GetComponent<ChaosDeckGameplay>();
                if (c != null) return (c.NetworkTakingAim, c.NetworkAim);
            }
            catch { }
            return (false, -1);
        }

        /// Pulls taken so far (the chamber counter shown in game), never the bullet position.
        static int ChamberOf(GameObject go)
        {
            try
            {
                var ch = go.GetComponent<ChaosDeckGameplay>();
                if (ch != null) return ch.Networkcurrentrevoler;
                var d = go.GetComponent<DeckGameplay>();
                if (d != null) return d.NetworkcurrentRevolver;
                var mm = go.GetComponent<BlorfGamePlayMatchMaking>();
                if (mm != null) return mm.Networkcurrentrevoler;
                var b = go.GetComponent<BlorfGamePlay>();
                if (b != null) return b.Networkcurrentrevoler;
            }
            catch { }
            return 0;
        }

        static Il2CppSystem.Collections.Generic.List<GameObject> CardObjects(GameObject go)
        {
            var ch = go.GetComponent<ChaosDeckGameplay>();
            if (ch != null) return ch.Cards;
            var d = go.GetComponent<DeckGameplay>();
            if (d != null) return d.Cards;
            var mm = go.GetComponent<BlorfGamePlayMatchMaking>();
            if (mm != null) return mm.Cards;
            var b = go.GetComponent<BlorfGamePlay>();
            return b != null ? b.Cards : null;
        }

        /// Number of cards in hand. For other players this is only the count, which is visible.
        static int HandCount(GameObject go)
        {
            try
            {
                var cards = CardObjects(go);
                if (cards == null) return 0;
                int n = 0;
                for (int i = 0; i < cards.Count; i++) if (cards[i] != null && cards[i].activeInHierarchy) n++;
                return n;
            }
            catch { return 0; }
        }

        /// Card values in MY hand only.
        List<int> MyHand()
        {
            var outp = new List<int>();
            try
            {
                var cards = CardObjects(me);
                if (cards == null) return outp;
                for (int i = 0; i < cards.Count; i++)
                {
                    var go = cards[i];
                    if (go == null || !go.activeInHierarchy) continue;
                    var c = go.GetComponent<Card>();
                    if (c != null) outp.Add(c.cardtype);
                }
            }
            catch { }
            return outp;
        }

        int TableCard()
        {
            try
            {
                var mgr = Manager.Instance;
                if (mode == "chaos" && mgr.ChaosDeckGame != null) return mgr.ChaosDeckGame.RoundCard;
                if (mode == "deckgameplay" && mgr.DeckGamePlayManager != null) return mgr.DeckGamePlayManager.RoundCard;
                if (mode == "deck" && mgr.BlorfGame != null) return mgr.BlorfGame.RoundCard;
                if (mgr.BlorfGameMatchMaking != null)
                {
                    var f = mgr.BlorfGameMatchMaking.GetIl2CppType().GetField("RoundCard", R.BindingFlags.Public | R.BindingFlags.NonPublic | R.BindingFlags.Instance);
                    if (f != null) return f.GetValue(mgr.BlorfGameMatchMaking).Unbox<int>();
                }
            }
            catch { }
            return 0;
        }

        /// Deck mode name as the game reports it (e.g. Basic / Devil); "?" if unavailable.
        string DeckMode()
        {
            try
            {
                var mgr = Manager.Instance;
                if (mode == "chaos") return "Chaos";
                if (mode == "deckgameplay" && mgr.DeckGamePlayManager != null) return ModeFields(mgr.DeckGamePlayManager);
                if (mode == "deck" && mgr.BlorfGame != null) return mgr.BlorfGame.NetworkDeckMode.ToString();
                if (mgr.BlorfGameMatchMaking != null)
                {
                    var f = mgr.BlorfGameMatchMaking.GetIl2CppType().GetField("NetworkDeckMode", R.BindingFlags.Public | R.BindingFlags.NonPublic | R.BindingFlags.Instance);
                    if (f != null) return f.GetValue(mgr.BlorfGameMatchMaking).ToString();
                    return "matchmaking";
                }
            }
            catch { }
            return "?";
        }

        /// "name=value" for every field of the manager whose name mentions a mode or the Devil.
        static string ModeFields(Component c)
        {
            var parts = new List<string>();
            var flags = R.BindingFlags.Public | R.BindingFlags.NonPublic | R.BindingFlags.Instance;
            foreach (var f in c.GetIl2CppType().GetFields(flags))
            {
                string n = f.Name.ToLowerInvariant();
                if (!(n.Contains("mode") || n.Contains("devil") || n.Contains("rule"))) continue;
                try { var v = f.GetValue(c); parts.Add(f.Name + "=" + (v != null ? v.ToString() : "null")); } catch { }
            }
            return parts.Count > 0 ? string.Join(";", parts) : "?";
        }

        /// The last thrown cards. Only called after a pull, when the call has already revealed them.
        List<int> RevealedCards()
        {
            var outp = new List<int>();
            try
            {
                var mgr = Manager.Instance;
                Il2CppMirror.SyncList<int> lr = null;
                if (mode == "chaos" && mgr.ChaosDeckGame != null) lr = mgr.ChaosDeckGame.LastRound;
                else if (mode == "deckgameplay" && mgr.DeckGamePlayManager != null) lr = mgr.DeckGamePlayManager.LastRound;
                else if (mode == "deck" && mgr.BlorfGame != null) lr = mgr.BlorfGame.LastRound;
                else if (mgr.BlorfGameMatchMaking != null) lr = mgr.BlorfGameMatchMaking.LastRound;
                if (lr != null) for (int i = 0; i < lr.Count; i++) outp.Add(lr[i]);
            }
            catch { }
            return outp;
        }

        // ── Event detection ──────────────────────────────────────────────────

        /// Polls every seat, then interprets the changes. A play is exactly one seat losing 1-3
        /// cards in a frame; several hands shrinking at once is the table being cleared.
        void PollAll(float now)
        {
            if (mode == "chaos") PollChaosFlags(now);
            // A seat whose player object is gone from the scene has left (or been dropped): log it
            // once, with whether they had already died and whether I'm host / still connected.
            try
            {
                var mgr = Manager.Instance;
                if (mgr != null && seats.Count > 1)
                {
                    var present = AllPlayers(mgr);
                    int myId = -1;
                    try { if (me != null) myId = me.GetInstanceID(); } catch { }
                    for (int i = 0; i < seats.Count; i++)
                    {
                        var s = seats[i];
                        if (s.Left || present.Exists(g => g.GetInstanceID() == s.Id)) continue;
                        s.Left = true;
                        Write(new J("left").F("t", now).I("p", i).S("name", s.Name).B("dead", s.Dead || s.EverDead)
                            .B("me", s.Id == myId).B("host", IsHost()).B("connected", IsConnected())
                            .I("present", present.Count));
                    }
                }
            }
            catch { }
            // My own hand whenever it changes (the deal included), so the advisor has it before
            // my first turn. Only my cards: the same thing my own screen shows.
            if (me != null)
            {
                var mine = MyHand();
                var sig = string.Join(",", mine);
                if (sig != myHandSig)
                {
                    myHandSig = sig;
                    Write(new J("hand").F("t", now).L("hand", mine).I("table", TableCard()));
                }
            }
            int n = seats.Count;
            var hand = new int[n]; var turn = new bool[n]; var dead = new bool[n]; var chamber = new int[n];
            int shrinking = 0, shrinkSeat = -1;
            for (int i = 0; i < n; i++)
            {
                var go = seats[i].Go;
                hand[i] = HandCount(go); turn[i] = TurnOf(go); dead[i] = DeadOf(go); chamber[i] = ChamberOf(go);
                if (hand[i] < seats[i].Hand) { shrinking++; shrinkSeat = i; }
            }

            for (int i = 0; i < n; i++)
            {
                var (aiming, aim) = AimOf(seats[i].Go);
                if (aiming != seats[i].Aiming || (aiming && aim != seats[i].Aim))
                {
                    seats[i].Aiming = aiming;
                    seats[i].Aim = aim;
                    Write(new J("aim").F("t", now).I("p", i).B("on", aiming).I("slot", aim));
                }
                bool looking = LookingOf(seats[i].Go);
                if (looking != seats[i].Looking)
                {
                    seats[i].Looking = looking;
                    Write(new J("look").F("t", now).I("p", i).B("on", looking));
                }
                if (turn[i] && !seats[i].Turn)
                {
                    turnSeat = i;
                    turnStart = now;
                    if (lastThrower >= 0 && callerCandidate < 0 && i != lastThrower) callerCandidate = i;
                    var tj = new J("turn").F("t", now).I("p", i);
                    if (IsMe(seats[i].Go)) tj.L("hand", MyHand()).I("table", TableCard());
                    Write(tj);
                }
            }

            // While a Liar call is resolving the game clears the caller's hand; that's not a play.
            if (shrinking == 1 && !pulledThisRound && !LiarCallActive())
            {
                int i = shrinkSeat, k = seats[i].Hand - hand[i];
                if (k <= 3)
                {
                    var pj = new J("play").F("t", now).I("p", i).I("n", k).I("left", hand[i])
                        .F("think", turnSeat == i ? now - turnStart : -1f);
                    if (IsMe(seats[i].Go)) pj.L("hand", MyHand());
                    Write(pj);
                    MelonLogger.Msg(seats[i].Name + " plays " + k + " (" + hand[i] + " left)");
                    aqLastGameActivity = now;
                    lastThrower = i;
                    callerCandidate = -1;
                }
            }

            bool meDead = iDied;
            for (int i = 0; i < n; i++) if (IsMe(seats[i].Go) && dead[i]) meDead = true;
            iDied = meDead;
            for (int i = 0; i < n; i++)
            {
                var s = seats[i];
                if (chamber[i] != s.Chamber) Write(new J("chamber").F("t", now).I("p", i).I("from", s.Chamber).I("to", chamber[i]));
                if (dead[i] && !s.Dead) Write(new J("dead").F("t", now).I("p", i));
                // Fresh deal: my hand grows after a call ended the last round (or at game start).
                // Cards briefly hiding and reappearing mid-round must not count as a new round.
                // My fresh deal starts the next round. Other seats only count once I'm dead (my
                // hand stays empty then): their counts can tick up right after a call, before
                // the deal, which logged rounds early with the old table card.
                if (hand[i] > s.Hand && (pulledThisRound || round == 0) && (IsMe(s.Go) || meDead)) roundAt = now;
                // A redeal with no shot (e.g. an AFK player timing out on a forced call): back to a
                // full hand after having played some. Starts a new round; not a cheat.
                if (hand[i] == 5 && s.MinHand < 5 && !pulledThisRound && round > 0 && (IsMe(s.Go) || meDead) && roundAt < 0f)
                {
                    roundAt = now;
                    lastThrower = -1;
                    Write(new J("redeal").F("t", now).I("p", i));
                }
                if (hand[i] > 0) s.MinHand = Math.Min(s.MinHand, hand[i]);
                bool callShot = mode != "chaos" || fLiar;
                if (callShot && (chamber[i] > s.Chamber || (dead[i] && !s.Dead)) && !pulledSeats.Contains(i))
                {
                    pulledSeats.Add(i);
                    if (!pulledThisRound)
                    {
                        // First pull of the round: work out the call from public events only.
                        pulledThisRound = true;
                        callRevealed = RevealedCards();
                        bool devil = callRevealed.Contains(-1);
                        if (mode == "chaos")
                        {
                            // Decoded rule: the Master (4) is always honest; else it must match the table.
                            int table = TableCard();
                            callLie = callRevealed.Exists(c => c != table && c != 4);
                        }
                        else callLie = !devil && i == lastThrower;
                        callAccused = lastThrower;
                        callCaller = mode == "chaos" || callLie || devil ? (callerCandidate >= 0 ? callerCandidate : turnSeat) : i;
                    }
                    // 2026-11-06: the dead flag can arrive a few frames after the chamber ticks (13 of 3,111 pulls);
                    // writing immediately told the advisor "survives" for a player who then died. Hold the event
                    // until the flag settles or 0.6 s has passed (FlushPendingPulls).
                    if (dead[i])
                        Write(new J("pull").F("t", now).I("p", i).B("died", true).I("pulls", chamber[i])
                            .I("accused", callAccused).I("caller", callCaller).B("lie", callLie).L("revealed", callRevealed));
                    else
                        pendingPulls.Add(new PendingPull { At = now, Seat = i, Pulls = chamber[i], Accused = callAccused,
                            Caller = callCaller, Lie = callLie, Revealed = new List<int>(callRevealed) });
                    MelonLogger.Msg((pulledSeats.Count == 1 ? "call: " + (callRevealed.Contains(-1) ? "DEVIL" : callLie ? "LIE" : "truth") + ", " : "  also: ")
                        + seats[i].Name + " pulls (" + chamber[i] + ")" + (dead[i] ? " and dies" : ""));
                    aqLastGameActivity = now;
                }
            }
            for (int i = 0; i < n; i++)
            {
                var s = seats[i];
                s.Hand = hand[i]; s.Turn = turn[i]; s.Dead = dead[i]; s.Chamber = chamber[i];
            }
            FlushPendingPulls(now, dead);
        }

        class PendingPull { public float At; public int Seat, Pulls, Accused, Caller; public bool Lie; public List<int> Revealed; }
        readonly List<PendingPull> pendingPulls = new List<PendingPull>();

        /// Flush with the seats' last known dead flags (round start, game start, game end).
        void FlushPendingPullsNow()
        {
            if (pendingPulls.Count == 0) return;
            var d = new bool[seats.Count];
            for (int i = 0; i < seats.Count; i++) d[i] = seats[i].Dead || DeadOf(seats[i].Go);
            FlushPendingPulls(float.MaxValue, d);
        }

        /// Write held pull events once the seat's dead flag is known (dead now, or 0.6 s without it).
        void FlushPendingPulls(float now, bool[] dead)
        {
            for (int k = pendingPulls.Count - 1; k >= 0; k--)
            {
                var q = pendingPulls[k];
                bool isDead = q.Seat < dead.Length && dead[q.Seat];
                if (!isDead && now - q.At < 0.6f) continue;
                Write(new J("pull").F("t", q.At).I("p", q.Seat).B("died", isDead).I("pulls", q.Pulls)
                    .I("accused", q.Accused).I("caller", q.Caller).B("lie", q.Lie).L("revealed", q.Revealed));
                if (isDead) MelonLogger.Msg("  " + seats[q.Seat].Name + " dies (flag arrived " + (now - q.At).ToString("0.00") + "s after the pull)");
                pendingPulls.RemoveAt(k);
            }
        }

        /// A Liar call is being resolved (the game's own synced flag).
        bool LiarCallActive()
        {
            try
            {
                var mgr = Manager.Instance;
                if (mode == "chaos") return fLiar;
                if (mode == "deckgameplay" && mgr.DeckGamePlayManager != null) return mgr.DeckGamePlayManager.NetworkLiarCalled;
            }
            catch { }
            return false;
        }

        /// Chaos only: log the manager's phase flags. A Liar call ending starts the next round
        /// (hands carry over, so a fresh deal is not the only round boundary).
        void PollChaosFlags(float now)
        {
            var cm = Manager.Instance != null ? Manager.Instance.ChaosDeckGame : null;
            if (cm == null) return;
            bool liar = false, chaos = false, master = false, caught = false;
            try { liar = cm.NetworkLiarCalled; chaos = cm.NetworkChaosStarted; master = cm.NetworkMasterStarted; caught = cm.NetworkLiarCaught; } catch { return; }
            if (liar != fLiar || chaos != fChaos || master != fMaster || caught != fCaught)
            {
                Write(new J("cflags").F("t", now).B("liar", liar).B("chaos", chaos).B("master", master).B("caught", caught));
                if (chaos && !fChaos) MelonLogger.Msg("CHAOS card thrown: everyone aims");
                if (fLiar && !liar && pulledThisRound) roundAt = now; // the call resolved: next round
            }
            fLiar = liar; fChaos = chaos; fMaster = master; fCaught = caught;
        }

        void WriteRound()
        {
            FlushPendingPullsNow();
            round++;
            lastThrower = -1;
            callerCandidate = -1;
            pulledThisRound = false;
            pulledSeats.Clear();
            foreach (var st in seats) st.MinHand = 99;
            var j = new J("round").F("t", Time.time).I("n", round).I("table", TableCard()).L("hand", MyHand()).S("deck", DeckMode());
            var hands = new List<int>(); var pulls = new List<int>(); var alive = new List<int>();
            foreach (var s in seats) { hands.Add(s.Hand); pulls.Add(s.Chamber); alive.Add(s.Dead ? 0 : 1); }
            Write(j.L("hands", hands).L("pulls", pulls).L("alive", alive));
            MelonLogger.Msg("round " + round + " | table card " + TableCard() + " | my hand " + string.Join(",", MyHand()));
            aqLastGameActivity = Time.time;
            for (int i = 0; i < seats.Count; i++)
                if (seats[i].Head != null) Write(new J("anchor").I("p", i).V("head", seats[i].Head.position).V("body", seats[i].Go.transform.position));
        }

        void WritePoses(float now)
        {
            if (log == null) return;
            for (int i = 0; i < seats.Count; i++)
            {
                var s = seats[i];
                if (s.Dead || s.Head == null) continue;
                Write(new J("pose").F("t", now).I("p", i).V("pos", s.Head.position).Q("rot", s.Head.rotation).B("look", s.Looking));
            }
        }

        // ── Field discovery ──────────────────────────────────────────────────

        /// Lists fields and properties (names and types) of the game classes, with values only
        /// for float / vector / quaternion members, to find which ones carry head or look angles.
        void DumpFields()
        {
            try
            {
                Directory.CreateDirectory(LogDir);
                var sb = new StringBuilder();
                var mgr = Manager.Instance;
                var lp = mgr != null ? mgr.GetLocalPlayer() : null;
                var go = lp != null ? lp.gameObject : null;
                if (go == null) { MelonLogger.Msg("join a game first"); return; }
                foreach (var comp in go.GetComponents<Component>())
                {
                    if (comp == null) continue;
                    DumpType(sb, comp);
                }
                if (mgr.DeckGamePlayManager != null)
                {
                    sb.AppendLine("##### DeckGamePlayManager");
                    DumpType(sb, mgr.DeckGamePlayManager);
                    sb.AppendLine("  modes: " + ModeFields(mgr.DeckGamePlayManager));
                }
                var anim = go.GetComponentInChildren<Animator>();
                sb.AppendLine("Animator: " + (anim == null ? "none" : "isHuman=" + anim.isHuman));
                sb.AppendLine("Transforms with head/neck/eye/cam/look in the name:");
                ListTransforms(sb, go.transform, 0);
                var path = Path.Combine(LogDir, "fields.txt");
                File.WriteAllText(path, sb.ToString());
                MelonLogger.Msg("field dump written to " + path);
            }
            catch (Exception ex) { MelonLogger.Warning("dump: " + ex.Message); }
        }

        static void ListTransforms(StringBuilder sb, Transform t, int depth)
        {
            string n = t.name.ToLowerInvariant();
            if (n.Contains("head") || n.Contains("neck") || n.Contains("eye") || n.Contains("cam") || n.Contains("look"))
                sb.AppendLine(new string(' ', depth * 2) + t.name + " rot=" + t.rotation.eulerAngles);
            for (int i = 0; i < t.childCount; i++) ListTransforms(sb, t.GetChild(i), depth + 1);
        }

        static void DumpType(StringBuilder sb, Component comp)
        {
            var flags = R.BindingFlags.Public | R.BindingFlags.NonPublic | R.BindingFlags.Instance | R.BindingFlags.DeclaredOnly;
            for (var t = comp.GetIl2CppType(); t != null; t = t.BaseType)
            {
                string tn = t.Name;
                if (tn == "MonoBehaviour" || tn == "Behaviour" || tn == "Component" || tn == "Object" || tn == "NetworkBehaviour") break;
                sb.AppendLine("== " + tn);
                try
                {
                    foreach (var f in t.GetFields(flags))
                        sb.AppendLine("  field " + f.FieldType.Name + " " + f.Name + PoseValue(f.FieldType.Name, () => f.GetValue(comp)));
                }
                catch { }
                try
                {
                    foreach (var p in t.GetProperties(flags))
                        sb.AppendLine("  prop  " + p.PropertyType.Name + " " + p.Name);
                }
                catch { }
            }
        }

        static string PoseValue(string typeName, Func<Il2CppSystem.Object> get)
        {
            try
            {
                switch (typeName)
                {
                    case "Single": return " = " + get().Unbox<float>().ToString(Inv);
                    case "Vector2": return " = " + get().Unbox<Vector2>();
                    case "Vector3": return " = " + get().Unbox<Vector3>();
                    case "Quaternion": return " = " + get().Unbox<Quaternion>().eulerAngles;
                    default: return "";
                }
            }
            catch { return ""; }
        }

        // ── Log file ─────────────────────────────────────────────────────────

        void OpenLog()
        {
            Directory.CreateDirectory(LogDir);
            var path = Path.Combine(LogDir, "game_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".jsonl");
            log = new StreamWriter(path, false, new UTF8Encoding(false)) { AutoFlush = false };
            round = 0;
            // A new game: forget the last game's round state, which can end mid-round (someone
            // leaves). Otherwise the new deal looks like players gaining cards mid-round.
            lastThrower = -1;
            callerCandidate = -1;
            pulledThisRound = false;
            pulledSeats.Clear();
            pendingPulls.Clear();
            Write(new J("start").S("mode", mode).S("utc", DateTime.UtcNow.ToString("o")).S("version", "0.1.0"));
            WriteMmr("start");
            Rescan(Manager.Instance);
            MelonLogger.Msg("logging to " + path);
        }

        void CloseLog()
        {
            if (log == null) return;
            // Why the log is closing: still in a game object-wise, still connected, host or not.
            try
            {
                int present = 0;
                try { var mgr = Manager.Instance; if (mgr != null) present = AllPlayers(mgr).Count; } catch { }
                FlushPendingPullsNow();
                Write(new J("end").F("t", Time.time).B("in_game", AQInGame()).B("connected", IsConnected())
                    .B("host", IsHost()).B("i_died", iDied).I("present", present));
                WriteMmr("end");
            }
            catch { }
            try { log.Flush(); log.Dispose(); } catch { }
            log = null;
            mode = null;
            iDied = false;
            seats.Clear();
        }

        float lastFlush;
        /// When a Devil's Deal banner showed (-1: none pending): the revealed cards are logged
        /// a moment later, once the game has them.
        float dealAt = -1f;

        /// Every card thrown this round in throw order (DeckGamePlayManager.LastRoundSpotOn,
        /// whose last 4 a Devil's Deal reveals), plus the last throw (LastRound).
        void WriteDealCards(float now)
        {
            var spot = new List<int>(); var last = new List<int>();
            try
            {
                var dm = Manager.Instance != null ? Manager.Instance.DeckGamePlayManager : null;
                if (dm != null)
                {
                    var l = dm.LastRoundSpotOn;
                    if (l != null) for (int i = 0; i < l.Count; i++) spot.Add(l[i]);
                    var lr = dm.LastRound;
                    if (lr != null) for (int i = 0; i < lr.Count; i++) last.Add(lr[i]);
                }
            }
            catch { }
            Write(new J("dealcards").F("t", now).L("round_cards", spot).L("last", last));
        }

        static Plugin instance;

        /// The menu's ChangeSettings re-applies the saved display settings every time its scene starts, and
        /// LoadPrefs sets Screen.fullScreen. Screen changes take effect at the end of the frame, so re-applying
        /// farm mode from a postfix in the same frame means the small window never flashes to fullscreen.
        void FarmReapply(string who)
        {
            if (!farmMode) return;
            try
            {
                SetFarmMode(true, true);
                MelonLogger.Msg("[FARM] ChangeSettings." + who + " ran, farm display re-applied in the same frame");
            }
            catch (Exception ex) { MelonLogger.Msg("[FARM] re-apply after " + who + " failed: " + ex.Message); }
        }
        bool FarmSkip(string who)
        {
            if (!farmMode) return false;
            MelonLogger.Msg("[FARM] skipped ChangeSettings." + who + " (would have requested fullscreen)");
            return true;
        }
        [HarmonyLib.HarmonyPatch(typeof(ChangeSettings), "ApplyDisplay")]
        static class FarmApplyDisplayPatch { static bool Prefix() { return !(instance != null && instance.FarmSkip("ApplyDisplay")); } }
        [HarmonyLib.HarmonyPatch(typeof(ChangeSettings), "ApplySettings")]
        static class FarmApplySettingsPatch { static void Postfix() { instance?.FarmReapply("ApplySettings"); } }
        [HarmonyLib.HarmonyPatch(typeof(ChangeSettings), "LoadPrefs")]
        static class FarmLoadPrefsPatch { static bool Prefix() { return !(instance != null && instance.FarmSkip("LoadPrefs")); } }
        [HarmonyLib.HarmonyPatch(typeof(ChangeSettings), "SetFullScreen")]
        static class FarmSetFullScreenPatch { static bool Prefix() { return !(instance != null && instance.FarmSkip("SetFullScreen")); } }
        [HarmonyLib.HarmonyPatch(typeof(ChangeSettings), "SetResolution")]
        static class FarmSetResolutionPatch { static bool Prefix() { return !(instance != null && instance.FarmSkip("SetResolution")); } }

        /// Every client shows a "Devil's Deal!" banner with the caller's name when someone calls
        /// it; a failed deal otherwise looks just like a wrong Liar call. Read only: logged.
        [HarmonyLib.HarmonyPatch(typeof(DeckGamePlayManager), nameof(DeckGamePlayManager.UserCode_RpcApplyDevilDealCallUi__String))]
        static class DealBannerPatch
        {
            static void Postfix(string __0)
            {
                try
                {
                    if (!string.IsNullOrEmpty(__0))
                    {
                        instance?.Write(new J("deal").F("t", Time.time).S("caller", __0));
                        if (instance != null) instance.dealAt = Time.time;
                    }
                }
                catch { }
            }
        }

        /// Every client hides the thrown cards' hand slots when someone plays (RpcHideCards: slot
        /// indexes only, never card values), so which slots a player picks is on everyone's screen.
        /// Logged per throw, to look for players whose picks give their bluffs away.
        // The RPC's own body isn't exposed by the interop assembly; it hands the indexes to
        // ApplyHiddenCardIndexes, which is.
        [HarmonyLib.HarmonyPatch(typeof(DeckGameplay), "ApplyHiddenCardIndexes")]
        static class HideCardsPatch
        {
            static void Postfix(DeckGameplay __instance, object __0)
            {
                try
                {
                    var p = instance;
                    if (p == null || __instance == null) return;
                    var go = __instance.gameObject;
                    int seat = -1;
                    for (int i = 0; i < p.seats.Count; i++)
                        if (p.seats[i].Go != null && p.seats[i].Go.GetInstanceID() == go.GetInstanceID()) seat = i;
                    var idx = new List<int>();
                    if (__0 is Il2CppInterop.Runtime.InteropTypes.Arrays.Il2CppStructArray<int> a)
                        for (int i = 0; i < a.Length; i++) idx.Add(a[i]);
                    p.Write(new J("slots").F("t", Time.time).I("p", seat).S("name", NameOf(go)).L("idx", idx));
                }
                catch { }
            }
        }

        void Write(J j)
        {
            if (log == null) return;
            log.WriteLine(j.Done());
            if (Time.time - lastFlush > 1f) { log.Flush(); lastFlush = Time.time; }
        }

        public override void OnApplicationQuit() => CloseLog();

        // ── Chat translation ─────────────────────────────────────────
        int tlMsgCount = 0;
        readonly Queue<string> tlPending = new Queue<string>();
        static readonly Regex RichTextTags = new Regex("<[^>]+>");

        void TranslationTick()
        {
            var chat = ChatArayuz.instance;
            if (chat == null) return;
            var msgs = chat.messages;
            if (msgs == null) return;

            // inject any finished translations first (main thread)
            lock (tlPending)
            {
                while (tlPending.Count > 0)
                {
                    chat.AddMessage(tlPending.Dequeue());
                    tlMsgCount++;
                }
            }

            int count = msgs.Count;
            if (count <= tlMsgCount) { tlMsgCount = count; return; }

            for (int i = tlMsgCount; i < count; i++)
            {
                string raw = msgs[i];
                if (raw == null) continue;
                string stripped = RichTextTags.Replace(raw, "");
                if (NeedsTranslation(stripped))
                {
                    string capture = raw;
                    ThreadPool.QueueUserWorkItem(_ => TranslateMessage(capture));
                }
            }
            tlMsgCount = count;
        }

        static bool NeedsTranslation(string s)
        {
            foreach (char c in s)
            {
                if (c >= 0x4E00 && c <= 0x9FFF) return true;
                if (c >= 0x3040 && c <= 0x30FF) return true;
                if (c >= 0xAC00 && c <= 0xD7AF) return true;
                if (c >= 0x0400 && c <= 0x04FF) return true;
                if (c >= 0x0600 && c <= 0x06FF) return true;
                if (c >= 0x0E00 && c <= 0x0E7F) return true;
            }
            return false;
        }

        void TranslateMessage(string rawMsg)
        {
            try
            {
                string stripped = RichTextTags.Replace(rawMsg, "");
                int colon = stripped.IndexOf(':');
                string name = colon > 0 ? stripped.Substring(0, colon).Trim() : "?";
                string text = colon > 0 ? stripped.Substring(colon + 1).Trim() : stripped;
                if (string.IsNullOrWhiteSpace(text)) return;

                string encoded = Uri.EscapeDataString(text);
                string url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=" + encoded;

                using (var wc = new WebClient())
                {
                    wc.Encoding = Encoding.UTF8;
                    wc.Headers.Add("User-Agent", "Mozilla/5.0");
                    string json = wc.DownloadString(url);
                    string translated = ParseGoogleTranslate(json);
                    if (!string.IsNullOrEmpty(translated) && translated != text)
                    {
                        string line = "<color=#66CCFF>{TL} " + EscapeChat(name) + ":</color> " + EscapeChat(translated);
                        lock (tlPending) { tlPending.Enqueue(line); }
                    }
                }
            }
            catch { }
        }

        static string ParseGoogleTranslate(string json)
        {
            // response: [[["translated","source",...],...],...] — concat all translated segments
            var sb = new StringBuilder();
            int depth = 0, seg = 0;
            bool inStr = false, escape = false;
            var cur = new StringBuilder();
            for (int i = 0; i < json.Length; i++)
            {
                char c = json[i];
                if (escape) { if (inStr) cur.Append(c); escape = false; continue; }
                if (c == '\\') { escape = true; if (inStr) cur.Append(c); continue; }
                if (c == '"') {
                    if (inStr) {
                        if (depth == 3 && seg % 2 == 0) sb.Append(cur);
                        seg++;
                        cur.Clear();
                    } else { seg = 0; }
                    inStr = !inStr;
                    continue;
                }
                if (inStr) { cur.Append(c); continue; }
                if (c == '[') { depth++; seg = 0; }
                else if (c == ']') { depth--; seg = 0; }
                else if (c == ',') seg = 0;
            }
            return sb.ToString();
        }

        static string EscapeChat(string s) => s.Replace("<", "‹").Replace(">", "›");

        /// Tiny JSON line builder (keys are fixed identifiers, strings are escaped).
        class J
        {
            readonly StringBuilder sb = new StringBuilder(128);

            public J(string ev) { sb.Append("{\"e\":\"").Append(ev).Append('"'); }

            StringBuilder Key(string k) => sb.Append(",\"").Append(k).Append("\":");
            public J I(string k, int v) { Key(k).Append(v.ToString(Inv)); return this; }
            public J B(string k, bool v) { Key(k).Append(v ? "true" : "false"); return this; }
            public J F(string k, float v) { Key(k).Append(v.ToString("0.###", Inv)); return this; }
            public J V(string k, Vector3 v) { Key(k).Append('[').Append(N(v.x)).Append(',').Append(N(v.y)).Append(',').Append(N(v.z)).Append(']'); return this; }
            public J Q(string k, Quaternion q) { Key(k).Append('[').Append(N(q.x)).Append(',').Append(N(q.y)).Append(',').Append(N(q.z)).Append(',').Append(N(q.w)).Append(']'); return this; }
            public J L(string k, List<int> v) { Key(k).Append('[').Append(string.Join(",", v)).Append(']'); return this; }
            public J L(string k, List<string> v)
            {
                Key(k).Append('[');
                for (int i = 0; i < v.Count; i++)
                {
                    if (i > 0) sb.Append(',');
                    sb.Append('"');
                    foreach (char c in v[i] ?? "")
                    {
                        if (c == '"' || c == '\\') sb.Append('\\').Append(c);
                        else if (c < ' ') sb.Append("\\u").Append(((int)c).ToString("x4"));
                        else sb.Append(c);
                    }
                    sb.Append('"');
                }
                sb.Append(']');
                return this;
            }

            public J S(string k, string v)
            {
                Key(k).Append('"');
                foreach (char c in v ?? "")
                {
                    if (c == '"' || c == '\\') sb.Append('\\').Append(c);
                    else if (c < ' ') sb.Append("\\u").Append(((int)c).ToString("x4"));
                    else sb.Append(c);
                }
                sb.Append('"');
                return this;
            }

            static string N(float f) => f.ToString("0.####", Inv);
            public string Done() => sb.Append('}').ToString();
        }
    }
}
