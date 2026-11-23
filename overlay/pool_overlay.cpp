// overlay/pool_overlay.cpp — live aim overlay for 8 Ball Pool (Google Play
// Games on PC). Win32 + GDI only; reuses the dependency-free pool vision and
// solver. A transparent, click-through, always-on-top window follows the game
// client, captures it with PrintWindow, detects table + balls (adaptive to
// the dark felt) and draws the best solved shot guides over the game.
//
// Usage:
//   pool-overlay.exe [--window "8 Ball Pool"] [--table 900x530]
//                    [--rails 2] [--shots 1] [--fps 12]
// Hotkeys: F9 = show/hide guides, F12 = quit.
//
// This is a training / analysis aid. Using overlays during online matches
// violates the game's terms of service and can lead to account bans.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define UNICODE
#define _UNICODE
#include <windows.h>
#include <shellapi.h>
#include "pool/vision.hpp"
#include "pool/solver.hpp"
#include "pool/stabilize.hpp"
#include "pool/physics.hpp"
#include <algorithm>
#include <map>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "menu_ui.hpp"
#include <cwctype>
#include <exception>
#include <string>
#include <vector>

#ifndef PW_RENDERFULLCONTENT
#define PW_RENDERFULLCONTENT 0x00000002  // Win 8.1+, needed for GPU windows
#endif

namespace {

constexpr COLORREF kKey = RGB(255, 0, 255);  // transparent color key

struct Args {
    std::wstring window = L"8 Ball Pool";
    double tw = 900, th = 530;  // logical table matching 8BP felt aspect
    int rails = 2;
    int shots = 1;
    int fps = 12;
    // "edit anything" tunables (pool-overlay.ini + live hotkeys)
    double ballR = 0;        // px, 0 = auto
    double feltTolS = 0.20;  // dominant-surface match windows
    double feltTolV = 0.30;
    bool stripLines = true;
    bool aimPreview = true;
    bool trainer = true;       // classic trainer look: guide lines for every ball
    bool manualRect = false;
    double rectX0 = 0, rectY0 = 0, rectX1 = 1, rectY1 = 1;  // client fractions
};

Args g_args;
HWND g_ov = nullptr;
bool g_visible = true;
bool g_menuHidden = false;  // Insert: hide panel + HUD, keep only the guides

// Temporal stabilization state (anti-jitter).
static pool::BallStabilizer g_ballStab;
static pool::Rect g_stabRect;
static pool::Quad g_stabQuad;
static double g_stabRPx = 0;
static bool g_stabHasQuad = false, g_stabValid = false;
static HWND g_stabGame = nullptr;

// Memory double buffer: the whole frame is composed off-screen and blitted
// once per tick. Drawing straight onto the color-key layered window with
// several GetDC/ReleaseDC passes makes every intermediate state visible —
// the overlay would flicker nonstop.
static HDC g_memDC = nullptr;
static HBITMAP g_memBmp = nullptr, g_memOldBmp = nullptr;
static int g_memW = 0, g_memH = 0;

void fillKey(HDC dc, const RECT& rc);  // defined below (forward decl)

static HDC beginFrame(const RECT& rc) {
    int w = rc.right - rc.left, h = rc.bottom - rc.top;
    if (w <= 0 || h <= 0) return nullptr;
    if (!g_memDC) {
        g_memDC = CreateCompatibleDC(nullptr);
        if (!g_memDC) return nullptr;
    }
    if (g_memW != w || g_memH != h) {
        if (g_memBmp) {
            SelectObject(g_memDC, g_memOldBmp);
            DeleteObject(g_memBmp);
            g_memBmp = nullptr;
        }
        HDC wdc = GetDC(g_ov);
        g_memBmp = CreateCompatibleBitmap(wdc, w, h);
        ReleaseDC(g_ov, wdc);
        if (!g_memBmp) return nullptr;
        g_memOldBmp = static_cast<HBITMAP>(
            SelectObject(g_memDC, g_memBmp));
        g_memW = w;
        g_memH = h;
    }
    fillKey(g_memDC, rc);
    return g_memDC;
}

static void presentFrame(const RECT& rc) {
    if (!g_memDC || !g_ov) return;
    HDC wdc = GetDC(g_ov);
    BitBlt(wdc, rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top,
           g_memDC, 0, 0, SRCCOPY);
    ReleaseDC(g_ov, wdc);
}

void printUsage() {
    std::printf(
        "pool-overlay - live aim overlay for 8 Ball Pool (Google Play Games on PC)\n"
        "usage: pool-overlay.exe [options]\n"
        "  --window TITLE   game window title substring (default: \"8 Ball Pool\")\n"
        "  --table WxH      logical table size (default 900x530, matches 8BP felt)\n"
        "  --rails N        max cushion bounces for bank/kick shots (0-3, default 2)\n"
        "  --shots N        draw the best N shots (1-5, default 1)\n"
        "  --fps N          analysis frames per second (2-30, default 12)\n"
        "hotkeys: F9 = show/hide guides, F10 = save settings, F12 = quit\n"
        "         F7/F8 = ball size -/+ (0 = auto), [ / ] = felt tolerance\n"
         "         A = aim preview, R = manual rect, arrows = move rect,\n"
         "         1-5 = shots count, D = dump settings\n"
         "settings load from pool-overlay.ini next to the exe;\n"
         "pool-trainer tune <screenshot> --save generates it\n");
}

std::wstring lower(std::wstring s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](wchar_t c) { return static_cast<wchar_t>(::towlower(c)); });
    return s;
}

static void LOGF(const char* fmt, ...);  // defined below (after capture)

static std::wstring iniPath() {
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir(exe);
    size_t slash = dir.find_last_of(L'\\');
    if (slash != std::wstring::npos) dir.resize(slash);
    return dir + L"\\pool-overlay.ini";
}

// "edit anything" persistence: key=value lines consumed at startup,
// rewritten by F10 after live hotkey tuning.
void applyIni(Args& a, const std::wstring& path) {
    HANDLE f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, 0, nullptr);
    if (f == INVALID_HANDLE_VALUE) return;
    char buf[4096] = {};
    DWORD got = 0;
    ReadFile(f, buf, sizeof(buf) - 1, &got, nullptr);
    CloseHandle(f);
    auto strVal = [&](const char* key) -> const char* {
        const char* p = std::strstr(buf, key);
        if (!p) return nullptr;
        p += std::strlen(key);
        return *p == '=' ? p + 1 : nullptr;
    };
    auto numVal = [&](const char* key) -> double {
        const char* p = strVal(key);
        return p ? std::atof(p) : 0.0;
    };
    if (const char* p = strVal("window")) {
        if (strcmp(p, "auto") == 0) {
            a.window.clear();  // auto-detect any pool game
        } else {
            int wl = MultiByteToWideChar(CP_UTF8, 0, p, -1, nullptr, 0);
            if (wl > 1) {
                std::wstring w(wl - 1, 0);
                MultiByteToWideChar(CP_UTF8, 0, p, -1, w.data(), wl);
                a.window = w;
            }
        }
    }
    if (strVal("tableW")) a.tw = numVal("tableW");
    if (strVal("tableH")) a.th = numVal("tableH");
    if (strVal("rails")) a.rails = static_cast<int>(numVal("rails"));
    if (strVal("shots")) a.shots = static_cast<int>(numVal("shots"));
    if (strVal("fps")) a.fps = static_cast<int>(numVal("fps"));
    if (strVal("guides")) g_visible = numVal("guides") != 0;
    if (strVal("aimPreview")) a.aimPreview = numVal("aimPreview") != 0;
    if (strVal("trainer")) a.trainer = numVal("trainer") != 0;
    if (strVal("ballR")) a.ballR = numVal("ballR");
    if (strVal("feltTolS")) a.feltTolS = numVal("feltTolS");
    if (strVal("feltTolV")) a.feltTolV = numVal("feltTolV");
    if (strVal("stripLines")) a.stripLines = numVal("stripLines") != 0;
    if (strVal("manualRect")) a.manualRect = numVal("manualRect") != 0;
    if (strVal("rectX0")) a.rectX0 = numVal("rectX0");
    if (strVal("rectY0")) a.rectY0 = numVal("rectY0");
    if (strVal("rectX1")) a.rectX1 = numVal("rectX1");
    if (strVal("rectY1")) a.rectY1 = numVal("rectY1");
}

void saveIni() {
    std::wstring path = iniPath();
    HANDLE f = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr,
                           CREATE_ALWAYS, 0, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        LOGF("[!] cannot write ini");
        return;
    }
    char buf[1024];
    // Fixed window title or "auto" (= detect any pool game).
    std::string winNarrow = g_args.window.empty() ? "auto" : "";
    if (winNarrow.empty()) {
        int wl = WideCharToMultiByte(CP_UTF8, 0, g_args.window.c_str(), -1,
                                     nullptr, 0, nullptr, nullptr);
        winNarrow.resize(wl > 1 ? wl - 1 : 0);
        if (wl > 1)
            WideCharToMultiByte(CP_UTF8, 0, g_args.window.c_str(), -1,
                                winNarrow.data(), wl, nullptr, nullptr);
    }
    int n = std::snprintf(buf, sizeof(buf),
        "window=%s\ntableW=%.0f\ntableH=%.0f\nrails=%d\nshots=%d\n"
        "fps=%d\nguides=%d\naimPreview=%d\ntrainer=%d\nballR=%.1f\nfeltTolS=%.2f\n"
        "feltTolV=%.2f\nstripLines=%d\nmanualRect=%d\n"
        "rectX0=%.4f\nrectY0=%.4f\nrectX1=%.4f\nrectY1=%.4f\n",
        winNarrow.c_str(), g_args.tw, g_args.th, g_args.rails, g_args.shots,
        g_args.fps, g_visible ? 1 : 0, g_args.aimPreview ? 1 : 0,
        g_args.trainer ? 1 : 0,
        g_args.ballR, g_args.feltTolS, g_args.feltTolV,
        g_args.stripLines ? 1 : 0, g_args.manualRect ? 1 : 0, g_args.rectX0,
        g_args.rectY0, g_args.rectX1, g_args.rectY1);
    DWORD wrote = 0;
    WriteFile(f, buf, static_cast<DWORD>(n > 0 ? n : 0), &wrote, nullptr);
    CloseHandle(f);
    LOGF("[i] settings saved");
}

Args parseArgs() {
    Args a;
    int n = 0;
    LPWSTR* v = CommandLineToArgvW(GetCommandLineW(), &n);
    if (!v) return a;
    for (int i = 1; i < n; ++i) {
        std::wstring s = v[i];
        auto next = [&]() -> std::wstring { return (i + 1 < n) ? v[++i] : L""; };
        if (s == L"--window") a.window = next();
        else if (s == L"--table") swscanf(next().c_str(), L"%lfx%lf", &a.tw, &a.th);
        else if (s == L"--rails") a.rails = _wtoi(next().c_str());
        else if (s == L"--shots") a.shots = _wtoi(next().c_str());
        else if (s == L"--fps") a.fps = _wtoi(next().c_str());
        else if (s == L"--ballr") a.ballR = _wtof(next().c_str());
        else if (s == L"--tols") a.feltTolS = _wtof(next().c_str());
        else if (s == L"--tolv") a.feltTolV = _wtof(next().c_str());
        else if (s == L"--no-strip") a.stripLines = false;
        else if (s == L"--no-aim") a.aimPreview = false;
        else if (s == L"--no-trainer") a.trainer = false;
        else if (s == L"--manual-rect") a.manualRect = true;
        else if (s == L"--rect") {
            std::wstring v = next();
            swscanf(v.c_str(), L"%lf,%lf,%lf,%lf", &a.rectX0, &a.rectY0,
                    &a.rectX1, &a.rectY1);
            a.manualRect = true;
        }
        else if (s == L"--help" || s == L"-h") { printUsage(); ExitProcess(0); }
    }
    LocalFree(v);
    applyIni(a, iniPath());
    a.rails = std::min(std::max(a.rails, 0), 3);
    a.shots = std::min(std::max(a.shots, 1), 5);
    a.fps = std::min(std::max(a.fps, 2), 30);
    if (a.tw < 200) a.tw = 900;
    if (a.th < 100) a.th = 530;
    return a;
}

HWND findGameWindow(const std::wstring& needleLower) {
    struct Ctx { const std::wstring* needle; HWND best; } ctx{&needleLower, nullptr};
    EnumWindows([](HWND h, LPARAM lp) -> BOOL {
        auto* c = reinterpret_cast<Ctx*>(lp);
        if (!IsWindowVisible(h) || IsIconic(h)) return TRUE;
        wchar_t buf[512] = {};
        if (GetWindowTextW(h, buf, 512) <= 0) return TRUE;
        if (lower(buf).find(*c->needle) != std::wstring::npos) {
            c->best = h;
            return FALSE;
        }
        return TRUE;
    }, reinterpret_cast<LPARAM>(&ctx));
    return ctx.best;
}

// Capture the game client into a pool::Image (RGB8). Prefers PrintWindow
// (captures the window surface itself, so our overlay never appears in it);
// falls back to a screen blit of the client rect, hiding the overlay first.
bool captureClient(HWND hwnd, pool::Image& out) {
    RECT rc;
    if (!GetClientRect(hwnd, &rc)) return false;
    int w = rc.right - rc.left, h = rc.bottom - rc.top;
    if (w < 64 || h < 48) return false;

    HDC wdc = GetWindowDC(hwnd);
    HDC mem = CreateCompatibleDC(wdc);
    HBITMAP bmp = CreateCompatibleBitmap(wdc, w, h);
    HGDIOBJ oldBmp = SelectObject(mem, bmp);

    BOOL ok = PrintWindow(hwnd, mem, PW_CLIENTONLY | PW_RENDERFULLCONTENT);
    if (!ok) {
        ShowWindow(g_ov, SW_HIDE);
        POINT pt{};
        ClientToScreen(hwnd, &pt);
        HDC sdc = GetDC(nullptr);
        ok = BitBlt(mem, 0, 0, w, h, sdc, pt.x, pt.y, SRCCOPY | CAPTUREBLT);
        ReleaseDC(nullptr, sdc);
        ShowWindow(g_ov, SW_SHOWNOACTIVATE);
    }

    if (ok) {
        BITMAPINFO bi{};
        bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bi.bmiHeader.biWidth = w;
        bi.bmiHeader.biHeight = -h;  // top-down rows
        bi.bmiHeader.biPlanes = 1;
        bi.bmiHeader.biBitCount = 24;
        bi.bmiHeader.biCompression = BI_RGB;
        const int stride = ((w * 3 + 3) / 4) * 4;  // 24bpp rows are padded
        std::vector<unsigned char> buf(static_cast<size_t>(stride) * h);
        auto convertBits = [&]() {
            out.w = w;
            out.h = h;
            out.px.resize(static_cast<size_t>(w) * h * 3);
            for (int y = 0; y < h; ++y) {
                const unsigned char* row = &buf[static_cast<size_t>(y) * stride];
                unsigned char* dst = &out.px[static_cast<size_t>(y) * w * 3];
                for (int x = 0; x < w; ++x) {  // BGR -> RGB
                    dst[x * 3 + 0] = row[x * 3 + 2];
                    dst[x * 3 + 1] = row[x * 3 + 1];
                    dst[x * 3 + 2] = row[x * 3 + 0];
                }
            }
        };
        if (GetDIBits(mem, bmp, 0, h, buf.data(), &bi, DIB_RGB_COLORS)) {
            convertBits();
        } else {
            ok = FALSE;
        }
        // GPU-protected windows can render PrintWindow as a black frame:
        // detect it and fall back to a screen blit of the client rect.
        if (ok) {
            long sum = 0;
            int samples = 0;
            for (size_t i = 0; i < out.px.size(); i += 48) sum += out.px[i];
            if (samples > 0 && sum / samples < 6) {
                ShowWindow(g_ov, SW_HIDE);
                POINT pt{};
                ClientToScreen(hwnd, &pt);
                HDC sdc = GetDC(nullptr);
                ok = BitBlt(mem, 0, 0, w, h, sdc, pt.x, pt.y, SRCCOPY | CAPTUREBLT);
                ReleaseDC(nullptr, sdc);
                if (ok) {
                    GetDIBits(mem, bmp, 0, h, buf.data(), &bi, DIB_RGB_COLORS);
                    convertBits();
                }
                ShowWindow(g_ov, SW_SHOWNOACTIVATE);
            }
        }
    }

    SelectObject(mem, oldBmp);
    DeleteObject(bmp);
    DeleteDC(mem);
    ReleaseDC(hwnd, wdc);
    return ok != FALSE;
}

// ---- diagnostics: everything goes to pool-overlay.log (no console) ----
static std::wstring g_logPath;

static void LOGF(const char* fmt, ...) {
    if (g_logPath.empty()) return;
    FILE* f = _wfopen(g_logPath.c_str(), L"ab");
    if (!f) return;
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(f, fmt, ap);
    va_end(ap);
    fputc('\n', f);
    fclose(f);
}

// ---- game discovery that works with ANY pool game ----
// A fixed title (menu/ini) does a substring match; auto mode scores every
// visible window (pool/billiard/snooker keywords first) and capture-probes
// candidates one at a time until a plausible table is found.
static std::vector<std::wstring> g_scanTitles;
static std::vector<HWND> g_scanHwnds;
static std::vector<int> g_scanOrder;  // probe order (score-desc)
static int g_probeCursor = 0;
static HWND g_game = nullptr;
static bool g_running = true;
static bool g_forceScan = false;

static bool excludedWindow(HWND h) {
    wchar_t cls[64] = {};
    GetClassNameW(h, cls, 64);
    if (!wcscmp(cls, L"Progman") || !wcscmp(cls, L"WorkerW") ||
        !wcscmp(cls, L"Shell_TrayWnd") || !wcscmp(cls, L"Shell_SecondaryTrayWnd") ||
        !wcscmp(cls, L"ConsoleWindowClass"))
        return true;
    return (GetWindowLongW(h, GWL_EXSTYLE) & WS_EX_TOOLWINDOW) != 0;
}

static void scanWindows() {
    struct ScanOut { std::vector<std::wstring> titles; std::vector<HWND> hwnds; };
    ScanOut out;
    EnumWindows([](HWND h, LPARAM lp) -> BOOL {
        auto* o = reinterpret_cast<ScanOut*>(lp);
        if (!IsWindowVisible(h) || IsIconic(h)) return TRUE;
        if (h == g_ov) return TRUE;
        if (excludedWindow(h)) return TRUE;
        wchar_t buf[512] = {};
        if (GetWindowTextW(h, buf, 512) <= 0) return TRUE;
        RECT rc{};
        if (!GetClientRect(h, &rc) || rc.right - rc.left < 150 ||
            rc.bottom - rc.top < 100)
            return TRUE;
        o->titles.push_back(buf);
        o->hwnds.push_back(h);
        return TRUE;
    }, reinterpret_cast<LPARAM>(&out));
    if (out.titles.size() > 60) {
        out.titles.resize(60);
        out.hwnds.resize(60);
    }
    g_scanTitles = std::move(out.titles);
    g_scanHwnds = std::move(out.hwnds);
    // probe order: pool-keyword titles first
    std::vector<std::pair<int, int>> scored;  // (index, score)
    for (size_t i = 0; i < g_scanTitles.size(); ++i) {
        std::wstring lt = lower(g_scanTitles[i]);
        int score = 0;
        if (lt.find(L"pool") != std::wstring::npos ||
            lt.find(L"billiard") != std::wstring::npos ||
            lt.find(L"billar") != std::wstring::npos ||
            lt.find(L"biliar") != std::wstring::npos ||
            lt.find(L"8 ball") != std::wstring::npos ||
            lt.find(L"snooker") != std::wstring::npos ||
            lt.find(L"بليارد") != std::wstring::npos)
            score = 8;
        else if (lt.find(L"game") != std::wstring::npos ||
                 lt.find(L"لعبة") != std::wstring::npos)
            score = 3;
        scored.push_back({static_cast<int>(i), score});
    }
    std::stable_sort(scored.begin(), scored.end(),
                     [](const auto& a, const auto& b) {
                         return a.second > b.second;
                     });
    g_scanOrder.clear();
    for (const auto& e : scored) g_scanOrder.push_back(e.first);
    g_probeCursor = 0;
}

// Does a capture of this window contain a plausible pool table?
bool probeWindow(HWND h) {
    pool::Image cap;
    if (!captureClient(h, cap)) return false;
    if (cap.w < 220 || cap.h < 130) return false;
    pool::TableSpec t = pool::TableSpec::standard();
    try {
        pool::detectBalls(cap, {}, t);  // throws when no plausible table
        return true;
    } catch (...) {
        return false;
    }
}

static HWND findGameWindowAuto() {
    for (int tries = 0; tries < 2; ++tries) {
        if (g_scanHwnds.empty() || g_probeCursor >= (int)g_scanOrder.size()) {
            g_probeCursor = 0;
            return nullptr;  // scan exhausted; caller re-scans periodically
        }
        int idx = g_scanOrder[g_probeCursor++];
        if (idx < 0 || idx >= (int)g_scanHwnds.size()) continue;
        HWND h = g_scanHwnds[idx];
        if (!IsWindow(h)) continue;
        if (probeWindow(h)) return h;
    }
    return nullptr;
}

// ---- settings menu (ImGui-style) bound to the live configuration ----
// Float mirrors bind the pure menu model to the overlay's double/int args;
// sync/apply run once per frame so slider edits act immediately.
static mui::Metrics g_m;
static mui::State g_menu;
static std::vector<mui::Tab> g_tabs;
static float mBallR = 0, mTolS = 0.20f, mTolV = 0.30f, mRx0 = 0, mRy0 = 0,
             mRx1 = 1, mRy1 = 1, mFps = 12, mRails = 2, mShots = 1, mTw = 900,
             mTh = 530;
static int mWinIdx = 0, mPreset = 0;
static int mSaveClicks = 0, mRescanClicks = 0, mExitClicks = 0;
static int g_lastWinIdx = 0, g_lastPreset = 0;
static std::string g_gameLabel = "(no game found yet)";
static const char* kPresets[] = {"8 Ball Pool (900x530)",
                                 "Generic 2:1 (1000x500)",
                                 "Compact (800x400)",
                                 "Snooker-ish (1000x480)"};

static pool::TableSpec g_tbl;

static void rebuildTable() {
    g_tbl = pool::TableSpec::standard();
    g_tbl.w = g_args.tw;
    g_tbl.h = g_args.th;
    g_tbl.pockets.clear();
    auto addPk = [&](double x, double y) { g_tbl.pockets.push_back({{x, y}}); };
    addPk(0, 0);
    addPk(g_tbl.w / 2, 0);
    addPk(g_tbl.w, 0);
    addPk(0, g_tbl.h);
    addPk(g_tbl.w / 2, g_tbl.h);
    addPk(g_tbl.w, g_tbl.h);
}

static std::vector<std::string> g_winNames;
static std::vector<std::wstring> g_winTitles;
static std::vector<const char*> g_winOpts;

static void rebuildTabs() {
    int keep = g_menu.activeTab;
    g_tabs.clear();
    mui::Tab aim;
    aim.name = "AIM";
    aim.ctrls.push_back({mui::CT_Check, "Show guides", &g_visible});
    aim.ctrls.push_back(
        {mui::CT_Check, "Aim preview while aiming", &g_args.aimPreview});
    aim.ctrls.push_back(
        {mui::CT_Check, "Trainer lines (all balls)", &g_args.trainer});
    aim.ctrls.push_back(
        {mui::CT_Slider, "Rails", nullptr, &mRails, nullptr, 0.f, 3.f, nullptr,
         0, "%.0f"});
    aim.ctrls.push_back(
        {mui::CT_Slider, "Shots", nullptr, &mShots, nullptr, 1.f, 5.f, nullptr,
         0, "%.0f"});
    g_tabs.push_back(aim);

    mui::Tab det;
    det.name = "DETECT";
    det.ctrls.push_back({mui::CT_Check, "Strip aim line", &g_args.stripLines});
    det.ctrls.push_back({mui::CT_Slider, "Ball radius (0 = auto)", nullptr,
                         &mBallR, nullptr, 0.f, 30.f, nullptr, 0, "%.1f"});
    det.ctrls.push_back({mui::CT_Slider, "Felt tolerance S", nullptr, &mTolS,
                         nullptr, 0.05f, 0.6f});
    det.ctrls.push_back({mui::CT_Slider, "Felt tolerance V", nullptr, &mTolV,
                         nullptr, 0.05f, 0.6f});
    g_tabs.push_back(det);

    mui::Tab tab3;
    tab3.name = "TABLE";
    tab3.ctrls.push_back({mui::CT_Combo, "Preset", nullptr, nullptr, &mPreset,
                          0, 0, kPresets, 4});
    tab3.ctrls.push_back({mui::CT_Check, "Manual table rect", &g_args.manualRect});
    tab3.ctrls.push_back(
        {mui::CT_Slider, "Rect X0", nullptr, &mRx0, nullptr, 0.f, 1.f, nullptr,
         0, "%.3f"});
    tab3.ctrls.push_back(
        {mui::CT_Slider, "Rect Y0", nullptr, &mRy0, nullptr, 0.f, 1.f, nullptr,
         0, "%.3f"});
    tab3.ctrls.push_back(
        {mui::CT_Slider, "Rect X1", nullptr, &mRx1, nullptr, 0.f, 1.f, nullptr,
         0, "%.3f"});
    tab3.ctrls.push_back(
        {mui::CT_Slider, "Rect Y1", nullptr, &mRy1, nullptr, 0.f, 1.f, nullptr,
         0, "%.3f"});
    tab3.ctrls.push_back({mui::CT_Slider, "Table width", nullptr, &mTw, nullptr,
                          400.f, 2000.f, nullptr, 0, "%.0f"});
    tab3.ctrls.push_back({mui::CT_Slider, "Table height", nullptr, &mTh,
                          nullptr, 200.f, 1200.f, nullptr, 0, "%.0f"});
    g_tabs.push_back(tab3);

    mui::Tab game;
    game.name = "GAME";
    game.ctrls.push_back({mui::CT_Combo, "Window", nullptr, nullptr, &mWinIdx,
                          0, 0, g_winOpts.data(),
                          static_cast<int>(g_winOpts.size())});
    game.ctrls.push_back({mui::CT_Button, "Refresh window list", nullptr,
                          nullptr, &mRescanClicks});
    game.ctrls.push_back({mui::CT_Label, g_gameLabel.c_str()});
    g_tabs.push_back(game);

    mui::Tab misc;
    misc.name = "MISC";
    misc.ctrls.push_back(
        {mui::CT_Slider, "FPS", nullptr, &mFps, nullptr, 4.f, 30.f, nullptr, 0,
         "%.0f"});
    misc.ctrls.push_back({mui::CT_Button, "Save settings", nullptr, nullptr,
                          &mSaveClicks});
    misc.ctrls.push_back(
        {mui::CT_Button, "Exit", nullptr, nullptr, &mExitClicks});
    g_tabs.push_back(misc);

    g_menu.activeTab = keep >= 0 && keep < static_cast<int>(g_tabs.size())
                           ? keep
                           : 0;
}

static void rebuildWindowCombo() {
    g_winNames.clear();
    g_winTitles.clear();
    g_winOpts.clear();
    g_winNames.push_back("Auto-detect (any game)");
    g_winOpts.push_back(g_winNames.back().c_str());
    for (const auto& t : g_scanTitles) {
        int wl = WideCharToMultiByte(CP_UTF8, 0, t.c_str(), -1, nullptr, 0,
                                     nullptr, nullptr);
        std::string n(wl > 1 ? wl - 1 : 0, 0);
        if (wl > 1)
            WideCharToMultiByte(CP_UTF8, 0, t.c_str(), -1, n.data(), wl,
                                nullptr, nullptr);
        if (n.size() > 46) n = n.substr(0, 46) + "...";
        g_winTitles.push_back(t);
        g_winNames.push_back(n);
        g_winOpts.push_back(g_winNames.back().c_str());
    }
    // reflect the current selection
    if (g_args.window.empty()) {
        mWinIdx = 0;
    } else {
        mWinIdx = 0;
        for (size_t i = 0; i < g_winTitles.size(); ++i)
            if (g_winTitles[i] == g_args.window) mWinIdx = static_cast<int>(i) + 1;
    }
    g_lastWinIdx = mWinIdx;
    // current-game label (narrowed)
    const std::wstring cur = g_args.window.empty() ? std::wstring(L"(auto-detect)")
                                                   : g_args.window;
    int wl = WideCharToMultiByte(CP_UTF8, 0, cur.c_str(), -1, nullptr, 0,
                                 nullptr, nullptr);
    g_gameLabel = "Current: ";
    if (wl > 1) {
        std::string n(static_cast<size_t>(wl) - 1, 0);
        WideCharToMultiByte(CP_UTF8, 0, cur.c_str(), -1, n.data(), wl, nullptr,
                            nullptr);
        g_gameLabel += n;
    }
    rebuildTabs();
}

static void syncMenuFromArgs() {
    mBallR = static_cast<float>(g_args.ballR);
    mTolS = static_cast<float>(g_args.feltTolS);
    mTolV = static_cast<float>(g_args.feltTolV);
    mRx0 = static_cast<float>(g_args.rectX0);
    mRy0 = static_cast<float>(g_args.rectY0);
    mRx1 = static_cast<float>(g_args.rectX1);
    mRy1 = static_cast<float>(g_args.rectY1);
    mFps = static_cast<float>(g_args.fps);
    mRails = static_cast<float>(g_args.rails);
    mShots = static_cast<float>(g_args.shots);
    mTw = static_cast<float>(g_args.tw);
    mTh = static_cast<float>(g_args.th);
}

static void applyMenuToArgs() {
    g_args.ballR = mBallR;
    g_args.feltTolS = mTolS;
    g_args.feltTolV = mTolV;
    g_args.rectX0 = mRx0;
    g_args.rectY0 = mRy0;
    g_args.rectX1 = mRx1;
    g_args.rectY1 = mRy1;
    g_args.fps = static_cast<int>(mFps);
    g_args.rails = static_cast<int>(mRails);
    g_args.shots = static_cast<int>(mShots);
    if (mPreset != g_lastPreset) {
        g_lastPreset = mPreset;
        switch (mPreset) {
            case 1: mTw = 1000; mTh = 500; break;
            case 2: mTw = 800; mTh = 400; break;
            case 3: mTw = 1000; mTh = 480; break;
            default: mTw = 900; mTh = 530; break;
        }
    }
    if (g_args.tw != mTw || g_args.th != mTh) {
        g_args.tw = mTw;
        g_args.th = mTh;
        rebuildTable();
    }
    if (mWinIdx != g_lastWinIdx) {
        g_lastWinIdx = mWinIdx;
        if (mWinIdx <= 0 || mWinIdx > static_cast<int>(g_winTitles.size()))
            g_args.window.clear();  // auto
        else
            g_args.window = g_winTitles[static_cast<size_t>(mWinIdx) - 1];
        g_game = nullptr;  // re-resolve the target window
        g_forceScan = true;
    }
}

// GDI renderer of the pure menu model (dark ImGui-2019 look).
static COLORREF kPanel = RGB(26, 28, 33), kTitleBar = RGB(45, 91, 142),
                kTabActive = RGB(55, 90, 127), kTrack = RGB(42, 46, 54),
                kFill = RGB(62, 110, 170), kBox = RGB(18, 20, 24),
                kCheckC = RGB(66, 133, 244), kTextC = RGB(240, 240, 245),
                kDim = RGB(185, 190, 200), kBorder = RGB(70, 75, 85),
                kPop = RGB(32, 35, 41);

static void fillR(HDC dc, const mui::Rect& r, COLORREF c) {
    RECT rc = {static_cast<LONG>(r.x), static_cast<LONG>(r.y),
               static_cast<LONG>(r.x + r.w), static_cast<LONG>(r.y + r.h)};
    HBRUSH b = CreateSolidBrush(c);
    FillRect(dc, &rc, b);
    DeleteObject(b);
}

static void frameR(HDC dc, const mui::Rect& r, COLORREF c) {
    RECT rc = {static_cast<LONG>(r.x), static_cast<LONG>(r.y),
               static_cast<LONG>(r.x + r.w - 1), static_cast<LONG>(r.y + r.h - 1)};
    HBRUSH b = CreateSolidBrush(c);
    FrameRect(dc, &rc, b);
    DeleteObject(b);
}

static void textR(HDC dc, const mui::Rect& r, const wchar_t* t, COLORREF c,
                  UINT flags = DT_SINGLELINE | DT_LEFT | DT_VCENTER) {
    RECT rc = {static_cast<LONG>(r.x), static_cast<LONG>(r.y),
               static_cast<LONG>(r.x + r.w), static_cast<LONG>(r.y + r.h)};
    SetTextColor(dc, c);
    DrawTextW(dc, t, -1, &rc, flags);
}

static wchar_t* widen(const char* s, wchar_t* buf, int cap) {
    MultiByteToWideChar(CP_UTF8, 0, s, -1, buf, cap);
    return buf;
}

static void drawMenu(HDC dc) {
    if (!g_menu.open || g_tabs.empty()) return;
    static HFONT fUi = CreateFontW(-15, 0, 0, 0, FW_NORMAL, 0, 0, 0,
                                   DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                                   CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                                   DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
    const mui::Tab& t = g_tabs[static_cast<size_t>(g_menu.activeTab)];
    mui::Rect pr = mui::panelRect(g_m, g_menu, t);
    fillR(dc, pr, kPanel);
    frameR(dc, pr, kBorder);

    mui::Rect tr = mui::titleRect(g_m, g_menu);
    fillR(dc, tr, kTitleBar);
    textR(dc, {tr.x + 12, tr.y, tr.w - 50, tr.h}, L"Settings",
          RGB(255, 255, 255));
    mui::Rect cr = mui::closeRect(g_m, g_menu);
    fillR(dc, cr, RGB(35, 42, 55));
    textR(dc, cr, L"X", RGB(255, 255, 255),
          DT_SINGLELINE | DT_CENTER | DT_VCENTER);

    wchar_t wname[64];
    for (int i = 0; i < static_cast<int>(g_tabs.size()); ++i) {
        mui::Rect r = mui::tabRect(g_m, g_menu, i);
        bool act = i == g_menu.activeTab;
        fillR(dc, r, act ? kTabActive : RGB(36, 39, 46));
        widen(g_tabs[static_cast<size_t>(i)].name, wname, 64);
        textR(dc, r, wname, act ? RGB(255, 255, 255) : kDim,
              DT_SINGLELINE | DT_CENTER | DT_VCENTER);
    }

    HGDIOBJ oldFont = SelectObject(dc, fUi);
    SetBkMode(dc, TRANSPARENT);
    wchar_t wlabel[160], wtmp[96];
    for (int i = 0; i < static_cast<int>(t.ctrls.size()); ++i) {
        const mui::Ctrl& c = t.ctrls[static_cast<size_t>(i)];
        mui::Rect rr = mui::rowRect(g_m, g_menu, i);
        widen(c.label, wlabel, 160);
        if (c.type == mui::CT_Check) {
            mui::Rect box = {rr.x + 2, rr.y + (rr.h - 16) / 2, 16, 16};
            fillR(dc, box, kBox);
            frameR(dc, box, kBorder);
            if (c.bval && *c.bval) {
                HPEN pen = CreatePen(PS_SOLID, 2, kCheckC);
                HGDIOBJ op = SelectObject(dc, pen);
                MoveToEx(dc, static_cast<int>(box.x) + 3,
                         static_cast<int>(box.y) + 8, nullptr);
                LineTo(dc, static_cast<int>(box.x) + 6,
                       static_cast<int>(box.y) + 12);
                LineTo(dc, static_cast<int>(box.x) + 13,
                       static_cast<int>(box.y) + 4);
                SelectObject(dc, op);
                DeleteObject(pen);
            }
            textR(dc, {rr.x + 28, rr.y, rr.w - 30, rr.h}, wlabel, kTextC);
        } else if (c.type == mui::CT_Slider) {
            mui::Rect trk = mui::trackRect(g_m, g_menu, i);
            fillR(dc, trk, kTrack);
            float frac =
                c.fval ? mui::clampf((*c.fval - c.fmin) / (c.fmax - c.fmin),
                                     0.f, 1.f)
                       : 0.f;
            fillR(dc, {trk.x, trk.y, trk.w * frac, trk.h}, kFill);
            char vb[32] = {};
            if (c.fval) std::snprintf(vb, 32, c.fmt, *c.fval);
            textR(dc, trk, widen(vb, wtmp, 96), RGB(255, 255, 255),
                  DT_SINGLELINE | DT_CENTER | DT_VCENTER);
            textR(dc, {trk.x + trk.w + 10, rr.y, rr.w - trk.w - 12, rr.h},
                  wlabel, kDim);
        } else if (c.type == mui::CT_Combo) {
            mui::Rect trk = mui::trackRect(g_m, g_menu, i);
            fillR(dc, trk, kTrack);
            int idx = c.ival ? *c.ival : 0;
            if (idx < 0 || idx >= c.nopt) idx = 0;
            const char* curTxt = c.nopt > 0 ? c.options[idx] : "";
            textR(dc, {trk.x + 8, trk.y, trk.w - 26, trk.h},
                  widen(curTxt, wtmp, 96), kTextC);
            POINT tri[3] = {
                {static_cast<LONG>(trk.x + trk.w - 15),
                 static_cast<LONG>(trk.y + trk.h / 2 - 3)},
                {static_cast<LONG>(trk.x + trk.w - 6),
                 static_cast<LONG>(trk.y + trk.h / 2 - 3)},
                {static_cast<LONG>(trk.x + trk.w - 10.5f),
                 static_cast<LONG>(trk.y + trk.h / 2 + 4)}};
            HBRUSH b = CreateSolidBrush(RGB(215, 220, 230));
            HPEN pn = CreatePen(PS_SOLID, 1, RGB(215, 220, 230));
            HGDIOBJ ob = SelectObject(dc, b);
            HGDIOBJ opn = SelectObject(dc, pn);
            Polygon(dc, tri, 3);
            SelectObject(dc, ob);
            SelectObject(dc, opn);
            DeleteObject(b);
            DeleteObject(pn);
            textR(dc, {trk.x + trk.w + 10, rr.y, rr.w - trk.w - 12, rr.h},
                  wlabel, kDim);
            if (g_menu.comboTab == g_menu.activeTab && g_menu.comboCtrl == i) {
                mui::Rect pop = mui::popRect(g_m, g_menu, i, c.nopt);
                fillR(dc, pop, kPop);
                frameR(dc, pop, kBorder);
                for (int o = 0; o < c.nopt; ++o) {
                    mui::Rect orow = {pop.x, pop.y + o * g_m.rowH, pop.w,
                                      g_m.rowH};
                    textR(dc, {orow.x + 10, orow.y, orow.w - 10, orow.h},
                          widen(c.options[o], wtmp, 96),
                          o == idx ? RGB(120, 180, 255) : kTextC);
                }
            }
        } else if (c.type == mui::CT_Button) {
            mui::Rect br = {rr.x + 2, rr.y + 3, rr.w * 0.72f, rr.h - 6};
            fillR(dc, br, kTabActive);
            textR(dc, br, wlabel, RGB(255, 255, 255),
                  DT_SINGLELINE | DT_CENTER | DT_VCENTER);
        } else {
            textR(dc, {rr.x + 4, rr.y, rr.w, rr.h}, wlabel, kDim);
        }
    }
    SelectObject(dc, oldFont);
}

void fillKey(HDC dc, const RECT& rc) {
    HBRUSH bg = CreateSolidBrush(kKey);
    FillRect(dc, &rc, bg);
    DeleteObject(bg);
}

void clearOverlay() {
    if (!g_ov) return;
    RECT rc;
    GetClientRect(g_ov, &rc);
    HDC dc = beginFrame(rc);
    if (dc) presentFrame(rc);
}

// Draw the best shots' guides in client pixel coordinates. The overlay is
// exactly the game client size, so client coords map 1:1.
static void drawArrow(HDC dc, POINT a, POINT b, COLORREF col, int wdt) {
    HPEN pen = CreatePen(PS_SOLID, wdt, col);
    HGDIOBJ old = SelectObject(dc, pen);
    MoveToEx(dc, a.x, a.y, nullptr);
    LineTo(dc, b.x, b.y);
    double ang = std::atan2(static_cast<double>(b.y - a.y),
                            static_cast<double>(b.x - a.x));
    const double len = std::max(7.0, wdt * 5.0);
    const double spread = 0.42;
    for (double sg : {-spread, spread}) {
        double ex = b.x - len * std::cos(ang + sg);
        double ey = b.y - len * std::sin(ang + sg);
        MoveToEx(dc, b.x, b.y, nullptr);
        LineTo(dc, static_cast<LONG>(ex), static_cast<LONG>(ey));
    }
    SelectObject(dc, old);
    DeleteObject(pen);
}

// Trainer-style drawing (the classic 8BP trainer look):
//   - table border + ball circles,
//   - thin guide lines for EVERY ball's best shot (cue->ghost, ghost->pocket
//     with an arrowhead), ghost rings,
//   - the best 3 shots drawn thick: full cue path incl. cushion bounces,
//     ghost circle, object path, pocket ring and a tangent arrow.
// All mapping goes through the perspective quad when the felt was fitted as
// a trapezoid, so lines land on the drawn table.
void drawGuides(const pool::DetectResult& det, const std::vector<pool::Shot>& shots,
                int topN, HDC dc) {
    if (!g_ov || !dc) return;

    pool::Rect f = det.feltRect;  // same +/-2 shrink detection mapped with
    f.x0 += 2; f.y0 += 2; f.x1 -= 2; f.y1 -= 2;
    const pool::TableSpec& tbl = det.table;
    auto toPx = [&](pool::Vec2 l) -> POINT {
        if (det.hasQuad) {
            pool::Vec2 p = pool::quadLogicalToPx(det.quadPx, l, tbl.w, tbl.h);
            return POINT{static_cast<LONG>(p.x), static_cast<LONG>(p.y)};
        }
        return POINT{static_cast<LONG>(f.x0 + l.x / tbl.w * f.w()),
                     static_cast<LONG>(f.y0 + l.y / tbl.h * f.h())};
    };
    auto scaleW = [&]() {
        return det.hasQuad
                   ? ((det.quadPx.tr.x - det.quadPx.tl.x) +
                      (det.quadPx.br.x - det.quadPx.bl.x)) / 2
                   : f.w();
    };
    const int ballPx = std::max(3, static_cast<int>(tbl.ballR * scaleW() / tbl.w));

    // Table border (detected felt outline, perspective-corrected).
    {
        HPEN pen = CreatePen(PS_SOLID, 2, RGB(110, 195, 255));
        HGDIOBJ oldPen = SelectObject(dc, pen);
        HGDIOBJ oldBr = SelectObject(dc, GetStockObject(NULL_BRUSH));
        if (det.hasQuad) {
            POINT q[4] = {toPx({0, 0}), toPx({tbl.w, 0}), toPx({tbl.w, tbl.h}),
                          toPx({0, tbl.h})};
            Polygon(dc, q, 4);
        } else {
            Rectangle(dc, static_cast<LONG>(f.x0), static_cast<LONG>(f.y0),
                      static_cast<LONG>(f.x1), static_cast<LONG>(f.y1));
        }
        SelectObject(dc, oldBr);
        SelectObject(dc, oldPen);
        DeleteObject(pen);
    }
    // Ball circles at the detected positions with the detected sizes.
    // Merged cluster blobs report an inflated blob radius, so cap it at the
    // table-scaled ball radius to keep every circle on the actual balls.
    for (const auto& d : det.balls) {
        double rCap = det.ballRPx * 1.30;
        int r = std::max(3, static_cast<int>(std::min(d.rPx, rCap)));
        HPEN pen = CreatePen(
            PS_SOLID, 1, d.kind == "cue" ? RGB(255, 255, 255) : RGB(165, 205, 235));
        HGDIOBJ oldPen = SelectObject(dc, pen);
        HGDIOBJ oldBr = SelectObject(dc, GetStockObject(NULL_BRUSH));
        Ellipse(dc, static_cast<LONG>(d.px.x - r), static_cast<LONG>(d.px.y - r),
                static_cast<LONG>(d.px.x + r), static_cast<LONG>(d.px.y + r));
        SelectObject(dc, oldBr);
        SelectObject(dc, oldPen);
        DeleteObject(pen);
    }

    // Classic trainer lines: the best shot for EVERY object ball, thin.
    // Direct shots preferred over banks, banks over kicks; shots arrive
    // sorted by confidence so the first seen per ball is its best.
    if (g_args.trainer && g_visible && shots.size() >= 1 &&
        det.balls.size() >= 2) {
        auto typeRank = [](const pool::Shot& x) {
            return x.typeName == "direct" ? 0 : (x.typeName == "bank" ? 1 : 2);
        };
        std::map<int, const pool::Shot*> perBall;
        for (const auto& s : shots) {
            auto it = perBall.find(s.objectIdx);
            if (it == perBall.end()) {
                perBall[s.objectIdx] = &s;
                continue;
            }
            if (typeRank(s) < typeRank(*it->second)) it->second = &s;
        }
        const COLORREF cueThin = RGB(205, 230, 255);
        const COLORREF objThin = RGB(255, 215, 80);
        const COLORREF kickThin = RGB(95, 220, 255);
        const COLORREF ringDim = RGB(150, 180, 210);
        for (const auto& kv : perBall) {
            const pool::Shot& s = *kv.second;
            // Cue -> ghost (thin; cyan for kicks).
            for (size_t k = 0; k + 1 < s.cuePath.size(); ++k)
                drawArrow(dc, toPx(s.cuePath[k]), toPx(s.cuePath[k + 1]),
                          s.typeName == "kick" ? kickThin : cueThin, 1);
            // Ghost ring (thin, dim).
            if (s.typeName != "kick") {
                POINT g = toPx(s.ghost);
                HPEN pen = CreatePen(PS_SOLID, 1, ringDim);
                HGDIOBJ oldPen = SelectObject(dc, pen);
                HGDIOBJ oldBr = SelectObject(dc, GetStockObject(NULL_BRUSH));
                Ellipse(dc, g.x - ballPx, g.y - ballPx, g.x + ballPx,
                        g.y + ballPx);
                SelectObject(dc, oldBr);
                SelectObject(dc, oldPen);
                DeleteObject(pen);
            }
            // Ghost -> pocket, arrowhead at the pocket end (thin yellow).
            for (size_t k = 0; k + 1 < s.objPath.size(); ++k) {
                POINT a = toPx(s.objPath[k]), b = toPx(s.objPath[k + 1]);
                if (k + 2 == s.objPath.size()) {
                    drawArrow(dc, a, b, objThin, 1);
                } else {
                    HPEN pen = CreatePen(PS_SOLID, 1, objThin);
                    HGDIOBJ oldPen = SelectObject(dc, pen);
                    MoveToEx(dc, a.x, a.y, nullptr);
                    LineTo(dc, b.x, b.y);
                    SelectObject(dc, oldPen);
                    DeleteObject(pen);
                }
            }
        }
    }

    if (g_visible && shots.size() >= 1 && det.balls.size() >= 1) {
        const int nDraw =
            std::min<int>({topN, 3, static_cast<int>(shots.size())});
        const COLORREF cueCol[3] = {RGB(255, 255, 255), RGB(190, 190, 190),
                                    RGB(130, 130, 130)};
        for (int rank = nDraw - 1; rank >= 0; --rank) {
            const pool::Shot& s = shots[rank];
            const COLORREF col = rank < 3 ? cueCol[rank] : RGB(90, 90, 90);
            const int wdt = rank == 0 ? 3 : 2;
            HPEN pen = CreatePen(PS_SOLID, wdt, col);
            HGDIOBJ oldPen = SelectObject(dc, pen);
            // Cue ball path (through any bounces) to the ghost position.
            for (size_t k = 0; k + 1 < s.cuePath.size(); ++k) {
                POINT a = toPx(s.cuePath[k]), b = toPx(s.cuePath[k + 1]);
                MoveToEx(dc, a.x, a.y, nullptr);
                LineTo(dc, b.x, b.y);
            }
            // Ghost ball outline at contact.
            if (s.typeName != "kick") {
                POINT g = toPx(s.ghost);
                Ellipse(dc, g.x - ballPx, g.y - ballPx, g.x + ballPx, g.y + ballPx);
            }
            // Object ball path to the pocket (yellow).
            HPEN ypen = CreatePen(PS_SOLID, wdt, RGB(255, 255, 0));
            SelectObject(dc, ypen);
            for (size_t k = 0; k + 1 < s.objPath.size(); ++k) {
                POINT a = toPx(s.objPath[k]), b = toPx(s.objPath[k + 1]);
                MoveToEx(dc, a.x, a.y, nullptr);
                LineTo(dc, b.x, b.y);
            }
            if (s.pocketIdx >= 0 && s.pocketIdx < static_cast<int>(tbl.pockets.size())) {
                POINT p = toPx(tbl.pockets[s.pocketIdx].pos);
                HGDIOBJ oldBr = SelectObject(dc, GetStockObject(NULL_BRUSH));
                HPEN rpen = CreatePen(PS_SOLID, 2, RGB(255, 70, 70));
                HGDIOBJ oldPen2 = SelectObject(dc, rpen);
                Ellipse(dc, p.x - 7, p.y - 7, p.x + 7, p.y + 7);
                SelectObject(dc, oldPen2);
                DeleteObject(rpen);
                HBRUSH yb = CreateSolidBrush(RGB(255, 255, 0));
                oldBr = SelectObject(dc, yb);
                Ellipse(dc, p.x - 3, p.y - 3, p.x + 3, p.y + 3);
                SelectObject(dc, oldBr);
            }
            // Tangent arrow: where the cue ball departs after contact.
            if (s.typeName == "direct") {
                POINT g = toPx(s.ghost);
                const double len = ballPx * 2.6;
                POINT t{g.x + static_cast<LONG>(s.tangentDir.x * len),
                        g.y + static_cast<LONG>(s.tangentDir.y * len)};
                drawArrow(dc, g, t, RGB(220, 235, 255), 1);
            }
            SelectObject(dc, oldPen);
            DeleteObject(pen);
            DeleteObject(ypen);
        }
    }
    ReleaseDC(g_ov, dc);
}

std::wstring g_status;  // one-line HUD text, updated by the analysis loop

void drawHud(HDC dc, const RECT& crc) {
    if (!g_visible || g_status.empty() || !dc) return;
    RECT rc = crc;
    rc.bottom = rc.top + 22;
    RECT bg = {0, 0, rc.right, rc.bottom};
    HBRUSH br = CreateSolidBrush(RGB(0, 0, 0));
    FillRect(dc, &bg, br);
    DeleteObject(br);
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, RGB(255, 255, 120));
    DrawTextW(dc, g_status.c_str(), -1, &rc,
              DT_SINGLELINE | DT_LEFT | DT_VCENTER);
}

// Live simulation preview: the game's own aim guideline was detected, so show
// what the CURRENT aim will do — every ball's predicted path, the rest
// positions and a pocket flash (yellow = object potted, red = scratch).
void drawSimPreview(const pool::DetectResult& det, const std::vector<pool::Ball>& balls,
                    const pool::SimResult& sim, pool::Vec2 dirPx, HDC dc) {
    if (!g_ov || !dc) return;
    if (g_visible && !sim.final.empty() && balls.size() == sim.final.size()) {
        pool::Rect f = det.feltRect;  // same +/-2 shrink detection mapped with
        f.x0 += 2; f.y0 += 2; f.x1 -= 2; f.y1 -= 2;
        const pool::TableSpec& tbl = det.table;
        auto toPx = [&](pool::Vec2 l) {
            if (det.hasQuad) {
                pool::Vec2 p =
                    pool::quadLogicalToPx(det.quadPx, l, tbl.w, tbl.h);
                return POINT{static_cast<LONG>(p.x), static_cast<LONG>(p.y)};
            }
            return POINT{static_cast<LONG>(f.x0 + l.x / tbl.w * f.w()),
                         static_cast<LONG>(f.y0 + l.y / tbl.h * f.h())};
        };
        // Extend the aim ray from the cue ball in pixel space.
        for (size_t i = 0; i < balls.size(); ++i) {
            if (balls[i].id != 0) continue;
            POINT c{static_cast<LONG>(det.balls[i].px.x),
                    static_cast<LONG>(det.balls[i].px.y)};
            const double len = std::hypot(f.w(), f.h());
            HPEN pen = CreatePen(PS_SOLID, 2, RGB(255, 255, 255));
            HGDIOBJ old = SelectObject(dc, pen);
            MoveToEx(dc, c.x, c.y, nullptr);
            LineTo(dc, c.x + static_cast<LONG>(dirPx.x * len),
                   c.y + static_cast<LONG>(dirPx.y * len));
            SelectObject(dc, old);
            DeleteObject(pen);
            break;
        }
        // Predicted paths (thin) + rest positions (hollow rings).
        const COLORREF pal[6] = {RGB(255, 255, 255), RGB(255, 210, 40),
                                 RGB(60, 200, 255),  RGB(120, 255, 120),
                                 RGB(255, 120, 80),  RGB(230, 120, 255)};
        for (size_t i = 0; i < sim.paths.size(); ++i) {
            const COLORREF col = pal[i % 6];
            HPEN pen = CreatePen(PS_SOLID, 1, col);
            HGDIOBJ old = SelectObject(dc, pen);
            for (size_t k = 0; k + 1 < sim.paths[i].size(); ++k) {
                POINT a = toPx(sim.paths[i][k]), b = toPx(sim.paths[i][k + 1]);
                MoveToEx(dc, a.x, a.y, nullptr);
                LineTo(dc, b.x, b.y);
            }
            SelectObject(dc, old);
            DeleteObject(pen);
            if (sim.final[i].active) {
                POINT p = toPx(sim.final[i].pos);
                HPEN ring = CreatePen(PS_SOLID, 1, col);
                HGDIOBJ o2 = SelectObject(dc, ring);
                HGDIOBJ ob = SelectObject(dc, GetStockObject(NULL_BRUSH));
                Ellipse(dc, p.x - 6, p.y - 6, p.x + 6, p.y + 6);
                SelectObject(dc, ob);
                SelectObject(dc, o2);
            }
        }
        // Pocket flashes for potted balls.
        for (const auto& e : sim.events) {
            if (e.kind != "pot") continue;
            if (e.other < 0 || e.other >= static_cast<int>(tbl.pockets.size()))
                continue;
            POINT p = toPx(tbl.pockets[e.other].pos);
            HBRUSH br = CreateSolidBrush(e.ball == 0 ? RGB(255, 60, 60)
                                                     : RGB(255, 255, 0));
            HGDIOBJ ob = SelectObject(dc, br);
            Ellipse(dc, p.x - 7, p.y - 7, p.x + 7, p.y + 7);
            SelectObject(dc, ob);
            DeleteObject(br);
        }
    }
    ReleaseDC(g_ov, dc);
}

}  // namespace

int main() {
    SetProcessDPIAware();  // physical pixels everywhere (DPI-safe capture)
    g_logPath = iniPath();
    g_args = parseArgs();
    if (!g_args.window.empty() && g_args.window.size() > 200) g_args.window.resize(200);

    WNDCLASSW wc{};
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"PoolVisionOverlay";
    wc.hbrBackground = CreateSolidBrush(kKey);
    RegisterClassW(&wc);

    // No WS_EX_TRANSPARENT: the settings menu needs clicks. Click-through
    // elsewhere is restored dynamically every frame (see loop) so the game
    // keeps receiving input unless the cursor is over the panel.
    g_ov = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        wc.lpszClassName, L"", WS_POPUP, 0, 0, 100, 100, nullptr, nullptr,
        wc.hInstance, nullptr);
    if (!g_ov) {
        MessageBoxW(nullptr, L"Cannot create the overlay window.",
                    L"pool-overlay", MB_ICONERROR);
        return 1;
    }
    SetLayeredWindowAttributes(g_ov, kKey, 0, LWA_COLORKEY);
    ShowWindow(g_ov, SW_SHOWNOACTIVATE);

    // Logical table: pockets follow the (menu-adjustable) custom size.
    rebuildTable();
    rebuildWindowCombo();

    {
        char wbuf[512] = {};
        WideCharToMultiByte(CP_UTF8, 0,
                            g_args.window.empty() ? L"(auto)" : g_args.window.c_str(),
                            -1, wbuf, 512, nullptr, nullptr);
        LOGF("pool-overlay started. window=\"%s\" table=%.0fx%.0f rails=%d "
             "shots=%d fps=%d",
             wbuf, g_args.tw, g_args.th, g_args.rails, g_args.shots,
             g_args.fps);
    }

    int frame = 0;
    bool prevDown = false;
    for (;;) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) return 0;
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        // ---- settings menu, fully mouse-driven (no hotkeys) ----
        if (g_forceScan || frame % 60 == 0) {
            g_forceScan = false;
            scanWindows();
            rebuildWindowCombo();
        }
        {
            bool down = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
            POINT mp{};
            GetCursorPos(&mp);
            if (g_ov) ScreenToClient(g_ov, &mp);
            float mx = static_cast<float>(mp.x), my = static_cast<float>(mp.y);
            // Click-through everywhere except when the cursor hovers the
            // panel (sticky while dragging): the game keeps its input.
            bool sticky = g_menu.draggingPanel || g_menu.dragSlider;
            bool overPanel = false;
            if (g_menu.open && !g_menuHidden && !g_tabs.empty()) {
                mui::Rect pr = mui::panelRect(
                    g_m, g_menu, g_tabs[static_cast<size_t>(g_menu.activeTab)]);
                overPanel = pr.contains(mx - 4, my - 4) ||
                            pr.contains(mx + 4, my + 4);
            }
            LONG ex = GetWindowLongW(g_ov, GWL_EXSTYLE);
            LONG want = (overPanel || sticky)
                            ? (ex & ~WS_EX_TRANSPARENT)
                            : (ex | WS_EX_TRANSPARENT);
            if (want != ex) SetWindowLongW(g_ov, GWL_EXSTYLE, want);
            if (down && !prevDown && !g_menuHidden) {
                mui::HitResult hr = mui::hitTest(g_m, g_menu, g_tabs, mx, my);
                if (mui::press(g_m, g_menu, g_tabs, mx, my) &&
                    hr.hit == mui::H_Ctrl && hr.ctrl >= 0 &&
                    hr.ctrl < static_cast<int>(
                                  g_tabs[static_cast<size_t>(g_menu.activeTab)]
                                      .ctrls.size())) {
                    const mui::Ctrl& c =
                        g_tabs[static_cast<size_t>(g_menu.activeTab)]
                            .ctrls[static_cast<size_t>(hr.ctrl)];
                    if (c.type == mui::CT_Button) {
                        if (c.ival == &mSaveClicks) saveIni();
                        else if (c.ival == &mRescanClicks) g_forceScan = true;
                        else if (c.ival == &mExitClicks) g_running = false;
                    }
                }
            } else if (down && prevDown) {
                mui::drag(g_m, g_menu, g_tabs, mx, my);
            } else {
                mui::release(g_menu);
            }
            prevDown = down;
            syncMenuFromArgs();
            applyMenuToArgs();
        }
        if (!g_running) break;
        // Live tuning hotkeys ("edit anything").
        if (GetAsyncKeyState(VK_F10) & 1) saveIni();
        if (GetAsyncKeyState(VK_F7) & 1) {
            g_args.ballR = g_args.ballR > 3.5 ? g_args.ballR - 0.5 : 0;
            g_args.manualRect = false;
            std::printf("[i] ballR=%.1f (0=auto)\n", g_args.ballR);
        }
        if (GetAsyncKeyState(VK_F8) & 1) {
            g_args.ballR += 0.5;
            std::printf("[i] ballR=%.1f\n", g_args.ballR);
        }
        if (GetAsyncKeyState(VK_OEM_4) & 1) {  // '['
            g_args.feltTolV = std::max(0.05, g_args.feltTolV - 0.02);
            std::printf("[i] feltTolV=%.2f\n", g_args.feltTolV);
        }
        if (GetAsyncKeyState(VK_OEM_6) & 1) {  // ']'
            g_args.feltTolV = std::min(0.60, g_args.feltTolV + 0.02);
            std::printf("[i] feltTolV=%.2f\n", g_args.feltTolV);
        }
        if (GetAsyncKeyState('A') & 1) {
            g_args.aimPreview = !g_args.aimPreview;
            std::printf("[i] aim preview %s\n", g_args.aimPreview ? "ON" : "OFF");
        }
        if (GetAsyncKeyState('R') & 1) {
            g_args.manualRect = !g_args.manualRect;
            std::printf("[i] manual rect %s\n", g_args.manualRect ? "ON" : "OFF");
        }
        if (GetAsyncKeyState(VK_LEFT) & 1) g_args.rectX0 += 0.005;
        if (GetAsyncKeyState(VK_RIGHT) & 1) g_args.rectX0 -= 0.005;
        if (GetAsyncKeyState(VK_UP) & 1) g_args.rectY0 += 0.005;
        if (GetAsyncKeyState(VK_DOWN) & 1) g_args.rectY0 -= 0.005;
        for (int n = '1'; n <= '5'; ++n)
            if (GetAsyncKeyState(n) & 1) {
                g_args.shots = n - '0';
                std::printf("[i] shots=%d\n", g_args.shots);
            }
        if (GetAsyncKeyState('D') & 1) {
            std::printf("[i] dump: ballR=%.1f tolS=%.2f tolV=%.2f strip=%d "
                        "manual=%d rect=[%.3f,%.3f,%.3f,%.3f]\n",
                        g_args.ballR, g_args.feltTolS, g_args.feltTolV,
                        g_args.stripLines ? 1 : 0, g_args.manualRect ? 1 : 0,
                        g_args.rectX0, g_args.rectY0, g_args.rectX1, g_args.rectY1);
        }
        if (GetAsyncKeyState(VK_INSERT) & 1) {
            g_menuHidden = !g_menuHidden;
            std::printf("[i] UI %s (Insert)\n",
                        g_menuHidden ? "hidden" : "shown");
        }
        if (GetAsyncKeyState('X') & 1) {
            std::printf("[x] exit (X pressed)\n");
            saveIni();
            g_running = false;
        }

        if (!g_game || !IsWindow(g_game) || IsIconic(g_game)) {
            g_game = nullptr;
            if (!g_args.window.empty()) {
                g_game = findGameWindow(lower(g_args.window));
            } else if (frame % 2 == 0) {
                g_game = findGameWindowAuto();  // one capture probe per tick
            }
        }
        // Compose the entire frame in memory, present once (no flicker).
        RECT orc{};
        HDC fdc = nullptr;
        if (g_ov) {
            GetClientRect(g_ov, &orc);
            fdc = beginFrame(orc);
        }
        HWND game = g_game;
        if (game) {
            RECT rc;
            GetClientRect(game, &rc);
            POINT pt{};
            ClientToScreen(game, &pt);
            // Reposition only when the game rect really changed: a per-frame
            // SetWindowPos churns the DWM and reads as shaking.
            static RECT s_tgt = {};
            static bool s_tgtValid = false;
            RECT tgt = {pt.x, pt.y, pt.x + (rc.right - rc.left),
                        pt.y + (rc.bottom - rc.top)};
            if (!s_tgtValid || tgt.left != s_tgt.left || tgt.top != s_tgt.top ||
                tgt.right != s_tgt.right || tgt.bottom != s_tgt.bottom) {
                SetWindowPos(g_ov, HWND_TOPMOST, tgt.left, tgt.top,
                             tgt.right - tgt.left, tgt.bottom - tgt.top,
                             SWP_NOACTIVATE | SWP_SHOWWINDOW);
                s_tgt = tgt;
                s_tgtValid = true;
            }

            pool::Image cap;
            bool analyzed = false;
            if (captureClient(game, cap)) {
                try {
                    pool::DetectOptions dopt;
                    dopt.feltTolS = g_args.feltTolS;
                    dopt.feltTolV = g_args.feltTolV;
                    dopt.stripLines = g_args.stripLines;
                    if (g_args.ballR > 1) dopt.forceBallR = g_args.ballR;
                    if (g_args.manualRect && g_args.rectX1 > g_args.rectX0 &&
                        g_args.rectY1 > g_args.rectY0)
                        dopt.forceRect = pool::Rect{
                            g_args.rectX0 * cap.w, g_args.rectY0 * cap.h,
                            g_args.rectX1 * cap.w, g_args.rectY1 * cap.h};
                    pool::DetectResult det = pool::detectBalls(cap, dopt, g_tbl);

                    // Temporal stabilization: glide the felt geometry and the
                    // ball positions so the guides stop trembling between
                    // frames. Real moves snap instantly (smooth* helpers).
                    if (game != g_stabGame) {
                        g_stabGame = game;
                        g_stabValid = false;
                        g_stabHasQuad = false;
                        g_stabRPx = 0;
                        g_ballStab.reset();
                    }
                    if (g_stabValid) {
                        det.feltRect = pool::smoothRect(g_stabRect, det.feltRect,
                                                        0.45, 0.03);
                        if (det.hasQuad && g_stabHasQuad)
                            det.quadPx = pool::smoothQuad(g_stabQuad, det.quadPx,
                                                          0.45, 0.03);
                        det.ballRPx = pool::smoothScalar(g_stabRPx, det.ballRPx,
                                                         0.30);
                    }
                    g_stabRect = det.feltRect;
                    if (det.hasQuad) {
                        g_stabQuad = det.quadPx;
                        g_stabHasQuad = true;
                    }
                    g_stabRPx = det.ballRPx;
                    g_stabValid = true;
                    g_ballStab.apply(det.balls, det.ballRPx);
                    std::vector<pool::Ball> balls;
                    int cueIdx = -1;
                    for (size_t i = 0; i < det.balls.size(); ++i) {
                        const auto& d = det.balls[i];
                        pool::Ball b;
                        b.id = d.number;
                        b.kind = d.kind;
                        b.color = d.color;
                        b.pos = d.logical;
                        b.r = det.table.ballR;
                        if (d.kind == "cue" && cueIdx < 0) cueIdx = static_cast<int>(i);
                        balls.push_back(b);
                    }
                    // Aim-aware preview: while the player drags, the game draws
                    // its white guideline — predict the CURRENT aim's outcome
                    // with the physics sim. Idle frames show best-shot guides.
                    bool drewSim = false;
                    int aimPotted = 0;
                    bool aimScratch = false;
                    if (g_args.aimPreview && cueIdx >= 0 && balls.size() >= 2) {
                        std::vector<pool::Vec2> otherPx;
                        for (size_t i = 0; i < det.balls.size(); ++i)
                            if (static_cast<int>(i) != cueIdx)
                                otherPx.push_back(det.balls[i].px);
                        pool::Vec2 dirPx;
                        if (pool::estimateAimDirPx(cap, det.balls[cueIdx].px,
                                                   det.ballRPx, otherPx, dirPx)) {
                            pool::Vec2 dirL{dirPx.x, dirPx.y};
                            {
                                // Logical aim direction: map two pixel
                                // points through the felt quad (or the rect
                                // affine) and normalize the difference.
                                pool::Vec2 cuePx = det.balls[cueIdx].px;
                                pool::Vec2 tipPx{cuePx.x + dirPx.x * 40.0,
                                                 cuePx.y + dirPx.y * 40.0};
                                auto toLog = [&](pool::Vec2 p) {
                                    if (det.hasQuad)
                                        return pool::quadPxToLogical(
                                            det.quadPx, p, det.table.w,
                                            det.table.h);
                                    pool::Rect fr = det.feltRect;
                                    fr.x0 += 2; fr.y0 += 2;
                                    fr.x1 -= 2; fr.y1 -= 2;
                                    return pool::Vec2{
                                        (p.x - fr.x0) / fr.w() * det.table.w,
                                        (p.y - fr.y0) / fr.h() * det.table.h};
                                };
                                dirL = toLog(tipPx) - toLog(cuePx);
                            }
                            dirL = dirL.len() > 1e-12 ? dirL.norm()
                                                      : pool::Vec2{1, 0};
                            double power = pool::minPotPower(det.table, balls,
                                                             dirL, 0, 0);
                            if (power < 0) power = 1.0;
                            pool::SimResult sim = pool::simulate(
                                det.table, balls,
                                dirL * (pool::PhysicsParams{}.maxShotSpeed * power),
                                0, 0);
                            drawSimPreview(det, balls, sim, dirPx, fdc);
                            aimPotted = static_cast<int>(sim.potted.size());
                            aimScratch = sim.cueScratch;
                            drewSim = true;
                            if (frame % 24 == 0)
                                LOGF("[aim] deg=%.0f power=%.2f potted=%d "
                                     "scratch=%d",
                                     std::atan2(dirL.y, dirL.x) * 180.0 /
                                         3.14159265358979323846,
                                     power, static_cast<int>(sim.potted.size()),
                                     sim.cueScratch ? 1 : 0);
                        }
                    }
                    std::vector<pool::Shot> shots;
                    if (!drewSim && cueIdx >= 0 && balls.size() >= 2) {
                        pool::Solver solver(det.table, balls, cueIdx,
                                            pool::SolverOptions{g_args.rails, 80.0, 0.98});
                        shots = solver.allShots();
                    }
                    if (!drewSim && g_visible)
                        drawGuides(det, shots, g_args.shots, fdc);
                    {
                        wchar_t buf[160];
                        _snwprintf(buf, 160,
                                   L"balls=%d shots=%d aim=%s pot=%d%s "
                                   L"| R=%.1f tolV=%.2f%s",
                                   static_cast<int>(balls.size()),
                                   static_cast<int>(shots.size()),
                                   drewSim ? L"ON" : L"-",
                                   drewSim ? aimPotted : 0,
                                   drewSim && aimScratch ? L" SCRATCH!" : L"",
                                   g_args.ballR > 1 ? g_args.ballR : det.ballRPx,
                                   g_args.feltTolV,
                                   g_args.manualRect ? L" [MAN]" : L"");
                        g_status = buf;
                    }
                    if (!g_menu.open && !g_menuHidden) drawHud(fdc, orc);
                    analyzed = true;
                    if (frame % 24 == 0)
                        LOGF("[ok] balls=%d shots=%d best=%s",
                             static_cast<int>(balls.size()),
                             static_cast<int>(shots.size()),
                             shots.empty() ? "-" : shots[0].typeName.c_str());
                } catch (const std::exception& e) {
                    if (frame % 24 == 0) LOGF("[--] %s", e.what());
                    int wl = MultiByteToWideChar(CP_UTF8, 0, e.what(), -1,
                                                 nullptr, 0);
                    if (wl > 1) {
                        std::wstring w(wl - 1, 0);
                        MultiByteToWideChar(CP_UTF8, 0, e.what(), -1, w.data(), wl);
                        g_status = L"[!] " + w;
                    }
                    if (!g_menu.open && !g_menuHidden) drawHud(fdc, orc);
                }
            }
        } else {
            g_status = L"waiting for the game window...";
            if (!g_menu.open && !g_menuHidden) drawHud(fdc, orc);
            if (frame % 60 == 0) {
                char wbuf[512] = {};
                WideCharToMultiByte(
                    CP_UTF8, 0,
                    g_args.window.empty() ? L"(auto-detect any pool game)"
                                          : g_args.window.c_str(),
                    -1, wbuf, 512, nullptr, nullptr);
                LOGF("[--] game window not found (looking for \"%s\")", wbuf);
            }
        }
        // menu last so it draws above the guides
        if (fdc && g_menu.open && !g_menuHidden) drawMenu(fdc);
        if (fdc) presentFrame(orc);
        ++frame;
        Sleep(1000 / g_args.fps);
    }
    if (g_memBmp) {
        SelectObject(g_memDC, g_memOldBmp);
        DeleteObject(g_memBmp);
    }
    if (g_memDC) DeleteDC(g_memDC);
    LOGF("bye");
    return 0;
}
