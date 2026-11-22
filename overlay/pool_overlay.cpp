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
#include "pool/physics.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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
    bool manualRect = false;
    double rectX0 = 0, rectY0 = 0, rectX1 = 1, rectY1 = 1;  // client fractions
};

Args g_args;
HWND g_ov = nullptr;
bool g_visible = true;

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
        int wl = MultiByteToWideChar(CP_UTF8, 0, p, -1, nullptr, 0);
        if (wl > 1) {
            std::wstring w(wl - 1, 0);
            MultiByteToWideChar(CP_UTF8, 0, p, -1, w.data(), wl);
            a.window = w;
        }
    }
    if (strVal("tableW")) a.tw = numVal("tableW");
    if (strVal("tableH")) a.th = numVal("tableH");
    if (strVal("rails")) a.rails = static_cast<int>(numVal("rails"));
    if (strVal("shots")) a.shots = static_cast<int>(numVal("shots"));
    if (strVal("fps")) a.fps = static_cast<int>(numVal("fps"));
    if (strVal("guides")) g_visible = numVal("guides") != 0;
    if (strVal("aimPreview")) a.aimPreview = numVal("aimPreview") != 0;
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
        std::printf("[!] cannot write ini\n");
        return;
    }
    char buf[1024];
    int n = std::snprintf(buf, sizeof(buf),
        "window=8 ball pool\ntableW=%.0f\ntableH=%.0f\nrails=%d\nshots=%d\n"
        "fps=%d\nguides=%d\naimPreview=%d\nballR=%.1f\nfeltTolS=%.2f\n"
        "feltTolV=%.2f\nstripLines=%d\nmanualRect=%d\n"
        "rectX0=%.4f\nrectY0=%.4f\nrectX1=%.4f\nrectY1=%.4f\n",
        g_args.tw, g_args.th, g_args.rails, g_args.shots, g_args.fps,
        g_visible ? 1 : 0, g_args.aimPreview ? 1 : 0, g_args.ballR,
        g_args.feltTolS, g_args.feltTolV, g_args.stripLines ? 1 : 0,
        g_args.manualRect ? 1 : 0, g_args.rectX0, g_args.rectY0,
        g_args.rectX1, g_args.rectY1);
    DWORD wrote = 0;
    WriteFile(f, buf, static_cast<DWORD>(n > 0 ? n : 0), &wrote, nullptr);
    CloseHandle(f);
    std::printf("[i] saved %ls\n", path.c_str());
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
        if (g_visible) ShowWindow(g_ov, SW_SHOWNOACTIVATE);
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
                if (g_visible) ShowWindow(g_ov, SW_SHOWNOACTIVATE);
            }
        }
    }

    SelectObject(mem, oldBmp);
    DeleteObject(bmp);
    DeleteDC(mem);
    ReleaseDC(hwnd, wdc);
    return ok != FALSE;
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
    HDC dc = GetDC(g_ov);
    fillKey(dc, rc);
    ReleaseDC(g_ov, dc);
}

// Draw the best shots' guides in client pixel coordinates. The overlay is
// exactly the game client size, so client coords map 1:1.
void drawGuides(const pool::DetectResult& det, const std::vector<pool::Shot>& shots,
                int topN) {
    if (!g_ov) return;
    RECT rc;
    GetClientRect(g_ov, &rc);
    HDC dc = GetDC(g_ov);
    fillKey(dc, rc);

    if (g_visible && shots.size() >= 1 && det.balls.size() >= 1) {
        pool::Rect f = det.feltRect;  // same +/-2 shrink detection mapped with
        f.x0 += 2; f.y0 += 2; f.x1 -= 2; f.y1 -= 2;
        const pool::TableSpec& tbl = det.table;
        auto toPx = [&](pool::Vec2 l) {
            return POINT{static_cast<LONG>(f.x0 + l.x / tbl.w * f.w()),
                         static_cast<LONG>(f.y0 + l.y / tbl.h * f.h())};
        };
        const int ballPx = std::max(3, static_cast<int>(tbl.ballR * f.w() / tbl.w));
        const int nDraw = std::min<int>(topN, static_cast<int>(shots.size()));
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
                HBRUSH yb = CreateSolidBrush(RGB(255, 255, 0));
                HGDIOBJ ob = SelectObject(dc, yb);
                Ellipse(dc, p.x - 5, p.y - 5, p.x + 5, p.y + 5);
                SelectObject(dc, ob);
                DeleteObject(yb);
            }
            // 90-degree tangent hint after contact (direct shots).
            if (s.typeName == "direct") {
                HPEN tp = CreatePen(PS_SOLID, 1, RGB(170, 170, 180));
                SelectObject(dc, tp);
                POINT g = toPx(s.ghost);
                const int len = static_cast<int>(ballPx * 2.2);
                MoveToEx(dc, g.x - static_cast<LONG>(s.tangentDir.x * len),
                         g.y - static_cast<LONG>(s.tangentDir.y * len), nullptr);
                LineTo(dc, g.x + static_cast<LONG>(s.tangentDir.x * len),
                       g.y + static_cast<LONG>(s.tangentDir.y * len));
                DeleteObject(tp);
            }
            SelectObject(dc, oldPen);
            DeleteObject(pen);
            DeleteObject(ypen);
        }
    }
    ReleaseDC(g_ov, dc);
}

std::wstring g_status;  // one-line HUD text, updated by the analysis loop

void drawHud() {
    if (!g_ov || !g_visible || g_status.empty()) return;
    HDC dc = GetDC(g_ov);
    RECT rc;
    GetClientRect(g_ov, &rc);
    rc.bottom = rc.top + 22;
    RECT bg = {0, 0, rc.right, rc.bottom};
    HBRUSH br = CreateSolidBrush(RGB(0, 0, 0));
    FillRect(dc, &bg, br);
    DeleteObject(br);
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, RGB(255, 255, 120));
    DrawTextW(dc, g_status.c_str(), -1, &rc,
              DT_SINGLELINE | DT_LEFT | DT_VCENTER);
    ReleaseDC(g_ov, dc);
}

// Live simulation preview: the game's own aim guideline was detected, so show
// what the CURRENT aim will do — every ball's predicted path, the rest
// positions and a pocket flash (yellow = object potted, red = scratch).
void drawSimPreview(const pool::DetectResult& det, const std::vector<pool::Ball>& balls,
                    const pool::SimResult& sim, pool::Vec2 dirPx) {
    if (!g_ov) return;
    RECT rc;
    GetClientRect(g_ov, &rc);
    HDC dc = GetDC(g_ov);
    fillKey(dc, rc);
    if (g_visible && !sim.final.empty() && balls.size() == sim.final.size()) {
        pool::Rect f = det.feltRect;  // same +/-2 shrink detection mapped with
        f.x0 += 2; f.y0 += 2; f.x1 -= 2; f.y1 -= 2;
        const pool::TableSpec& tbl = det.table;
        auto toPx = [&](pool::Vec2 l) {
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
    g_args = parseArgs();
    if (!g_args.window.empty() && g_args.window.size() > 200) g_args.window.resize(200);

    WNDCLASSW wc{};
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"PoolVisionOverlay";
    wc.hbrBackground = CreateSolidBrush(kKey);
    RegisterClassW(&wc);

    g_ov = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW |
            WS_EX_NOACTIVATE,
        wc.lpszClassName, L"", WS_POPUP, 0, 0, 100, 100, nullptr, nullptr,
        wc.hInstance, nullptr);
    if (!g_ov) {
        std::fprintf(stderr, "fatal: cannot create overlay window\n");
        return 1;
    }
    SetLayeredWindowAttributes(g_ov, kKey, 0, LWA_COLORKEY);
    ShowWindow(g_ov, SW_SHOWNOACTIVATE);

    // Logical table for 8 Ball Pool: pockets must follow the custom size.
    pool::TableSpec tbl = pool::TableSpec::standard();
    tbl.w = g_args.tw;
    tbl.h = g_args.th;
    tbl.pockets.clear();
    auto addPk = [&](double x, double y) { tbl.pockets.push_back({{x, y}}); };
    addPk(0, 0); addPk(tbl.w / 2, 0); addPk(tbl.w, 0);
    addPk(0, tbl.h); addPk(tbl.w / 2, tbl.h); addPk(tbl.w, tbl.h);

    const std::wstring needle = lower(g_args.window);
    std::string needleNarrow;
    for (wchar_t c : needle) needleNarrow.push_back(static_cast<char>(c));

    std::printf("pool-overlay started. window=\"%s\" table=%.0fx%.0f rails=%d shots=%d fps=%d\n",
                needleNarrow.c_str(), g_args.tw, g_args.th, g_args.rails,
                g_args.shots, g_args.fps);
    std::printf("hotkeys: F9 = show/hide guides, F12 = quit\n");

    HWND game = nullptr;
    int frame = 0;
    for (;;) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) return 0;
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if ((GetAsyncKeyState(VK_F9) & 1) && g_ov) {
            g_visible = !g_visible;
            ShowWindow(g_ov, g_visible ? SW_SHOWNOACTIVATE : SW_HIDE);
            std::printf("[i] guides %s\n", g_visible ? "shown" : "hidden");
        }
        if (GetAsyncKeyState(VK_F12) & 1) break;
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

        if (!game || !IsWindow(game) || IsIconic(game)) game = findGameWindow(needle);
        if (game) {
            RECT rc;
            GetClientRect(game, &rc);
            POINT pt{};
            ClientToScreen(game, &pt);
            SetWindowPos(g_ov, HWND_TOPMOST, pt.x, pt.y, rc.right - rc.left,
                         rc.bottom - rc.top, SWP_NOACTIVATE | SWP_SHOWWINDOW);

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
                    pool::DetectResult det = pool::detectBalls(cap, dopt, tbl);
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
                            pool::Rect f = det.feltRect;
                            f.x0 += 2; f.y0 += 2; f.x1 -= 2; f.y1 -= 2;
                            pool::Vec2 dirL{dirPx.x * det.table.w / f.w(),
                                            dirPx.y * det.table.h / f.h()};
                            dirL = dirL.len() > 1e-12 ? dirL.norm()
                                                      : pool::Vec2{1, 0};
                            double power = pool::minPotPower(det.table, balls,
                                                             dirL, 0, 0);
                            if (power < 0) power = 1.0;
                            pool::SimResult sim = pool::simulate(
                                det.table, balls,
                                dirL * (pool::PhysicsParams{}.maxShotSpeed * power),
                                0, 0);
                            drawSimPreview(det, balls, sim, dirPx);
                            aimPotted = static_cast<int>(sim.potted.size());
                            aimScratch = sim.cueScratch;
                            drewSim = true;
                            if (frame % 24 == 0)
                                std::printf(
                                    "[aim] deg=%.0f power=%.2f potted=%d scratch=%d\n",
                                    std::atan2(dirL.y, dirL.x) * 180.0 / 3.14159265358979323846,
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
                    if (!drewSim) drawGuides(det, shots, g_args.shots);
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
                    drawHud();
                    analyzed = true;
                    if (frame % 24 == 0)
                        std::printf("[ok] balls=%d shots=%d best=%s\n",
                                    static_cast<int>(balls.size()), shots.size(),
                                    shots.empty() ? "-" : shots[0].typeName.c_str());
                } catch (const std::exception& e) {
                    if (frame % 24 == 0) std::printf("[--] %s\n", e.what());
                    int wl = MultiByteToWideChar(CP_UTF8, 0, e.what(), -1,
                                                 nullptr, 0);
                    if (wl > 1) {
                        std::wstring w(wl - 1, 0);
                        MultiByteToWideChar(CP_UTF8, 0, e.what(), -1, w.data(), wl);
                        g_status = L"[!] " + w;
                    }
                    drawHud();
                }
            }
            if (!analyzed) clearOverlay();
        } else {
            clearOverlay();
            if (frame % 60 == 0)
                std::printf("[--] game window not found (looking for \"%s\")\n",
                            needleNarrow.c_str());
        }
        ++frame;
        Sleep(1000 / g_args.fps);
    }
    std::printf("bye\n");
    return 0;
}
