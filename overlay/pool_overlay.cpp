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
#include <algorithm>
#include <cstdio>
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
        "hotkeys: F9 = show/hide guides, F12 = quit\n");
}

std::wstring lower(std::wstring s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](wchar_t c) { return static_cast<wchar_t>(::towlower(c)); });
    return s;
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
        else if (s == L"--help" || s == L"-h") { printUsage(); ExitProcess(0); }
    }
    LocalFree(v);
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
        if (GetDIBits(mem, bmp, 0, h, buf.data(), &bi, DIB_RGB_COLORS)) {
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
        } else {
            ok = FALSE;
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
                    pool::DetectResult det = pool::detectBalls(cap, {}, tbl);
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
                    std::vector<pool::Shot> shots;
                    if (cueIdx >= 0 && balls.size() >= 2) {
                        pool::Solver solver(det.table, balls, cueIdx,
                                            pool::SolverOptions{g_args.rails, 80.0, 0.98});
                        shots = solver.allShots();
                    }
                    drawGuides(det, shots, g_args.shots);
                    analyzed = true;
                    if (frame % 24 == 0)
                        std::printf("[ok] balls=%d shots=%d best=%s\n",
                                    static_cast<int>(balls.size()), shots.size(),
                                    shots.empty() ? "-" : shots[0].typeName.c_str());
                } catch (const std::exception& e) {
                    if (frame % 24 == 0) std::printf("[--] %s\n", e.what());
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
