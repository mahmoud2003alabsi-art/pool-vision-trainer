// test_stick.cpp — live cue stick + power slider detectors: angle recovery
// from a drawn shaft, pull-back power, the no-stick case, the navy
// background false-positive guard, and the slider knob fraction.
#include "pool/vision.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static int gFail = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);     \
            ++gFail;                                                        \
        }                                                                   \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                               \
    do {                                                                    \
        if (std::fabs((a) - (b)) > (eps)) {                                 \
            std::printf("FAIL %s:%d  %.4f != %.4f\n", __FILE__, __LINE__,   \
                        static_cast<double>(a), static_cast<double>(b));    \
            ++gFail;                                                        \
        }                                                                   \
    } while (0)

using namespace pool;

static void put(std::vector<uint8_t>& px, int w, int x, int y, int r, int g,
                int b) {
    if (x < 0 || y < 0 || x >= w || y >= static_cast<int>(px.size() / 3 / w))
        return;
    size_t i = (static_cast<size_t>(y) * w + x) * 3;
    px[i] = static_cast<uint8_t>(r);
    px[i + 1] = static_cast<uint8_t>(g);
    px[i + 2] = static_cast<uint8_t>(b);
}

static void disc(std::vector<uint8_t>& px, int w, double cx, double cy,
                 double rad, int r, int g, int b) {
    for (int y = static_cast<int>(cy - rad) - 1;
         y <= static_cast<int>(cy + rad) + 1; ++y)
        for (int x = static_cast<int>(cx - rad) - 1;
             x <= static_cast<int>(cx + rad) + 1; ++x) {
            double dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy <= rad * rad)
                put(px, w, x, y, r, g, b);
        }
}

static void thickLine(std::vector<uint8_t>& px, int w, double x0, double y0,
                      double x1, double y1, double wd, int r, int g, int b) {
    double len = std::hypot(x1 - x0, y1 - y0);
    for (double t = 0; t <= len; t += 0.7) {
        double cx = x0 + (x1 - x0) * t / len;
        double cy = y0 + (y1 - y0) * t / len;
        disc(px, w, cx, cy, wd, r, g, b);
    }
}

static Image makeImage(int w, int h, std::vector<uint8_t>& px) {
    Image img;
    img.w = w;
    img.h = h;
    img.px = px;
    return img;
}

// Felt + cue ball + an optional stick. Stick at screen angle `stickDeg`
// (y grows down), tip touching the ball unless `gapR` > 0.
struct Scene {
    Image img;
    Vec2 cue{400, 300};
};

static Scene buildScene(double stickDeg, double gapR, bool navyAbove) {
    const int W = 800, H = 600;
    std::vector<uint8_t> px(static_cast<size_t>(W) * H * 3, 0);
    for (size_t i = 0; i < px.size(); i += 3) {
        px[i] = 30; px[i + 1] = 32; px[i + 2] = 38;  // dark page bg
    }
    Rect field{100, 100, 700, 500};
    for (int y = 100; y < 500; ++y)
        for (int x = 100; x < 700; ++x)
            put(px, W, x, y, 45, 150, 90);  // green felt
    Scene sc;
    double R = 14.0;
    // cue ball
    disc(px, W, sc.cue.x, sc.cue.y, R, 250, 250, 250);
    // stick: wood shaft with a dark edge, screen angle stickDeg
    double a = stickDeg * 3.14159265358979323846 / 180.0;
    double dx = std::cos(a), dy = std::sin(a);
    double tipR = R * 1.05 + gapR * R;
    double len = 8.5 * R;
    double x0 = sc.cue.x + dx * tipR, y0 = sc.cue.y + dy * tipR;
    double x1 = sc.cue.x + dx * (tipR + len), y1 = sc.cue.y + dy * (tipR + len);
    thickLine(px, W, x0, y0, x1, y1, 2.0, 150, 105, 60);   // tan shaft
    thickLine(px, W, x0 + dy * 3, y0 - dx * 3, x1 + dy * 3, y1 - dx * 3,
              1.2, 60, 40, 25);                            // dark edge
    if (navyAbove) {
        // navy game background block above the felt (hue ~215)
        for (int y = 20; y < 100; ++y)
            for (int x = 150; x < 650; ++x)
                put(px, W, x, y, 25, 40, 68);
        // the stick still crosses it (over the dark UI)
        thickLine(px, W, x0, y0 - 0, x1, y1, 2.0, 150, 105, 60);
    }
    sc.img = makeImage(W, H, px);
    return sc;
}

int main() {
    Rect field{100, 100, 700, 500};
    double R = 14.0;

    // 1. Stick touching the ball at 45 deg up-right -> aim = down-left.
    {
        Scene sc = buildScene(315.0, 0.0, false);  // screen: up-right
        CueStick st = findCueStick(sc.img, sc.cue, R, field);
        CHECK(st.found);
        if (st.found) {
            // aim direction = opposite the shaft: down-left ~135 deg
            double aimDeg = std::atan2(st.dir.y, st.dir.x) * 180.0 /
                            3.14159265358979323846;
            double d = std::fabs(aimDeg - 135.0);
            if (d > 180) d = 360 - d;
            CHECK(d < 15.0);
            CHECK(st.power < 0.25);  // touching = idle
        }
    }

    // 2. Pulled-back stick: gap 4R -> power ~0.5.
    {
        Scene sc = buildScene(300.0, 4.0, false);
        CueStick st = findCueStick(sc.img, sc.cue, R, field);
        CHECK(st.found);
        if (st.found) CHECK_NEAR(st.power, 0.5, 0.18);
    }

    // 3. No stick at all -> not found.
    {
        const int W = 800, H = 600;
        std::vector<uint8_t> px(static_cast<size_t>(W) * H * 3, 0);
        for (int y = 100; y < 500; ++y)
            for (int x = 100; x < 700; ++x)
                put(px, W, x, y, 45, 150, 90);
        disc(px, W, 400, 300, R, 250, 250, 250);
        Image img = makeImage(W, H, px);
        CueStick st = findCueStick(img, {400, 300}, R, field);
        CHECK(!st.found);
    }

    // 4. Navy background above the felt: the stick over it is still found
    //    and the navy itself must not fabricate extra directions (the
    //    recovered aim stays within tolerance of the true one).
    {
        Scene sc = buildScene(285.0, 0.0, true);
        CueStick st = findCueStick(sc.img, sc.cue, R, field);
        CHECK(st.found);
        if (st.found) {
            double aimDeg = std::atan2(st.dir.y, st.dir.x) * 180.0 /
                            3.14159265358979323846;
            double d = std::fabs(aimDeg - 105.0);
            if (d > 180) d = 360 - d;
            CHECK(d < 15.0);
        }
    }

    // 5. Power slider: orange bar with a whiter knob at 40% of its height.
    {
        const int W = 800, H = 600;
        std::vector<uint8_t> px(static_cast<size_t>(W) * H * 3, 0);
        for (int y = 100; y < 500; ++y)
            for (int x = 100; x < 700; ++x)
                put(px, W, x, y, 45, 150, 90);
        // bar: hue 28, s 0.57, v 0.95 -> rgb(242,164,102)-ish
        for (int y = 150; y < 390; ++y)
            for (int x = 60; x < 65; ++x)
                put(px, W, x, y, 242, 164, 102);
        // knob at y = 150 + 0.40*240 = 246: whiter (v ~1, s ~0.3)
        for (int y = 240; y <= 252; ++y)
            for (int x = 58; x < 67; ++x)
                put(px, W, x, y, 255, 222, 180);
        Image img = makeImage(W, H, px);
        double pw = findPowerSlider(img, field, R);
        CHECK(pw >= 0);
        if (pw >= 0) CHECK_NEAR(pw, 0.40, 0.10);
    }

    // 6. Bar without a visible knob -> unknown (-1), never a fake number.
    {
        const int W = 800, H = 600;
        std::vector<uint8_t> px(static_cast<size_t>(W) * H * 3, 0);
        for (int y = 100; y < 500; ++y)
            for (int x = 100; x < 700; ++x)
                put(px, W, x, y, 45, 150, 90);
        for (int y = 150; y < 390; ++y)
            for (int x = 60; x < 65; ++x)
                put(px, W, x, y, 242, 164, 102);
        Image img = makeImage(W, H, px);
        double pw = findPowerSlider(img, field, R);
        CHECK_NEAR(pw, -1.0, 1e-9);
    }

    if (gFail == 0) std::printf("stick/slider tests OK\n");
    return gFail == 0 ? 0 : 1;
}
