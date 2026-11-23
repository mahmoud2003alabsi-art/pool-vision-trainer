// test_vision.cpp — perspective-felt tests: bilinear quad round trip, a
// synthetic Pool-Club-style trapezoid table (cue must be found and the
// logical positions must land within tolerance), and a flat-table
// regression so the rectangle path stays intact.
#include "pool/vision.hpp"
#include "pool/solver.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static int gFail = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
            ++gFail;                                                         \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                \
    do {                                                                     \
        if (std::fabs((a) - (b)) > (eps)) {                                  \
            std::printf("FAIL %s:%d  %.4f != %.4f\n", __FILE__, __LINE__,    \
                        static_cast<double>(a), static_cast<double>(b));     \
            ++gFail;                                                         \
        }                                                                    \
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

// Fill a felt quad: per-row spans between the left and right edges.
static void fillQuad(std::vector<uint8_t>& px, int w, const Quad& q, int r,
                     int g, int b) {
    for (int y = static_cast<int>(q.tl.y); y <= static_cast<int>(q.bl.y); ++y) {
        double t = (y - q.tl.y) / std::max(1.0, q.bl.y - q.tl.y);
        int lo = static_cast<int>(q.tl.x + (q.bl.x - q.tl.x) * t);
        int hi = static_cast<int>(q.tr.x + (q.br.x - q.tr.x) * t);
        for (int x = lo; x <= hi; ++x) put(px, w, x, y, r, g, b);
    }
}

static Image makeImage(int w, int h, std::vector<uint8_t>& px) {
    Image img;
    img.w = w;
    img.h = h;
    img.px = std::move(px);
    return img;
}

int main() {
    // 1. Bilinear quad round trip on a strong trapezoid.
    {
        Quad q{{300, 100}, {700, 100}, {950, 560}, {50, 560}};
        for (int i = 0; i <= 10; ++i)
            for (int j = 0; j <= 10; ++j) {
                Vec2 l{i * 90.0, j * 45.0};
                Vec2 p = quadLogicalToPx(q, l, 900, 450);
                Vec2 back = quadPxToLogical(q, p, 900, 450);
                CHECK_NEAR(back.x, l.x, 1e-4);
                CHECK_NEAR(back.y, l.y, 1e-4);
            }
    }

    // 2. Synthetic Pool-Club-style perspective table.
    {
        const int W = 1000, H = 620;
        std::vector<uint8_t> px(static_cast<size_t>(W) * H * 3, 0);
        for (size_t i = 0; i < px.size(); i += 3) {
            px[i] = 40; px[i + 1] = 44; px[i + 2] = 52;  // dark page bg
        }
        Quad felt{{300, 100}, {700, 100}, {950, 560}, {50, 560}};
        fillQuad(px, W, felt, 60, 170, 90);  // bright green felt
        TableSpec tbl = TableSpec::standard();  // 900x450, ballR 14
        auto toPx = [&](Vec2 l) {
            return quadLogicalToPx(felt, l, tbl.w, tbl.h);
        };
        // Pocket holes (dark discs at the 6 logical pockets).
        for (const auto& pk : tbl.pockets) {
            Vec2 p = toPx(pk.pos);
            disc(px, W, p.x, p.y, 12, 12, 12, 14);
        }
        // Cue ball at logical (150, 225); object balls elsewhere.
        Vec2 cue = toPx({150, 225});
        disc(px, W, cue.x, cue.y, 10, 250, 250, 250);
        Vec2 yellow = toPx({620, 110});
        disc(px, W, yellow.x, yellow.y, 10, 235, 200, 40);
        Vec2 purple = toPx({700, 300});
        disc(px, W, purple.x, purple.y, 10, 140, 60, 190);
        Vec2 black = toPx({430, 400});
        disc(px, W, black.x, black.y, 10, 25, 25, 28);
        Image img = makeImage(W, H, px);

        DetectResult det = detectBalls(img, {}, tbl);
        CHECK(det.hasQuad);
        // The fitted quad must track the drawn trapezoid.
        CHECK_NEAR(det.quadPx.tl.y, felt.tl.y, 6.0);
        CHECK_NEAR(det.quadPx.bl.y, felt.bl.y, 6.0);
        CHECK(det.balls.size() >= 4);
        int cueIdx = -1;
        for (size_t i = 0; i < det.balls.size(); ++i)
            if (det.balls[i].kind == "cue") cueIdx = static_cast<int>(i);
        CHECK(cueIdx >= 0);
        if (cueIdx >= 0) {
            Vec2 l = det.balls[static_cast<size_t>(cueIdx)].logical;
            CHECK_NEAR(l.x, 150.0, 30.0);  // perspective + fit tolerance
            CHECK_NEAR(l.y, 225.0, 30.0);
        }
        // Ball radius must follow the table scale, not merged clusters.
        double expectedR = ((felt.tr.x - felt.tl.x) + (felt.br.x - felt.bl.x)) /
                           2 / tbl.w * tbl.ballR;
        CHECK_NEAR(det.ballRPx, expectedR, 5.0);
        // Solver works on the detected positions: at least one shot.
        if (cueIdx >= 0 && det.balls.size() >= 2) {
            std::vector<Ball> balls;
            for (size_t i = 0; i < det.balls.size(); ++i) {
                Ball b;
                b.id = det.balls[i].number;
                b.kind = det.balls[i].kind;
                b.color = det.balls[i].color;
                b.pos = det.balls[i].logical;
                b.r = tbl.ballR;
                balls.push_back(b);
            }
            Solver solver(det.table, balls, cueIdx,
                          SolverOptions{1, 80.0, 0.98});
            CHECK(solver.allShots().size() >= 1);
        }
    }

    // 3. Flat rectangle table regression (no perspective).
    {
        const int W = 1000, H = 640;
        std::vector<uint8_t> px(static_cast<size_t>(W) * H * 3, 0);
        for (size_t i = 0; i < px.size(); i += 3) {
            px[i] = 30; px[i + 1] = 32; px[i + 2] = 38;
        }
        for (int y = 120; y < 520; ++y)
            for (int x = 200; x < 900; ++x)
                put(px, W, x, y, 55, 165, 95);
        TableSpec tbl = TableSpec::standard();
        double sx = 700.0 / tbl.w, sy = 400.0 / tbl.h;
        disc(px, W, 200 + 130 * sx, 120 + 225 * sy, 13, 250, 250, 250);
        disc(px, W, 200 + 560 * sx, 120 + 110 * sy, 13, 235, 200, 40);
        Image img = makeImage(W, H, px);
        DetectResult det = detectBalls(img, {}, tbl);
        CHECK(det.balls.size() >= 2);
        bool cue = false;
        for (const auto& d : det.balls) cue |= d.kind == "cue";
        CHECK(cue);
        // Flat felt: no meaningful trapezoid, or one that matches the rect.
        if (det.hasQuad) {
            CHECK_NEAR(det.quadPx.tl.x, 200.0, 25.0);
            CHECK_NEAR(det.quadPx.br.x, 900.0, 25.0);
        }
        int cueIdx = -1;
        for (size_t i = 0; i < det.balls.size(); ++i)
            if (det.balls[i].kind == "cue") cueIdx = static_cast<int>(i);
        if (cueIdx >= 0) {
            std::vector<Ball> balls;
            for (size_t i = 0; i < det.balls.size(); ++i) {
                Ball b;
                b.id = det.balls[i].number;
                b.kind = det.balls[i].kind;
                b.color = det.balls[i].color;
                b.pos = det.balls[i].logical;
                b.r = tbl.ballR;
                balls.push_back(b);
            }
            Solver solver(det.table, balls, cueIdx,
                          SolverOptions{1, 80.0, 0.98});
            CHECK(solver.allShots().size() >= 1);
        }
    }

    // 4. Flipped stripe ball (white cap facing up) must NOT be classified
    //    as the cue: its colored stripe colors the rim ring. The real
    //    white ball must win the cue vote even when the stripe is more
    //    isolated.
    {
        const int W = 1000, H = 640;
        std::vector<uint8_t> px(static_cast<size_t>(W) * H * 3, 0);
        for (size_t i = 0; i < px.size(); i += 3) {
            px[i] = 30; px[i + 1] = 32; px[i + 2] = 38;
        }
        for (int y = 120; y < 520; ++y)
            for (int x = 200; x < 900; ++x)
                put(px, W, x, y, 55, 165, 95);
        TableSpec tbl = TableSpec::standard();
        double sx = 700.0 / tbl.w, sy = 400.0 / tbl.h;
        // Real cue ball: white all the way to the edge.
        double cueX = 200 + 130 * sx, cueY = 120 + 225 * sy;
        disc(px, W, cueX, cueY, 13, 250, 250, 250);
        // Flipped stripe: colored disc with a white cap (pole up).
        double stX = 200 + 560 * sx, stY = 120 + 110 * sy;
        disc(px, W, stX, stY, 13, 235, 200, 40);       // yellow stripe body
        disc(px, W, stX, stY, 13 * 0.60, 252, 250, 246);  // white cap on top
        Image img = makeImage(W, H, px);
        DetectResult det = detectBalls(img, {}, tbl);
        CHECK(det.balls.size() >= 2);
        int cueIdx = -1;
        for (size_t i = 0; i < det.balls.size(); ++i)
            if (det.balls[i].kind == "cue") cueIdx = static_cast<int>(i);
        CHECK(cueIdx >= 0);
        if (cueIdx >= 0) {
            Vec2 p = det.balls[static_cast<size_t>(cueIdx)].px;
            CHECK_NEAR(p.x, cueX, 10.0);   // cue = the real white ball...
            CHECK_NEAR(p.y, cueY, 10.0);   // ...not the flipped stripe
        }
        bool flippedBecameCue = false;
        for (size_t i = 0; i < det.balls.size(); ++i) {
            const auto& d = det.balls[i];
            if (std::hypot(d.px.x - stX, d.px.y - stY) < 15.0 &&
                d.kind == "cue")
                flippedBecameCue = true;
        }
        CHECK(!flippedBecameCue);
    }

    // 5. A flipped stripe alone on the table: nothing may be classified
    //    as the cue ball (it used to become the cue all by itself).
    {
        const int W = 1000, H = 640;
        std::vector<uint8_t> px(static_cast<size_t>(W) * H * 3, 0);
        for (size_t i = 0; i < px.size(); i += 3) {
            px[i] = 30; px[i + 1] = 32; px[i + 2] = 38;
        }
        for (int y = 120; y < 520; ++y)
            for (int x = 200; x < 900; ++x)
                put(px, W, x, y, 55, 165, 95);
        TableSpec tbl = TableSpec::standard();
        double sx = 700.0 / tbl.w, sy = 400.0 / tbl.h;
        double stX = 200 + 350 * sx, stY = 120 + 200 * sy;
        disc(px, W, stX, stY, 13, 235, 200, 40);
        disc(px, W, stX, stY, 13 * 0.60, 252, 250, 246);
        Image img = makeImage(W, H, px);
        DetectResult det = detectBalls(img, {}, tbl);
        CHECK(det.balls.size() >= 1);
        for (const auto& d : det.balls) CHECK(d.kind != "cue");
    }

    // 6. 8BP palette calibration (measured from real 8 Ball Pool renders):
    //    yellow hue 41..62 must read ball 1, orange 15..41 ball 5, dark red
    //    maroon ball 7 (v~0.46), purple 255..345 ball 4.
    {
        const int W = 1000, H = 640;
        TableSpec tbl = TableSpec::standard();
        double sx = 700.0 / tbl.w, sy = 400.0 / tbl.h;
        struct Case { int r, g, b; const char* kind; int num; };
        const Case cases[] = {
            {222, 180, 70, "solid", 1},    // yellow  (hue ~45)
            {210, 140, 50, "solid", 5},    // orange  (hue ~34)
            {120, 45, 28, "solid", 7},     // maroon  (dark red v~0.47)
            {130, 88, 165, "solid", 4},    // purple  (hue ~280)
            {235, 200, 40, "solid", 1},    // golden yellow (old orange bin)
        };
        for (const auto& cs : cases) {
            std::vector<uint8_t> px(static_cast<size_t>(W) * H * 3, 0);
            for (size_t i = 0; i < px.size(); i += 3) {
                px[i] = 30; px[i + 1] = 32; px[i + 2] = 38;
            }
            for (int y = 120; y < 520; ++y)
                for (int x = 200; x < 900; ++x)
                    put(px, W, x, y, 55, 165, 95);
            disc(px, W, 200 + 350 * sx, 120 + 200 * sy, 13, cs.r, cs.g, cs.b);
            Image img = makeImage(W, H, px);
            DetectResult det = detectBalls(img, {}, tbl);
            bool ok = false;
            for (const auto& d : det.balls)
                if (d.kind == cs.kind && d.number == cs.num) ok = true;
            if (!ok) {
                std::printf("FAIL palette: rgb(%d,%d,%d) -> ", cs.r, cs.g,
                            cs.b);
                for (const auto& d : det.balls)
                    std::printf("%s/%d ", d.kind.c_str(), d.number);
                std::printf("\n");
                ++gFail;
            }
        }
    }

    if (gFail == 0) {
        std::printf("ALL VISION TESTS PASSED\n");
        return 0;
    }
    std::printf("%d VISION TEST FAILURES\n", gFail);
    return 1;
}
