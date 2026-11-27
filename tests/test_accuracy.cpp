// test_accuracy.cpp — checks for the measured-physics accuracy layer,
// tied to PUBLISHED numbers (not invented): carom peak, rolling deceler-
// ation, aim tolerance and the success-probability model behavior.
#include "pool/accuracy.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace pool;
using accuracy::caromAngleDeg;
using accuracy::cueDepartureDir;
using accuracy::metersPerUnit;
using accuracy::shotSuccessProbability;
using accuracy::slidingDecel;
using accuracy::stunCaromAngleDeg;
using accuracy::unitsPerMeter;
using accuracy::aimToleranceDeg;
using accuracy::rollingDecel;

static int g_checks = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                  \
    do {                                                                       \
        ++g_checks;                                                            \
        if (std::fabs((a) - (b)) > (eps)) {                                    \
            std::printf("FAIL %s:%d  %.6f != %.6f (eps %.6f)\n", __FILE__,     \
                        __LINE__, static_cast<double>(a),                      \
                        static_cast<double>(b), static_cast<double>(eps));     \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static TableSpec tbl() { return TableSpec::standard(); }  // 900x450, R=12

static Ball makeBall(int id, const char* kind, const char* color, double x,
                     double y) {
    Ball b;
    b.id = id;
    b.kind = kind;
    b.color = color;
    b.pos = {x, y};
    b.r = 12;
    return b;
}

int main() {
    // ------------------------------------------------------------------
    // Unit bridge (published: 1 unit = 2.822 mm on a 9ft table).
    // ------------------------------------------------------------------
    CHECK_NEAR(metersPerUnit() * 1000.0, 2.8222, 0.001);
    CHECK_NEAR(unitsPerMeter() * metersPerUnit(), 1.0, 1e-12);

    // Rolling deceleration: mu=0.0127 (AJP 2009 high-speed imaging) ->
    // ~44.15 units/s^2 (published practical value 44.1).
    CHECK_NEAR(rollingDecel(), 44.145, 0.05);
    CHECK_NEAR(rollingDecel(0.0128), 44.496, 0.05);

    // Sliding deceleration: mu=0.21 (Mathavan 2014) -> ~730 units/s^2.
    CHECK_NEAR(slidingDecel(), 729.96, 2.0);

    // ------------------------------------------------------------------
    // Carom angle (30-degree rule, Alciatore / pooltool): peak 33.7 deg
    // at cut ~28 deg.
    // ------------------------------------------------------------------
    {
        double peakVal = 0, peakAt = 0;
        for (double p = 1.0; p <= 89.0; p += 0.01) {
            double v = caromAngleDeg(p);
            if (v > peakVal) { peakVal = v; peakAt = p; }
        }
        CHECK(peakVal > 33.5 && peakVal < 34.0);   // published: 33.7
        CHECK(peakAt > 27.5 && peakAt < 29.0);     // published: ~28
        // carom(30) ~ 33.67 per the same formula.
        CHECK_NEAR(caromAngleDeg(30.0), 33.67, 0.1);
        CHECK_NEAR(caromAngleDeg(0.0), 0.0, 1e-9);
        CHECK_NEAR(caromAngleDeg(90.0), 0.0, 1e-9);
        CHECK(caromAngleDeg(28.0) > caromAngleDeg(10.0));
    }

    // Stun departure: exactly along the tangent => 90 - cut.
    CHECK_NEAR(stunCaromAngleDeg(30.0), 60.0, 1e-9);
    CHECK_NEAR(stunCaromAngleDeg(45.0), 45.0, 1e-9);

    // cueDepartureDir: unit length, carom angle matches the formula, and
    // it bends toward the tangent side of the cut.
    {
        TableSpec t = tbl();
        std::vector<Ball> balls = {makeBall(0, "cue", "white", 450, 100),
                                   makeBall(3, "solid", "red", 450, 300)};
        Vec2 aim{0, 1};                       // straight down the table
        Vec2 loc{0, 1};                       // line of centers (straight)
        Vec2 dep = cueDepartureDir(aim, loc);
        CHECK_NEAR(dep.len(), 1.0, 1e-9);
        // For a straight-in shot the post-contact cue direction should be
        // the original path (rolling continuation dominates tangential 0).
        CHECK_NEAR(std::acos(std::clamp(dep.dot(aim), -1.0, 1.0)) * 180.0 /
                       3.14159265358979323846,
                   0.0, 1e-6);
        // 45-degree cut: departure angle from the ORIGINAL path must equal
        // caromAngleDeg(45) within numerical error.
        Vec2 aim45{std::cos(3.14159265358979323846 / 4),
                   std::sin(3.14159265358979323846 / 4)};
        Vec2 loc45{1, 0};  // object departs along +X
        Vec2 dep45 = cueDepartureDir(aim45, loc45);
        double ang = std::acos(std::clamp(dep45.dot(aim45), -1.0, 1.0)) *
                     180.0 / 3.14159265358979323846;
        CHECK_NEAR(ang, caromAngleDeg(45.0), 0.1);
        (void)t; (void)balls;
    }

    // ------------------------------------------------------------------
    // Aim tolerance (Dr. Dave: ~3 in effective mouth at 5 ft ~ 2.9 deg).
    // 3 in = 76.2 mm = 27.0 units; 5 ft = 1524 mm = 540 units.
    // ------------------------------------------------------------------
    CHECK_NEAR(aimToleranceDeg(76.2 / (metersPerUnit() * 1000.0),
                               1524.0 / (metersPerUnit() * 1000.0)),
               2.86, 0.05);
    CHECK(aimToleranceDeg(27.0, 270.0) > aimToleranceDeg(27.0, 540.0));
    CHECK_NEAR(aimToleranceDeg(27.0, 0.0), 0.0, 1e-9);

    // ------------------------------------------------------------------
    // Success probability (Monte-Carlo over the noise model).
    // ------------------------------------------------------------------
    {
        TableSpec t = tbl();
        // Straight-in shot: cue (450,100), object (450,300), bottom-middle
        // pocket (450,450). Published benchmark: straight shots succeed
        // with p > 0.9 under the standard noise.
        std::vector<Ball> balls = {makeBall(0, "cue", "white", 450, 100),
                                   makeBall(3, "solid", "red", 450, 300)};
        Vec2 aim = (t.pockets[4].pos - balls[0].pos).norm();
        double pStraight = shotSuccessProbability(t, balls, 0, aim, 0.35, 0, 0,
                                                  3);
        CHECK(pStraight > 0.9);
        CHECK(pStraight <= 1.0);

        // Blocked: a blocker ball on the object->pocket line kills the shot
        // (p < 0.1): the object cannot reach the pocket without passing
        // through the blocker, and only potting the TARGET counts.
        std::vector<Ball> blocked = {makeBall(0, "cue", "white", 450, 100),
                                     makeBall(3, "solid", "red", 450, 300),
                                     makeBall(5, "solid", "blue", 450, 390)};
        double pBlocked = shotSuccessProbability(t, blocked, 0, aim, 0.35, 0,
                                                 0, 3);
        CHECK(pBlocked < 0.1);

        // Determinism: same seed, same probability.
        double pAgain = shotSuccessProbability(t, balls, 0, aim, 0.35, 0, 0,
                                               3);
        CHECK(pAgain == pStraight);

        // Thin 45+ deg cut is harder than the straight shot.
        std::vector<Ball> cut = {makeBall(0, "cue", "white", 450, 100),
                                 makeBall(3, "solid", "red", 550, 300)};
        Vec2 aimCut = (t.pockets[4].pos - cut[1].pos).norm();
        double pCut = shotSuccessProbability(t, cut, 0, aimCut, 0.5, 0, 0, 3);
        CHECK(pCut <= pStraight);
        CHECK(pCut >= 0.0);

        // Unreachable target: probability 0.
        double pNone = shotSuccessProbability(t, balls, 0, aim, 0.35, 0, 0,
                                              42);
        CHECK(pNone == 0.0);

        // No cue index: probability 0.
        double pNoCue = shotSuccessProbability(t, balls, -1, aim, 0.35, 0, 0,
                                               3);
        CHECK(pNoCue == 0.0);
    }

    std::printf("accuracy tests: %d checks passed\n", g_checks);
    return 0;
}
