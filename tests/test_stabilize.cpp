// test_stabilize.cpp — anti-jitter guarantees: rect/quad/scalar gliding
// with snap on real moves, ball track EMA, short holds for vanished balls,
// kind flapping protection and the single-cue invariant.
#include "pool/stabilize.hpp"

#include <cmath>
#include <cstdio>

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

int main() {
    // 1. Rect gliding: a small jitter is damped, a big jump snaps.
    {
        Rect prev{100, 100, 300, 200};
        Rect jitter{102, 99, 302, 201};
        Rect s = smoothRect(prev, jitter, 0.45, 0.03);
        CHECK_NEAR(s.x0, 100 + 2 * 0.45, 1e-9);  // moved only 45% of the way
        CHECK(s.x1 < 302.0);
        Rect jump{160, 100, 360, 200};  // 60 px >> 3% of ~224 diag
        Rect sn = smoothRect(prev, jump, 0.45, 0.03);
        CHECK_NEAR(sn.x0, jump.x0, 1e-9);
        CHECK_NEAR(sn.x1, jump.x1, 1e-9);
        Rect cold{};
        CHECK_NEAR(smoothRect(cold, jitter, 0.45, 0.03).x0, jitter.x0, 1e-9);
    }

    // 2. Rect steady state converges to the input.
    {
        Rect s{100, 100, 300, 200};
        for (int i = 0; i < 30; ++i)
            s = smoothRect(s, Rect{110, 105, 310, 205}, 0.45, 0.03);
        CHECK_NEAR(s.x0, 110.0, 0.01);
        CHECK_NEAR(s.y1, 205.0, 0.01);
    }

    // 3. Quad gliding: corner noise damped, big corner jump snaps whole.
    {
        Quad prev{{100, 100}, {300, 102}, {298, 250}, {102, 248}};
        Quad noise{{101, 101}, {301, 103}, {299, 251}, {103, 249}};
        Quad s = smoothQuad(prev, noise, 0.45, 0.03);
        CHECK(s.tl.x > 100.0 && s.tl.x < 101.0);
        Quad jump{{100, 180}, {300, 182}, {298, 330}, {102, 328}};  // 80px
        Quad sn = smoothQuad(prev, jump, 0.45, 0.03);
        CHECK_NEAR(sn.tl.y, jump.tl.y, 1e-9);
    }

    // 4. Scalar glide for the ball radius.
    {
        double r = smoothScalar(13.0, 13.2, 0.30);
        CHECK(r > 13.0 && r < 13.2);
        CHECK_NEAR(smoothScalar(13.0, 20.0, 0.30), 20.0, 1e-9);  // snap
        CHECK_NEAR(smoothScalar(0.0, 13.0, 0.30), 13.0, 1e-9);   // cold start
    }

    // 5. Ball tracks: matched positions glide; a moving ball is followed.
    {
        BallStabilizer st(5, 0.5, 1.5);
        Detection a;
        a.px = {100, 100};
        a.logical = {50, 25};
        a.kind = "cue";
        std::vector<Detection> balls{a};
        st.apply(balls, 12.0);
        CHECK_NEAR(balls[0].px.x, 100.0, 1e-9);  // first frame: passthrough
        Detection b;
        b.px = {102, 101};
        b.logical = {52, 26};
        b.kind = "cue";
        balls = {b};
        for (int i = 0; i < 20; ++i) {
            balls = {b};
            st.apply(balls, 12.0);
        }
        CHECK_NEAR(balls[0].px.x, 102.0, 0.01);  // converged
        CHECK(balls[0].kind == "cue");
    }

    // 6. Vanished balls are held a few frames, then dropped.
    {
        BallStabilizer st(5, 0.5, 1.5);
        Detection a;
        a.px = {100, 100};
        a.kind = "solid";
        std::vector<Detection> balls{a};
        st.apply(balls, 12.0);
        for (int i = 1; i <= 5; ++i) {
            balls.clear();
            st.apply(balls, 12.0);
            CHECK(balls.size() == 1);  // held
        }
        balls.clear();
        st.apply(balls, 12.0);  // age 6 > hold 5
        CHECK(balls.empty());
    }

    // 7. Kind flapping: unknown never replaces a known kind.
    {
        BallStabilizer st(5, 0.5, 1.5);
        Detection a;
        a.px = {100, 100};
        a.kind = "stripe";
        a.color = "yellow";
        a.number = 9;
        std::vector<Detection> balls{a};
        st.apply(balls, 12.0);
        Detection u;
        u.px = {100.5, 100.5};
        u.kind = "unknown";
        u.number = -1;
        balls = {u};
        st.apply(balls, 12.0);
        CHECK(balls[0].kind == "stripe");
        CHECK(balls[0].number == 9);
    }

    // 8. Single-cue invariant: a fresh cue beats a held cue; two fresh
    //    cues are not created by holds.
    {
        BallStabilizer st(5, 0.5, 1.5);
        Detection cue;
        cue.px = {100, 100};
        cue.kind = "cue";
        std::vector<Detection> balls{cue};
        st.apply(balls, 12.0);
        // Next frame: cue moved far (beyond match radius) so a NEW track
        // starts; the old track is held -> two cues for a moment.
        Detection cue2;
        cue2.px = {180, 180};
        cue2.kind = "cue";
        balls = {cue2};
        st.apply(balls, 12.0);
        int cues = 0;
        for (const auto& d : balls)
            if (d.kind == "cue") ++cues;
        CHECK(cues == 1);  // held cue dropped in favor of the fresh one
        CHECK_NEAR(balls[0].px.x, 180.0, 1e-9);
    }

    if (gFail == 0) std::printf("stabilize tests OK\n");
    return gFail == 0 ? 0 : 1;
}
