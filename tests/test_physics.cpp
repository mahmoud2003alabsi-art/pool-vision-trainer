// test_physics.cpp — unit tests for the billiards simulation.
#include "pool/physics.hpp"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace pool;

static int g_checks = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static TableSpec tbl() { return TableSpec::standard(); }  // 900x450, R=12

// Two balls on the center line: cue at x=100, object at x=300, both y=225.
static std::vector<Ball> twoBalls() {
    Ball cue;
    cue.id = 0;
    cue.kind = "cue";
    cue.color = "white";
    cue.pos = {100, 225};
    cue.r = 12;
    Ball obj;
    obj.id = 3;
    obj.kind = "solid";
    obj.color = "red";
    obj.pos = {300, 225};
    obj.r = 12;
    return {cue, obj};
}

int main() {
    // 1. Stun (no spin) head-on: object launches forward, cue nearly stops.
    {
        auto b = twoBalls();
        SimResult r = simulate(tbl(), b, Vec2{500, 0}, 0, 0);
        CHECK(r.firstContact == 3);
        CHECK(r.final[0].pos.x < 280.0);                    // cue rests at impact
        CHECK(r.final[1].pos.x > 320.0);                    // object launched
        CHECK(r.final[1].pos.x > r.final[0].pos.x + 60.0);  // object far ahead
        bool hit = false;
        for (auto& e : r.events) hit |= e.kind == "ball-ball";
        CHECK(hit);
    }

    // 2. Draw (bottom spin): cue comes BACK past the impact zone.
    {
        auto b = twoBalls();
        SimResult r = simulate(tbl(), b, Vec2{500, 0}, -0.8, 0);
        CHECK(r.final[0].pos.x < 240.0);
        CHECK(r.final[1].pos.x > 320.0);
    }

    // 3. Follow (top spin): cue continues forward behind the object.
    {
        auto b = twoBalls();
        SimResult r = simulate(tbl(), b, Vec2{500, 0}, 0.8, 0);
        CHECK(r.final[0].pos.x > 324.0);
        CHECK(r.final[1].pos.x > r.final[0].pos.x);  // object still ahead
    }

    // 4. Cushion bounce: ball rebounds, stays in bounds, event recorded.
    {
        auto b = twoBalls();
        b[1].pos = {100, 400};  // move object far away on the bottom rail
        b[1].id = 3;
        SimResult r = simulate(tbl(), b, Vec2{0, -400}, 0, 0);
        bool cushion = false;
        for (auto& e : r.events) cushion |= e.kind == "cushion" && e.ball == 0;
        CHECK(cushion);
        CHECK(r.final[0].pos.y > 150.0);                    // bounced back
        CHECK(r.final[0].pos.x >= 12.0 && r.final[0].pos.x <= 888.0);
        CHECK(r.final[0].pos.y >= 12.0 && r.final[0].pos.y <= 438.0);
    }

    // 5. Side english changes the exit angle after a cushion bounce.
    {
        std::vector<Ball> b = twoBalls();
        b[1].pos = {100, 400};
        SimResult noE = simulate(tbl(), b, Vec2{220, -420}, 0, 0);
        SimResult withE = simulate(tbl(), b, Vec2{220, -420}, 0, 0.8);
        CHECK(std::fabs(noE.final[0].pos.x - withE.final[0].pos.x) > 40.0);
    }

    // 6. Pot: stun shot down the center line into the bottom-center pocket.
    {
        auto b = twoBalls();
        b[0].pos = {450, 60};
        b[1].pos = {450, 120};
        SimResult r = simulate(tbl(), b, Vec2{0, 420}, 0, 0);
        CHECK(r.potted.size() == 1);
        CHECK(r.potted[0] == 3);
        CHECK(!r.cueScratch);
    }

    // 7. Follow scratches the cue when the object sits right before a pocket;
    //    draw on the same geometry pulls the cue away (no scratch).
    {
        auto b = twoBalls();
        b[0].pos = {450, 300};
        b[1].pos = {450, 380};  // bottom-center pocket at (450, 450)
        SimResult follow = simulate(tbl(), b, Vec2{0, 420}, 0.8, 0);
        CHECK(follow.cueScratch);
        CHECK(follow.potted.size() >= 1);
        SimResult draw = simulate(tbl(), b, Vec2{0, 420}, -0.8, 0);
        CHECK(!draw.cueScratch);
        CHECK(draw.final[0].pos.y < 368.0);  // pulled back from the impact zone
    }

    // 8. minPotPower finds a power that pots, and that power really pots.
    {
        auto b = twoBalls();
        b[0].pos = {376, 188};  // behind the object on the pocket line
        b[1].pos = {600, 300};
        Vec2 dir = (Vec2{900, 450} - b[1].pos).norm();  // bottom-right pocket
        double pw = minPotPower(tbl(), b, dir, 0, 0, 3);
        CHECK(pw > 0.0 && pw <= 1.0);
        SimResult r = simulate(tbl(), b, dir * (PhysicsParams{}.maxShotSpeed * pw), 0, 0);
        bool potted = false;
        for (int id : r.potted) potted |= id == 3;
        CHECK(potted);
    }

    // 9. Determinism: identical inputs give identical final positions.
    {
        auto b1 = twoBalls();
        auto b2 = twoBalls();
        b1[1].pos = {500, 120};
        b2[1].pos = {500, 120};
        SimResult r1 = simulate(tbl(), b1, Vec2{480, 30}, 0.3, -0.2);
        SimResult r2 = simulate(tbl(), b2, Vec2{480, 30}, 0.3, -0.2);
        CHECK(r1.final.size() == r2.final.size());
        for (size_t i = 0; i < r1.final.size(); ++i) {
            CHECK(r1.final[i].pos.x == r2.final[i].pos.x);
            CHECK(r1.final[i].pos.y == r2.final[i].pos.y);
        }
    }

    // 10. All final balls stay inside the field.
    {
        auto b = twoBalls();
        b[1].pos = {700, 300};
        SimResult r = simulate(tbl(), b, Vec2{900, -450}, 0.5, 0.5);
        for (const auto& fb : r.final)
            if (fb.active)
                CHECK(fb.pos.x >= 11.9 && fb.pos.x <= 888.1 && fb.pos.y >= 11.9 &&
                      fb.pos.y <= 438.1);
    }

    std::printf("ALL PHYSICS TESTS PASSED (%d checks)\n", g_checks);
    return 0;
}
