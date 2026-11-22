// test_solver.cpp — unit tests for the geometry core and the shot solver.
// Plain asserts, no framework. Exits 0 on success.
#include "pool/solver.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace pool;

namespace {

int g_checks = 0;

void expectNear(double got, double want, double tol, const char* what) {
    ++g_checks;
    if (std::fabs(got - want) > tol) {
        std::fprintf(stderr, "FAIL %s: got %.4f want %.4f (tol %.4f)\n", what,
                     got, want, tol);
        std::exit(1);
    }
}

void expectTrue(bool cond, const char* what) {
    ++g_checks;
    if (!cond) {
        std::fprintf(stderr, "FAIL %s\n", what);
        std::exit(1);
    }
}

TableSpec stdTable() {
    TableSpec t;
    t.w = 900; t.h = 450; t.ballR = 12; t.pocketR = 22;
    t.pockets = {
        {{0, 0}}, {{450, 0}}, {{900, 0}},
        {{0, 450}}, {{450, 450}}, {{900, 450}},
    };
    return t;
}

}  // namespace

int main() {
    // ---------------- geometry primitives ----------------
    // Reflect across x-axis (y = 0).
    Vec2 r1 = reflectPoint({3, 4}, {0, 0}, {1, 0});
    expectNear(r1.x, 3, 1e-9, "reflect x-axis: x");
    expectNear(r1.y, -4, 1e-9, "reflect x-axis: y");
    // Reflect across vertical line x = 5.
    Vec2 r2 = reflectPoint({2, 3}, {5, 0}, {0, 1});
    expectNear(r2.x, 8, 1e-9, "reflect x=5: x");
    expectNear(r2.y, 3, 1e-9, "reflect x=5: y");

    // Segment-point distance: closest point inside vs clamped.
    expectNear(segPointDist({0, 0}, {10, 0}, {5, 3}), 3, 1e-9, "seg dist interior");
    expectNear(segPointDist({0, 0}, {10, 0}, {-4, 0}), 4, 1e-9, "seg dist clamped");

    // Segment vs circle.
    expectTrue(segHitsCircle({0, 0}, {10, 0}, {5, 2}, 3), "seg circle hit");
    expectTrue(!segHitsCircle({0, 0}, {10, 0}, {5, 5}, 3), "seg circle miss");

    // Segment vs infinite line crossing.
    Vec2 hit;
    expectTrue(segLineHit({0, 100}, {100, -100}, {0, 0}, {1, 0}, hit),
               "seg line hit exists");
    expectNear(hit.y, 0, 1e-9, "seg line hit y");
    expectNear(hit.x, 50, 1e-9, "seg line hit x");

    // 30-degree cut: cue aims along +x, object placed so line of centers
    // is 30 degrees off the aim direction.
    {
        TableSpec t = stdTable();
        std::vector<Ball> balls(3);
        balls[0].kind = "cue";  balls[0].pos = {200, 300};   // cue
        balls[1].kind = "solid";                                 // object
        balls[2].kind = "stripe"; balls[2].pos = {700, 100};     // bystander

        double R = t.ballR;
        // Aim direction (1,0). Ghost at 50 px ahead of the cue.
        Vec2 ghost{250, 300};
        Vec2 dir{std::cos(30 * 3.14159265 / 180), std::sin(30 * 3.14159265 / 180)};
        balls[1].pos = ghost + dir * (2 * R);  // object center
        // Pocket along dir.
        t.pockets[5].pos = balls[1].pos + dir * 150;

        Solver solver(t, balls, 0);
        Shot s = solver.solveDirect(1, 5);
        expectTrue(s.feasible, "direct 30deg feasible");
        expectNear(s.cutDeg, 30.0, 0.5, "direct cut angle");
        expectNear(s.ghost.x, 250, 1e-6, "ghost x");
        expectNear(s.ghost.y, 300, 1e-6, "ghost y");
        expectTrue(s.confidence > 0 && s.confidence <= 1, "confidence range");
        // Tangent direction is perpendicular to the line of centers.
        double dot = s.tangentDir.norm().dot(dir);
        expectNear(std::fabs(dot), 0.0, 1e-6, "tangent perpendicular");
    }

    // Obstruction: a ball sitting on the cue->ghost path must be detected.
    {
        TableSpec t = stdTable();
        std::vector<Ball> balls(3);
        balls[0].kind = "cue";    balls[0].pos = {100, 225};
        balls[1].kind = "solid";  balls[1].pos = {500, 225};
        balls[2].kind = "stripe"; balls[2].pos = {300, 228};  // blocks the path
        t.pockets[2].pos = {900, 225};

        Solver solver(t, balls, 0);
        Shot s = solver.solveDirect(1, 2);
        expectTrue(!s.feasible, "blocked direct rejected");
        expectTrue(s.note.find("blocked") != std::string::npos,
                   "blocked note mentions blocking");
    }

    // Ghost ball: straight full-ball shot straight down into a side pocket.
    {
        TableSpec t = stdTable();
        std::vector<Ball> balls(2);
        balls[0].kind = "cue";   balls[0].pos = {300, 30};   // above the object
        balls[1].kind = "solid"; balls[1].pos = {300, 100};
        t.pockets[4].pos = {300, 450};  // bottom side pocket, straight below
        Solver solver(t, balls, 0);
        Shot s = solver.solveDirect(1, 4);
        expectTrue(s.feasible, "straight shot feasible");
        expectNear(s.ghost.x, 300, 1e-9, "straight ghost x");
        expectNear(s.ghost.y, 100 - 2 * 12, 1e-9, "straight ghost y");
        expectNear(s.cutDeg, 0.0, 1e-6, "straight cut 0");
    }

    // KICK 1 rail: reflect across the top rail (y = 0).
    {
        TableSpec t = stdTable();
        std::vector<Ball> balls(2);
        balls[0].kind = "cue";   balls[0].pos = {100, 100};
        balls[1].kind = "solid"; balls[1].pos = {300, 300};

        Solver solver(t, balls, 0);
        Shot s = solver.solveKick(1, {0});
        expectTrue(s.feasible, "kick 1 rail feasible");
        expectTrue(s.cuePath.size() == 3, "kick path C -> rail -> T");
        Vec2 bounce = s.cuePath[1];
        expectNear(bounce.y, 0, 1e-9, "kick bounce on top rail");
        expectNear(bounce.x, 150, 1e-9, "kick bounce x (mirror math)");
        expectTrue(s.confidence > 0 && s.confidence <= 1, "kick confidence range");
    }

    // BANK 1 rail: object departs straight up, bounces on the top rail and
    // comes straight back down into a side pocket below it.
    {
        TableSpec t = stdTable();
        std::vector<Ball> balls(2);
        balls[0].kind = "cue";   balls[0].pos = {500, 220};
        balls[1].kind = "solid"; balls[1].pos = {300, 100};
        t.pockets[4].pos = {300, 450};  // fold (300,450) across y=0 -> (300,-450)
        Solver solver(t, balls, 0);
        Shot s = solver.solveBank(1, 4, {0});
        expectTrue(s.feasible, "bank straight-up feasible");
        expectNear(s.ghost.x, 300, 1e-9, "bank ghost x");
        // Ghost lies 2R below the object (departure direction points up).
        expectNear(s.ghost.y, 100 + 2 * 12, 1e-9, "bank ghost y (2R below)");
        expectTrue(s.objPath.size() >= 3, "bank object path has bounce");
        expectNear(s.objPath[1].y, 0, 1e-9, "bank bounce on top rail");
    }

    // 2-rail kick consistency: bounce points land on the requested rails in
    // order and the final segment ends at the object ball.
    {
        TableSpec t = stdTable();
        std::vector<Ball> balls(2);
        balls[0].kind = "cue";   balls[0].pos = {120, 340};
        balls[1].kind = "solid"; balls[1].pos = {700, 120};

        Solver solver(t, balls, 0);
        Shot s = solver.solveKick(1, {3, 0});  // left rail then top rail
        if (s.feasible) {
            expectTrue(s.cuePath.size() == 4, "kick 2-rail path length");
            expectNear(s.cuePath[1].x, 0, 1e-9, "2-rail bounce1 on left rail");
            expectNear(s.cuePath[2].y, 0, 1e-9, "2-rail bounce2 on top rail");
            expectNear(s.cuePath.back().x, 700, 1e-9, "2-rail end at object");
        }
        // Feasibility itself depends on whether bounce points fall within
        // rail extents; the solver must never return a path with a bounce
        // outside a rail extent.
        for (size_t i = 0; i < s.cuePath.size(); ++i) {
            const Vec2& p = s.cuePath[i];
            expectTrue(p.x >= -1e-6 && p.x <= t.w + 1e-6 && p.y >= -1e-6 &&
                           p.y <= t.h + 1e-6,
                       "kick path stays in field");
        }
    }

    // allShots: scene with a clean direct shot must include it ranked first;
    // every returned shot is feasible with confidence in (0, 1].
    {
        TableSpec t = stdTable();
        std::vector<Ball> balls(4);
        balls[0].kind = "cue";   balls[0].pos = {380, 120};
        balls[1].kind = "solid"; balls[1].pos = {520, 180};   // red 3
        balls[2].kind = "stripe"; balls[2].pos = {620, 350};  // 12
        balls[3].kind = "black"; balls[3].pos = {700, 380};   // 8

        Solver solver(t, balls, 0, SolverOptions{2, 80.0, 0.98});
        std::vector<Shot> shots = solver.allShots();
        expectTrue(!shots.empty(), "allShots non-empty");
        for (const auto& s : shots) {
            expectTrue(s.feasible, "allShots feasible only");
            expectTrue(s.confidence > 0 && s.confidence <= 1, "conf range");
        }
        // The curated geometry must yield the direct red->corner shot.
        bool hasDirect = false;
        for (const auto& s : shots)
            if (s.typeName == "direct" && s.objectIdx == 1) hasDirect = true;
        expectTrue(hasDirect, "direct shot for ball 1 exists");
        // Sorted by confidence, descending.
        for (size_t i = 0; i + 1 < shots.size(); ++i)
            expectTrue(shots[i].confidence >= shots[i + 1].confidence,
                       "shots sorted desc");
    }

    // Cut-too-thin rejection: object almost behind the cue's back.
    {
        TableSpec t = stdTable();
        std::vector<Ball> balls(2);
        balls[0].kind = "cue";   balls[0].pos = {100, 225};
        balls[1].kind = "solid"; balls[1].pos = {400, 225};
        // Pocket almost directly left of the object -> cut ~180 deg.
        t.pockets[3].pos = {150, 225};
        Solver solver(t, balls, 0);
        Shot s = solver.solveDirect(1, 3);
        expectTrue(!s.feasible, "impossible cut rejected");
    }

    std::printf("ALL TESTS PASSED (%d checks)\n", g_checks);
    return 0;
}
