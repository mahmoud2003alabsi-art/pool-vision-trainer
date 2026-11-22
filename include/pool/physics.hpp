// physics.hpp — event-driven 2D billiards simulation for the pool trainer.
//
// Simplified but plausible model (dependency-free, deterministic):
//   - rolling friction: constant deceleration while a ball moves
//   - ball-ball collisions: equal-mass impulse along the line of centers
//     with restitution; spin kick on the cue ball right after contact
//   - cushions: normal velocity reflected with restitution; side english
//     converts into a tangential kick and decays per bounce
//   - pockets: a ball whose center comes within the capture radius of a
//     pocket anchor is potted
//   - cue contact point (relative to ball center, in ball-radius units):
//     +Y = follow (top spin), -Y = draw (back spin), X = side english
#pragma once
#include "pool/geom.hpp"
#include <string>
#include <vector>

namespace pool {

struct PhysicsParams {
    double maxShotSpeed = 1200;     // logical units/s at 100% power
    double frictionDecel = 160;     // rolling deceleration, units/s^2
    double restitutionBall = 0.94;
    double restitutionCushion = 0.72;
    double followFactor = 0.50;     // forward kick of cue after contact (top)
    double drawFactor = 0.50;       // backward pull of cue after contact
    double sideCushionFactor = 0.35;  // tangential cushion kick per english
    double spinDecayTau = 2.5;      // s, exponential spin decay
    double stopSpeed = 4.0;         // below this a ball rests
    double pocketCapture = 21.0;    // center distance to pocket anchor
    double dt = 1.0 / 240.0;        // simulation step
    double maxTime = 25.0;          // safety cutoff, seconds
};

struct SimEvent {
    double t = 0;
    std::string kind;  // "ball-ball" | "cushion" | "pot" | "rest"
    int ball = -1;     // primary ball id
    int other = -1;    // second ball id / pocket index / rail index (0-3)
    Vec2 pos;
};

struct SimBall {
    int id = -1;
    Vec2 pos, vel;
    double spinF = 0;   // follow(+)/draw(-)
    double spinS = 0;   // side english
    bool active = true; // false once potted
};

struct SimResult {
    std::vector<SimBall> final;
    std::vector<SimEvent> events;
    std::vector<std::vector<Vec2>> paths;  // per input ball, index-aligned
    std::vector<int> potted;               // ball ids in potting order
    bool cueScratch = false;               // cue ball was potted
    int firstContact = -1;                 // id of first ball the cue touched
    double duration = 0;
};

// `balls` must include the cue ball (id 0 or kind "cue"). `cueVel` is applied
// to it; `spinF`/`spinS` in [-1, 1]. Deterministic: same input, same output.
SimResult simulate(const TableSpec& t, const std::vector<Ball>& balls,
                   const Vec2& cueVel, double spinF, double spinS,
                   const PhysicsParams& p = PhysicsParams());

// Lowest power in [0.10, 1.00] (5% granularity) at which the shot pots some
// non-cue ball — or exactly `targetId` when targetId >= 0. Returns -1.0 when
// even full power fails.
double minPotPower(const TableSpec& t, const std::vector<Ball>& balls,
                   const Vec2& cueDir, double spinF, double spinS,
                   int targetId = -1, const PhysicsParams& p = PhysicsParams());

}  // namespace pool
