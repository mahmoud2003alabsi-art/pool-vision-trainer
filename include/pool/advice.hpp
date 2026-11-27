// advice.hpp — the full shot recipe ("hacker mode"): turn a solved Shot
// into an executable recipe — recommended power, recommended cue contact
// point (stun / draw / follow / english) and the predicted outcome of
// playing that recipe, computed with the real physics simulation plus the
// Monte-Carlo success probability from accuracy.hpp.
//
// The advisor tries a fixed set of spin presets and, for each, the lowest
// power that pots the intended object ball (plus small speed margins for
// execution noise). Among the candidates that pot it prefers, in order:
//   1. pots the target AND no scratch,
//   2. pots the target (scratch tolerated),
//   3. lower power,
//   4. less spin (stun first).
// This is what makes the advice "hacker-grade": when a stun follow-through
// would scratch into a pocket, the advisor automatically switches to draw.
// Pure math on top of solver.hpp + physics.hpp + accuracy.hpp; deterministic.
#pragma once
#include "pool/accuracy.hpp"
#include "pool/physics.hpp"
#include "pool/solver.hpp"
#include <string>
#include <vector>

namespace pool {

struct ShotAdvice {
    // the source shot (index into the solver's shot list it was built from)
    int shotIdx = -1;
    int objId = -1;        // object ball id the recipe pots
    int pocketIdx = -1;    // target pocket (kick: -1)
    std::string typeName;  // "direct" | "bank" | "kick"

    // the recipe
    double power = 0;      // 0..1 recommended pull
    double spinF = 0;      // +follow / -draw, ball-radius units
    double spinS = 0;      // +right / -left english, ball-radius units
    std::string spinLabel; // "stun" | "draw" | "follow" [+ english]

    // predicted outcome (from the physics sim of exactly this recipe)
    bool potsTarget = false;
    bool scratch = false;
    int potted = 0;              // total balls potted
    Vec2 cueRest{0, 0};
    Vec2 objRest{0, 0};
    double pSuccess = 0;         // Monte-Carlo under execution noise [0..1]
    SimResult sim;               // full outcome for drawing paths
};

// "stun" | "draw" | "follow" (+ " + left/right english")
std::string spinLabel(double spinF, double spinS);

// Build the best recipe for one solved shot.
ShotAdvice adviseShot(const TableSpec& table, const std::vector<Ball>& balls,
                      const Shot& shot, int cueIdx, int shotIdx = -1);

// Advise the first `maxShots` feasible shots and return them sorted best
// first (success tier, then Monte-Carlo probability, then solver
// confidence). Infeasible shots are skipped.
std::vector<ShotAdvice> adviseShots(const TableSpec& table,
                                    const std::vector<Ball>& balls,
                                    const std::vector<Shot>& shots, int cueIdx,
                                    int maxShots = 3);

}  // namespace pool
