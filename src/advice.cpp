// advice.cpp — implementation of the shot recipe advisor.
#include "pool/advice.hpp"
#include <algorithm>
#include <cmath>
#include <tuple>

namespace pool {

std::string spinLabel(double spinF, double spinS) {
    std::string lbl = spinF > 0.25   ? "follow"
                      : spinF < -0.25 ? "draw"
                                      : "stun";
    if (spinS > 0.25) lbl += " + right english";
    else if (spinS < -0.25) lbl += " + left english";
    return lbl;
}

namespace {

struct Cand {
    double power = 0;
    double spinF = 0;
    double spinS = 0;
    bool pots = false;   // pots the target
    bool scratch = false;
    int potted = 0;
    SimResult sim;
};

// Candidate powers worth trying around the minimum potting power.
std::vector<double> powerCandidates(double pMin) {
    std::vector<double> out;
    if (pMin >= 0) {
        out.push_back(pMin);
        out.push_back(std::min(1.0, pMin + 0.12));  // noise margin
        out.push_back(std::min(1.0, pMin + 0.25));
    }
    return out;
}

// Preference order: success tier, lower power, less spin.
std::tuple<int, double, double> rankKey(const Cand& c) {
    int tier = (c.pots && !c.scratch) ? 0 : (c.pots ? 1 : 2);
    return {tier, c.power, std::fabs(c.spinF) + std::fabs(c.spinS)};
}

Cand bestForPreset(const TableSpec& t, const std::vector<Ball>& balls,
                   int cueIdx, const Vec2& dir, int targetId, double spinF,
                   double spinS, const PhysicsParams& p) {
    Cand best;
    bool have = false;
    double pMin = minPotPower(t, balls, dir, spinF, spinS, targetId, p);
    for (double pw : powerCandidates(pMin)) {
        SimResult sim =
            simulate(t, balls, dir * (p.maxShotSpeed * pw), spinF, spinS, p);
        Cand c;
        c.power = pw;
        c.spinF = spinF;
        c.spinS = spinS;
        c.pots = std::find(sim.potted.begin(), sim.potted.end(), targetId) !=
                 sim.potted.end();
        c.scratch = sim.cueScratch;
        c.potted = static_cast<int>(sim.potted.size());
        c.sim = std::move(sim);
        if (!have || rankKey(c) < rankKey(best)) {
            best = std::move(c);
            have = true;
        }
    }
    return best;
}

}  // namespace

ShotAdvice adviseShot(const TableSpec& table, const std::vector<Ball>& balls,
                      const Shot& shot, int cueIdx, int shotIdx) {
    ShotAdvice a;
    a.shotIdx = shotIdx;
    a.typeName = shot.typeName;
    a.pocketIdx = shot.pocketIdx;
    if (cueIdx < 0 || shot.objectIdx < 0 ||
        shot.objectIdx >= static_cast<int>(balls.size()) || !shot.feasible)
        return a;
    a.objId = balls[shot.objectIdx].id;

    const Vec2 dir = (shot.ghost - balls[cueIdx].pos).norm();
    const int targetId = a.objId;

    // Spin presets: stun first (simplest to execute), then draw / follow
    // (positional play + scratch escape), then soft english variants that
    // spin the cue off a rail-bound scratch line.
    static const struct { double f, s; } presets[] = {
        {0, 0}, {0, -0.65}, {0, 0.55}, {0, -0.65}, {0, 0.55},
        {-0.45, -0.4}, {-0.45, 0.4}, {0.45, -0.4}, {0.45, 0.4}};
    // first three are the core; the rest add english escapes
    const int nCore = 3, nAll = 9;

    Cand best;
    bool have = false;
    for (int i = 0; i < nAll; ++i) {
        // Only try english escapes after the core presets failed to find a
        // scratch-free pot — keeps the common case cheap.
        if (i >= nCore && have && best.pots && !best.scratch) break;
        Cand c = bestForPreset(table, balls, cueIdx, dir, targetId,
                               presets[i].f, presets[i].s, PhysicsParams{});
        if (!have || rankKey(c) < rankKey(best)) {
            best = std::move(c);
            have = true;
        }
    }
    if (!have) return a;

    a.power = best.power;
    a.spinF = best.spinF;
    a.spinS = best.spinS;
    a.spinLabel = spinLabel(best.spinF, best.spinS);
    a.potsTarget = best.pots;
    a.scratch = best.scratch;
    a.potted = best.potted;
    a.sim = best.sim;
    // rest positions: cue rest; object rest = where it lies, or the pocket
    // anchor when the recipe pots it (active=false there).
    for (size_t i = 0; i < best.sim.final.size(); ++i) {
        if (static_cast<int>(i) == cueIdx)
            a.cueRest = best.sim.final[i].pos;
        if (static_cast<int>(i) == shot.objectIdx) {
            if (best.sim.final[i].active)
                a.objRest = best.sim.final[i].pos;
            else
                for (const auto& e : best.sim.events)
                    if (e.kind == "pot" && e.ball == a.objId &&
                        e.other >= 0 &&
                        e.other < static_cast<int>(table.pockets.size()))
                        a.objRest = table.pockets[e.other].pos;
        }
    }
    // Monte-Carlo success probability of exactly this recipe.
    a.pSuccess = accuracy::shotSuccessProbability(
        table, balls, cueIdx, dir, best.power, best.spinF, best.spinS,
        targetId);
    return a;
}

std::vector<ShotAdvice> adviseShots(const TableSpec& table,
                                    const std::vector<Ball>& balls,
                                    const std::vector<Shot>& shots, int cueIdx,
                                    int maxShots) {
    std::vector<ShotAdvice> out;
    if (cueIdx < 0) return out;
    int advised = 0;
    for (size_t i = 0; i < shots.size() &&
                        advised < std::max(1, maxShots); ++i) {
        if (!shots[i].feasible || shots[i].objectIdx < 0) continue;
        ShotAdvice a = adviseShot(table, balls, shots[i], cueIdx,
                                  static_cast<int>(i));
        if (a.power <= 0) continue;  // no executable recipe for this shot
        out.push_back(a);
        ++advised;
    }
    // Rank: scratch-free pot first, then Monte-Carlo probability, then the
    // solver's own confidence (cut angle / distance quality).
    std::sort(out.begin(), out.end(), [](const ShotAdvice& x,
                                         const ShotAdvice& y) {
        int tx = (x.potsTarget && !x.scratch) ? 0 : (x.potsTarget ? 1 : 2);
        int ty = (y.potsTarget && !y.scratch) ? 0 : (y.potsTarget ? 1 : 2);
        if (tx != ty) return tx < ty;
        if (std::fabs(x.pSuccess - y.pSuccess) > 0.02)
            return x.pSuccess > y.pSuccess;
        return x.potted > y.potted;
    });
    return out;
}

}  // namespace pool
