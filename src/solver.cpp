// solver.cpp — ghost ball, mirror folding (bank/kick), obstruction, scoring.
#include "pool/solver.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace pool {

Solver::Solver(const TableSpec& table, const std::vector<Ball>& balls, int cueIdx,
               const SolverOptions& opt)
    : t_(table), balls_(balls), cueIdx_(cueIdx), opt_(opt) {}

void Solver::railLine(const TableSpec& t, int rail, Vec2& lp, Vec2& ld) {
    switch (rail) {
        case 0: lp = {0, 0}; ld = {1, 0}; break;       // top
        case 1: lp = {t.w, 0}; ld = {0, 1}; break;     // right
        case 2: lp = {0, t.h}; ld = {1, 0}; break;     // bottom
        default: lp = {0, 0}; ld = {0, 1}; break;      // left
    }
}

std::vector<Vec2> Solver::foldImages(const Vec2& P,
                                     const std::vector<int>& rails) const {
    // Fold from the LAST rail backwards: image across rails[N-1], then that
    // image across rails[N-2], ... The final image is reached by aiming
    // straight through the first rail.
    std::vector<Vec2> imgs{P};
    for (auto it = rails.rbegin(); it != rails.rend(); ++it) {
        Vec2 lp, ld;
        railLine(t_, *it, lp, ld);
        imgs.push_back(reflectPoint(imgs.back(), lp, ld));
    }
    return imgs;  // imgs[0] = real point, imgs[N] = fully folded
}

bool Solver::unfoldPath(const Vec2& from, const std::vector<Vec2>& imgs,
                        const std::vector<int>& rails, std::vector<Vec2>& pts,
                        std::string& err) const {
    Vec2 cur = from;
    int targetIdx = static_cast<int>(imgs.size()) - 1;  // most folded image
    for (int rail : rails) {
        Vec2 lp, ld, hit;
        railLine(t_, rail, lp, ld);
        if (!segLineHit(cur, imgs[targetIdx], lp, ld, hit)) {
            err = "no crossing with rail";
            return false;
        }
        if (!bounceOk(hit, rail)) {
            err = "bounce outside rail or at pocket mouth";
            return false;
        }
        pts.push_back(hit);
        cur = hit;
        --targetIdx;
    }
    return true;
}

bool Solver::pathClear(const Vec2& a, const Vec2& b, int ignoreA,
                       int ignoreB) const {
    double clearance = 2.0 * t_.ballR * opt_.clearance;
    for (int i = 0; i < static_cast<int>(balls_.size()); ++i) {
        if (i == ignoreA || i == ignoreB || i == cueIdx_) continue;
        if (segHitsCircle(a, b, balls_[i].pos, clearance)) return false;
    }
    return true;
}

bool Solver::bounceOk(const Vec2& p, int rail) const {
    // Extent check: point must sit on the rail segment.
    if (rail == 0 || rail == 2) {
        if (p.x < 0 || p.x > t_.w) return false;
    } else {
        if (p.y < 0 || p.y > t_.h) return false;
    }
    // Pocket-mouth check: too close to any pocket center means the ball
    // would fall in (or the geometry is degenerate).
    for (const auto& pk : t_.pockets) {
        if (p.dist(pk.pos) < t_.pocketR + t_.ballR) return false;
    }
    return true;
}

bool Solver::inField(const Vec2& p) const {
    return p.x >= 0 && p.x <= t_.w && p.y >= 0 && p.y <= t_.h;
}

// --------------------------------------------------------------------------
// DIRECT: cue -> ghost -> object falls straight toward the pocket.
Shot Solver::solveDirect(int objIdx, int pocketIdx) const {
    Shot s;
    s.type = Shot::Type::Direct;
    s.typeName = "direct";
    s.objectIdx = objIdx;
    s.pocketIdx = pocketIdx;
    if (cueIdx_ < 0) { s.feasible = false; s.note = "no cue ball"; return s; }

    const Vec2 C = balls_[cueIdx_].pos;
    const Vec2 T = balls_[objIdx].pos;
    const Vec2 P = t_.pockets[pocketIdx].pos;
    const double R = t_.ballR;

    Vec2 dir = (P - T).norm();              // object departure direction
    Vec2 ghost = T - dir * (2 * R);         // ghost ball center
    s.ghost = ghost;
    s.objPath = {T, P};
    s.cuePath = {C, ghost};

    if (!inField(ghost)) { s.feasible = false; s.note = "ghost outside field"; return s; }

    Vec2 aim = (ghost - C).norm();
    double cut = degBetween(aim, dir);
    s.cutDeg = cut;
    if (cut > opt_.maxCutDeg) { s.feasible = false; s.note = "cut too thin"; return s; }

    if (!pathClear(C, ghost, cueIdx_, objIdx)) { s.feasible = false; s.note = "cue path blocked"; return s; }
    if (!pathClear(T, P, objIdx, cueIdx_)) { s.feasible = false; s.note = "object path blocked"; return s; }

    double dCG = C.dist(ghost), dTP = T.dist(P);
    s.confidence = clamp01(0.45 * (1 - cut / 90.0) +
                           0.30 * (1 - dCG / (t_.w + t_.h)) +
                           0.25 * (1 - dTP / (t_.w + t_.h)));
    // 90-degree rule: cue departs along the tangent of the line of centers.
    s.tangentDir = dir.rot90();
    return s;
}

// --------------------------------------------------------------------------
// BANK: cue -> object ball, which then bounces off `rails` into the pocket.
Shot Solver::solveBank(int objIdx, int pocketIdx,
                       const std::vector<int>& rails) const {
    Shot s;
    s.type = Shot::Type::Bank;
    s.typeName = "bank";
    s.objectIdx = objIdx;
    s.pocketIdx = pocketIdx;
    if (cueIdx_ < 0) { s.feasible = false; s.note = "no cue ball"; return s; }

    const Vec2 C = balls_[cueIdx_].pos;
    const Vec2 T = balls_[objIdx].pos;
    const Vec2 P = t_.pockets[pocketIdx].pos;
    const double R = t_.ballR;
    const int N = static_cast<int>(rails.size());

    // Fold the pocket image; the object ball must depart toward the folded
    // pocket so that after unfolding it reaches the real pocket.
    std::vector<Vec2> imgs = foldImages(P, rails);  // imgs[0]=P .. imgs[N]=folded
    Vec2 folded = imgs.back();
    Vec2 dep = (folded - T).norm();                 // departure direction
    Vec2 ghost = T - dep * (2 * R);
    s.ghost = ghost;

    if (!inField(ghost)) { s.feasible = false; s.note = "ghost outside field"; return s; }

    Vec2 aim = (ghost - C).norm();
    double cut = degBetween(aim, dep);
    s.cutDeg = cut;
    if (cut > opt_.maxCutDeg) { s.feasible = false; s.note = "cut too thin"; return s; }

    if (!pathClear(C, ghost, cueIdx_, objIdx)) { s.feasible = false; s.note = "cue path blocked"; return s; }

    // Unfold the object path: T -> bounce(s) -> P.
    std::vector<Vec2> hits;
    std::string err;
    if (!unfoldPath(T, imgs, rails, hits, err)) { s.feasible = false; s.note = err; return s; }
    s.objPath.push_back(T);
    for (auto& h : hits) s.objPath.push_back(h);
    s.objPath.push_back(P);

    for (size_t i = 0; i + 1 < s.objPath.size(); ++i) {
        if (!pathClear(s.objPath[i], s.objPath[i + 1], objIdx, cueIdx_)) {
            s.feasible = false; s.note = "object path blocked"; return s;
        }
    }

    double objLen = 0, cueLen = C.dist(ghost);
    for (size_t i = 0; i + 1 < s.objPath.size(); ++i)
        objLen += s.objPath[i].dist(s.objPath[i + 1]);
    s.confidence = clamp01(0.40 * (1 - cut / 90.0) +
                           0.35 * (1 - objLen / (t_.w + t_.h)) +
                           0.25 * (1 - cueLen / (t_.w + t_.h)) -
                           0.08 * (N - 1));
    s.tangentDir = dep.rot90();
    return s;
}

// --------------------------------------------------------------------------
// KICK: cue bounces off `rails` and then hits the object ball.
Shot Solver::solveKick(int objIdx, const std::vector<int>& rails) const {
    Shot s;
    s.type = Shot::Type::Kick;
    s.typeName = "kick";
    s.objectIdx = objIdx;
    s.pocketIdx = -1;
    if (cueIdx_ < 0) { s.feasible = false; s.note = "no cue ball"; return s; }

    const Vec2 C = balls_[cueIdx_].pos;
    const Vec2 T = balls_[objIdx].pos;
    const int N = static_cast<int>(rails.size());

    std::vector<Vec2> imgs = foldImages(T, rails);
    std::vector<Vec2> hits;
    std::string err;
    if (!unfoldPath(C, imgs, rails, hits, err)) { s.feasible = false; s.note = err; return s; }

    s.cuePath.push_back(C);
    for (auto& h : hits) s.cuePath.push_back(h);
    s.cuePath.push_back(T);
    s.ghost = T;  // contact point at the object ball

    for (size_t i = 0; i + 1 < s.cuePath.size(); ++i) {
        if (!pathClear(s.cuePath[i], s.cuePath[i + 1], cueIdx_, objIdx)) {
            s.feasible = false; s.note = "cue path blocked"; return s;
        }
    }

    double pathLen = 0;
    for (size_t i = 0; i + 1 < s.cuePath.size(); ++i)
        pathLen += s.cuePath[i].dist(s.cuePath[i + 1]);
    s.cutDeg = 0;
    s.confidence = clamp01(0.75 * (1 - pathLen / (1.8 * (t_.w + t_.h))) *
                           std::pow(0.85, N - 1));
    s.tangentDir = (T - s.cuePath[s.cuePath.size() - 2]).norm().rot90();
    return s;
}

// --------------------------------------------------------------------------
std::vector<std::vector<int>> railSequences(int maxRails) {
    std::vector<std::vector<int>> out;
    for (int a = 0; a < 4; ++a) out.push_back({a});
    if (maxRails >= 2) {
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
                if (a != b) out.push_back({a, b});
    }
    if (maxRails >= 3) {
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) {
                if (a == b) continue;
                for (int c = 0; c < 4; ++c)
                    if (b != c) out.push_back({a, b, c});
            }
    }
    return out;
}

std::vector<Shot> Solver::allShots() const {
    std::vector<Shot> out;
    for (int obj = 0; obj < static_cast<int>(balls_.size()); ++obj) {
        if (obj == cueIdx_) continue;
        for (int pk = 0; pk < static_cast<int>(t_.pockets.size()); ++pk) {
            Shot d = solveDirect(obj, pk);
            if (d.feasible) out.push_back(d);
            for (const auto& rails : railSequences(opt_.maxRails)) {
                Shot b = solveBank(obj, pk, rails);
                if (b.feasible) out.push_back(b);
            }
        }
        for (const auto& rails : railSequences(opt_.maxRails)) {
            Shot k = solveKick(obj, rails);
            if (k.feasible) out.push_back(k);
        }
    }
    std::sort(out.begin(), out.end(), [](const Shot& a, const Shot& b) {
        return a.confidence > b.confidence;
    });
    return out;
}

}  // namespace pool
