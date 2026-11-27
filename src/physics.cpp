// physics.cpp — implementation of the event-driven billiards simulation.
// Pure math on top of geom.hpp; deterministic; no external dependencies.
#include "pool/physics.hpp"
#include <algorithm>
#include <cmath>

namespace pool {
namespace {

struct S {
    Vec2 p, v;
    double sf = 0, ss = 0;
    bool active = true;
    int id = -1;
};

// True when the cue ball was involved in a ball-ball contact this step.
void applyCueSpinKick(S& cue, double vrel, double preDirX, double preDirY,
                      const PhysicsParams& p) {
    if (cue.sf == 0) return;
    double k = cue.sf > 0 ? p.followFactor * cue.sf : p.drawFactor * cue.sf;
    // Kick along the cue's pre-impact direction (forward for follow,
    // backward for draw).
    cue.v.x += preDirX * k * vrel;
    cue.v.y += preDirY * k * vrel;
    cue.sf = 0;  // spin is consumed by the contact
}

}  // namespace

SimResult simulate(const TableSpec& t, const std::vector<Ball>& balls,
                   const Vec2& cueVel, double spinF, double spinS,
                   const PhysicsParams& p) {
    SimResult res;
    res.paths.resize(balls.size());

    // Locate the cue ball (id 0 preferred, else kind "cue").
    int cueIdx = -1;
    for (size_t i = 0; i < balls.size(); ++i) {
        if (balls[i].id == 0) { cueIdx = static_cast<int>(i); break; }
        if (cueIdx < 0 && balls[i].kind == "cue") cueIdx = static_cast<int>(i);
    }
    if (cueIdx < 0) return res;

    std::vector<S> st;
    st.reserve(balls.size());
    for (size_t i = 0; i < balls.size(); ++i) {
        S s;
        s.p = balls[i].pos;
        s.v = Vec2{};
        s.id = balls[i].id;
        st.push_back(s);
    }
    st[cueIdx].v = cueVel;
    st[cueIdx].sf = clampd(spinF, -1.0, 1.0);
    st[cueIdx].ss = clampd(spinS, -1.0, 1.0);

    const double R = t.ballR;
    const double D = 2.0 * R;

    // Pre-pass: separate initially overlapping balls (detection jitter) so
    // the simulation does not fire phantom contacts at t=0.
    for (int iter = 0; iter < 8; ++iter) {
        bool moved = false;
        for (size_t i = 0; i < st.size(); ++i) {
            for (size_t j = i + 1; j < st.size(); ++j) {
                Vec2 d = st[j].p - st[i].p;
                double dist = d.len();
                if (dist >= D || dist < 1e-9) continue;
                Vec2 n = d.norm();
                double push = (D - dist) / 2.0;
                st[i].p = st[i].p - n * push;
                st[j].p += n * push;
                moved = true;
            }
        }
        if (!moved) break;
    }

    double time = 0;
    bool firstContactDone = false;
    for (int step = 0; step < static_cast<int>(p.maxTime / p.dt); ++step) {
        bool anyMoving = false;

        // Integrate motion: friction + spin decay.
        for (auto& b : st) {
            if (!b.active) continue;
            double sp = b.v.len();
            if (sp > p.stopSpeed) {
                double eff = p.frictionDecel * (1.0 - 0.30 * b.sf);
                eff = std::max(eff, p.frictionDecel * 0.40);
                double ns = sp - eff * p.dt;
                if (ns <= p.stopSpeed) {
                    b.v = Vec2{};
                    b.sf = 0;
                    b.ss = 0;
                    res.events.push_back({time, "rest", b.id, -1, b.p});
                } else {
                    b.v = b.v.norm() * ns;
                    b.p += b.v * p.dt;  // semi-implicit Euler position update
                    anyMoving = true;
                }
            }
            double decay = std::exp(-p.dt / p.spinDecayTau);
            b.sf *= decay;
            b.ss *= decay;
        }

        // Ball-ball collisions.
        for (size_t i = 0; i < st.size(); ++i) {
            if (!st[i].active) continue;
            for (size_t j = i + 1; j < st.size(); ++j) {
                if (!st[j].active) continue;
                Vec2 d = st[j].p - st[i].p;
                double dist = d.len();
                if (dist >= D || dist < 1e-9) continue;
                // Continuous contact: rewind the pair to the exact 2R
                // crossing inside this step. Without this, fast steps skip
                // the ghost point, tilt the line of centers by a couple of
                // degrees and the object misses the pocket.
                Vec2 dRel = (st[i].v - st[j].v) * p.dt;
                double A = dRel.len2();
                double u = 0.0;
                if (A > 1e-12) {
                    double B = dRel.dot(d);
                    double disc = B * B - A * (dist * dist - D * D);
                    if (disc > 0) u = clampd((std::sqrt(disc) - B) / A, 0.0, 1.0);
                }
                st[i].p = st[i].p - st[i].v * (p.dt * u);
                st[j].p = st[j].p - st[j].v * (p.dt * u);
                d = st[j].p - st[i].p;
                dist = d.len();
                if (dist < 1e-9) continue;
                Vec2 n = d.norm();
                double vrel = (st[i].v - st[j].v).dot(n);
                // Positional separation regardless of approach direction.
                double overlap = D - dist;
                st[i].p = st[i].p - n * (overlap / 2.0);
                st[j].p += n * (overlap / 2.0);
                if (vrel <= 0) continue;  // already separating

                // The spin kick follows the CUE's own pre-impact direction.
                // Identify the cue BY INDEX (located at the top of simulate),
                // not by id == 0: a caller may legitimately give the cue ball
                // a non-zero id, and then the follow/draw kick and the
                // first-contact bookkeeping silently vanished.
                Vec2 cuePre;
                bool cueIsI = i == cueIdx, cueIsJ = j == cueIdx;
                if (cueIsI) cuePre = st[i].v.len() > 1e-9 ? st[i].v.norm() : Vec2{};
                if (cueIsJ) cuePre = st[j].v.len() > 1e-9 ? st[j].v.norm() : Vec2{};

                double jImp = (1.0 + p.restitutionBall) / 2.0 * vrel;
                st[i].v = st[i].v - n * jImp;
                st[j].v += n * jImp;
                res.events.push_back({time, "ball-ball", st[i].id, st[j].id,
                                      (st[i].p + st[j].p) * 0.5});
                if (!firstContactDone) {
                    if (cueIsI) res.firstContact = st[j].id;
                    else if (cueIsJ) res.firstContact = st[i].id;
                    if (res.firstContact >= 0) firstContactDone = true;
                }
                // Spin effects live on the cue ball only.
                if (cueIsI && st[i].sf != 0)
                    applyCueSpinKick(st[i], vrel, cuePre.x, cuePre.y, p);
                if (cueIsJ && st[j].sf != 0)
                    applyCueSpinKick(st[j], vrel, cuePre.x, cuePre.y, p);
                anyMoving = true;
            }
        }

        // Cushions (rail indices: 0 top, 1 right, 2 bottom, 3 left).
        for (auto& b : st) {
            if (!b.active) continue;
            int rail = -1;
            Vec2 nrm;
            if (b.p.y < R) { rail = 0; nrm = Vec2{0, 1}; }
            else if (b.p.x > t.w - R) { rail = 1; nrm = Vec2{-1, 0}; }
            else if (b.p.y > t.h - R) { rail = 2; nrm = Vec2{0, -1}; }
            else if (b.p.x < R) { rail = 3; nrm = Vec2{1, 0}; }
            if (rail < 0) continue;

            double vn = b.v.dot(nrm);  // normal speed toward the rail (<0)
            // Push back inside and reflect the normal component.
            if (rail == 0) b.p.y = R;
            if (rail == 1) b.p.x = t.w - R;
            if (rail == 2) b.p.y = t.h - R;
            if (rail == 3) b.p.x = R;
            if (vn >= 0) continue;  // moving away already
            Vec2 vN = nrm * vn;
            Vec2 vT = b.v - vN;
            double vnNew = -vn * p.restitutionCushion;
            // Side english kicks the ball along the rail and decays.
            Vec2 tang = nrm.rot90();
            double kick = b.ss * p.sideCushionFactor * (-vn);
            Vec2 vTNew = vT + tang * kick;
            b.v = vTNew + nrm * vnNew;
            b.ss *= 0.5;
            res.events.push_back({time, "cushion", b.id, rail, b.p});
            if (b.v.len() > p.stopSpeed) anyMoving = true;
        }

        // Pockets.
        for (size_t i = 0; i < st.size(); ++i) {
            if (!st[i].active) continue;
            for (size_t k = 0; k < t.pockets.size(); ++k) {
                if (st[i].p.dist(t.pockets[k].pos) < p.pocketCapture) {
                    st[i].active = false;
                    st[i].v = Vec2{};
                    res.potted.push_back(st[i].id);
                    if (i == cueIdx) res.cueScratch = true;
                    res.events.push_back({time, "pot", st[i].id,
                                          static_cast<int>(k), st[i].p});
                    break;
                }
            }
        }

        // Path sampling (~60 Hz); stopped/potted balls stop appending.
        if (step % 4 == 0) {
            for (size_t i = 0; i < st.size(); ++i) {
                auto& path = res.paths[i];
                if (path.empty() || (path.back() - st[i].p).len2() > 1e-12)
                    path.push_back(st[i].p);
            }
        }

        time += p.dt;
        if (!anyMoving) break;
    }

    for (size_t i = 0; i < st.size(); ++i) {
        SimBall fb;
        fb.id = st[i].id;
        fb.pos = st[i].p;
        fb.vel = st[i].v;
        fb.spinF = st[i].sf;
        fb.spinS = st[i].ss;
        fb.active = st[i].active;
        res.final.push_back(fb);
    }
    res.duration = time;
    return res;
}

double minPotPower(const TableSpec& t, const std::vector<Ball>& balls,
                   const Vec2& cueDir, double spinF, double spinS,
                   int targetId, const PhysicsParams& p) {
    // Locate the cue ball with the same convention as simulate (id 0
    // preferred, else kind "cue") so a non-zero-id cue is not mistaken for
    // a successful pot.
    int cueId = 0;
    for (size_t i = 0; i < balls.size(); ++i) {
        if (balls[i].id == 0) { cueId = 0; break; }
        if (balls[i].kind == "cue") { cueId = balls[i].id; break; }
    }
    Vec2 dir = cueDir.len() > 1e-12 ? cueDir.norm() : Vec2{1, 0};
    for (double power = 0.10; power <= 1.001; power += 0.05) {
        Vec2 vel = dir * (p.maxShotSpeed * power);
        SimResult r = simulate(t, balls, vel, spinF, spinS, p);
        bool ok = false;
        for (int id : r.potted) {
            if (id == cueId) continue;  // a scratch does not count as success
            if (targetId >= 0 && id != targetId) continue;
            ok = true;
            break;
        }
        if (ok) return power;
    }
    return -1.0;
}

}  // namespace pool
