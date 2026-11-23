// stabilize.hpp — temporal smoothing for live overlay rendering. The
// detector re-fits the felt and re-finds balls every frame; single-pixel
// noise makes every drawn line tremble. This header glides the felt rect,
// the perspective quad, the ball radius and the ball positions over time,
// holds briefly-missed balls so guides do not blink, and snaps instantly
// when something really moved. Pure math, no dependencies.
#pragma once
#include "pool/vision.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace pool {

// Exponential glide between prev and cur. Snaps to cur when a corner
// moves more than snapFrac of the rect diagonal (a real table move or a
// re-fit, not noise).
inline Rect smoothRect(const Rect& prev, const Rect& cur, double alpha,
                       double snapFrac) {
    if (prev.w() <= 1 || prev.h() <= 1) return cur;
    double diag = std::hypot(cur.w(), cur.h());
    double jump = std::hypot(cur.x0 - prev.x0, cur.y0 - prev.y0);
    jump = std::max(jump, std::hypot(cur.x1 - prev.x1, cur.y1 - prev.y1));
    if (jump > snapFrac * diag) return cur;
    Rect o = cur;
    o.x0 = prev.x0 + (cur.x0 - prev.x0) * alpha;
    o.y0 = prev.y0 + (cur.y0 - prev.y0) * alpha;
    o.x1 = prev.x1 + (cur.x1 - prev.x1) * alpha;
    o.y1 = prev.y1 + (cur.y1 - prev.y1) * alpha;
    return o;
}

// Corner-wise glide for the perspective quad; the whole quad snaps when
// any corner jumps beyond snapFrac of the TL->BR diagonal.
inline Quad smoothQuad(const Quad& prev, const Quad& cur, double alpha,
                       double snapFrac) {
    const Vec2 ps[4] = {prev.tl, prev.tr, prev.br, prev.bl};
    const Vec2 cs[4] = {cur.tl, cur.tr, cur.br, cur.bl};
    double diag = std::hypot(cur.br.x - cur.tl.x, cur.br.y - cur.tl.y);
    double jump = 0;
    for (int i = 0; i < 4; ++i)
        jump = std::max(jump, std::hypot(cs[i].x - ps[i].x, cs[i].y - ps[i].y));
    if (jump > snapFrac * std::max(1.0, diag)) return cur;
    Quad o;
    Vec2 os[4] = {o.tl, o.tr, o.br, o.bl};
    for (int i = 0; i < 4; ++i) {
        os[i].x = ps[i].x + (cs[i].x - ps[i].x) * alpha;
        os[i].y = ps[i].y + (cs[i].y - ps[i].y) * alpha;
    }
    o.tl = os[0]; o.tr = os[1]; o.br = os[2]; o.bl = os[3];
    return o;
}

// Scalar glide (ball radius etc.); snaps on a relative jump.
inline double smoothScalar(double prev, double cur, double alpha,
                           double snapFrac = 0.25) {
    if (prev <= 0) return cur;
    if (std::fabs(cur - prev) > snapFrac * prev) return cur;
    return prev + (cur - prev) * alpha;
}

// Ball tracks: match fresh detections to previous ones by nearest position
// (within matchFactor * ballR), glide matched positions, hold vanished
// tracks for a few frames, and never let two cue balls coexist.
struct TrackedBall {
    Detection d;
    int age = 0;  // frames since this track last matched a fresh detection
    long id = 0;
};

class BallStabilizer {
public:
    BallStabilizer(int holdFrames = 5, double alpha = 0.5,
                   double matchFactor = 1.5)
        : holdFrames_(holdFrames), alpha_(alpha), matchFactor_(matchFactor) {}

    void reset() {
        tracks_.clear();
        nextId_ = 1;
    }

    void apply(std::vector<Detection>& balls, double ballR) {
        const double matchR = matchFactor_ * std::max(1.0, ballR);
        std::vector<int> assigned(balls.size(), -1);
        std::vector<char> used(tracks_.size(), 0);
        for (size_t i = 0; i < balls.size(); ++i) {
            int best = -1;
            double bd = 1e18;
            for (size_t t = 0; t < tracks_.size(); ++t) {
                if (used[t]) continue;
                double dd = std::hypot(balls[i].px.x - tracks_[t].d.px.x,
                                       balls[i].px.y - tracks_[t].d.px.y);
                if (dd < bd) {
                    bd = dd;
                    best = static_cast<int>(t);
                }
            }
            if (best >= 0 && bd <= matchR) {
                assigned[i] = best;
                used[best] = 1;
            }
        }
        std::vector<Detection> out;
        std::vector<TrackedBall> nt;
        out.reserve(balls.size() + tracks_.size());
        nt.reserve(out.capacity());
        for (size_t i = 0; i < balls.size(); ++i) {
            Detection d = balls[i];
            if (assigned[i] >= 0) {
                const TrackedBall& tr = tracks_[static_cast<size_t>(assigned[i])];
                d.px.x = tr.d.px.x + (d.px.x - tr.d.px.x) * alpha_;
                d.px.y = tr.d.px.y + (d.px.y - tr.d.px.y) * alpha_;
                d.logical.x =
                    tr.d.logical.x + (d.logical.x - tr.d.logical.x) * alpha_;
                d.logical.y =
                    tr.d.logical.y + (d.logical.y - tr.d.logical.y) * alpha_;
                // Kind flapping (unknown <-> known): prefer the known kind.
                if (d.kind == "unknown" && tr.d.kind != "unknown") {
                    d.kind = tr.d.kind;
                    d.color = tr.d.color;
                    d.number = tr.d.number;
                }
                nt.push_back(TrackedBall{d, 0, tr.id});
            } else {
                nt.push_back(TrackedBall{d, 0, nextId_++});
            }
            out.push_back(nt.back().d);
        }
        for (size_t t = 0; t < tracks_.size(); ++t) {
            if (used[t]) continue;
            TrackedBall h = tracks_[t];
            ++h.age;
            if (h.age > holdFrames_) continue;  // really gone
            nt.push_back(h);
            out.push_back(h.d);
        }
        // At most one cue: a freshly detected cue beats a held one.
        int freshCue = -1, heldCue = -1;
        for (size_t i = 0; i < out.size(); ++i) {
            if (out[i].kind != "cue") continue;
            if (nt[i].age == 0) {
                if (freshCue < 0) freshCue = static_cast<int>(i);
            } else if (heldCue < 0) {
                heldCue = static_cast<int>(i);
            }
        }
        if (freshCue >= 0 && heldCue >= 0) {
            out.erase(out.begin() + heldCue);
            nt.erase(nt.begin() + heldCue);
        }
        tracks_.swap(nt);
        balls.swap(out);
    }

private:
    int holdFrames_;
    double alpha_;
    double matchFactor_;
    long nextId_ = 1;
    std::vector<TrackedBall> tracks_;
};

}  // namespace pool
