// geom.cpp — implementation of the geometry primitives.
#include "pool/geom.hpp"

namespace pool {

TableSpec TableSpec::standard() {
    TableSpec t;
    t.pockets = {
        {{0, 0},},            // top-left corner
        {{t.w / 2, 0},},      // top side
        {{t.w, 0},},          // top-right corner
        {{0, t.h},},          // bottom-left corner
        {{t.w / 2, t.h},},    // bottom side
        {{t.w, t.h},},        // bottom-right corner
    };
    return t;
}

double segPointDist(const Vec2& a, const Vec2& b, const Vec2& p) {
    Vec2 ab = b - a;
    double l2 = ab.len2();
    if (l2 < 1e-12) return (p - a).len();
    double t = (p - a).dot(ab) / l2;
    t = clampd(t, 0.0, 1.0);
    Vec2 proj = a + ab * t;
    return (p - proj).len();
}

bool segHitsCircle(const Vec2& a, const Vec2& b, const Vec2& c, double r) {
    return segPointDist(a, b, c) <= r;
}

bool segLineHit(const Vec2& a, const Vec2& b, const Vec2& lp, const Vec2& ld,
                Vec2& out) {
    Vec2 d = b - a;
    Vec2 L = ld.norm();
    double denom = d.cross(L);
    if (std::fabs(denom) < 1e-12) return false;  // parallel
    double t = (lp - a).cross(L) / denom;
    if (t < 0.0 || t > 1.0) return false;
    out = a + d * t;
    return true;
}

Vec2 reflectPoint(const Vec2& p, const Vec2& lp, const Vec2& ld) {
    Vec2 d = ld.norm();
    Vec2 v = p - lp;
    Vec2 proj = d * v.dot(d);
    return lp + (proj * 2.0 - v);
}

double degBetween(const Vec2& u, const Vec2& v) {
    Vec2 un = u.norm(), vn = v.norm();
    double c = clampd(un.dot(vn), -1.0, 1.0);
    return std::acos(c) * 180.0 / 3.14159265358979323846;
}

double clamp01(double v) { return clampd(v, 0.0, 1.0); }

double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace pool
