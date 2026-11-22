// geom.hpp — core geometry primitives for the pool vision trainer.
// Pure math, no dependencies. Logical table coordinates: the playing field
// is the rectangle [0..w] x [0..h]; ball centers stay inside it; cushion
// faces are the rectangle edges; pockets sit on the edges.
#pragma once
#include <cmath>
#include <string>
#include <vector>

namespace pool {

struct Vec2 {
    double x = 0, y = 0;
    Vec2() = default;
    Vec2(double xx, double yy) : x(xx), y(yy) {}

    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    Vec2 operator/(double s) const { return {x / s, y / s}; }
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }

    double dot(const Vec2& o) const { return x * o.x + y * o.y; }
    double cross(const Vec2& o) const { return x * o.y - y * o.x; }
    double len() const { return std::hypot(x, y); }
    double len2() const { return x * x + y * y; }
    double dist(const Vec2& o) const { return (*this - o).len(); }

    Vec2 norm() const {
        double l = len();
        return l > 1e-12 ? Vec2{x / l, y / l} : Vec2{0, 0};
    }
    // Rotate +90 degrees (counter-clockwise in math axes, "tangent" when
    // applied to a line-of-centers direction).
    Vec2 rot90() const { return {-y, x}; }
};

struct Ball {
    int id = -1;          // logical number: 0 = cue, 1..15, 8 = black
    std::string kind;     // "cue" | "solid" | "stripe" | "black"
    std::string color;    // color name
    Vec2 pos;
    double r = 10;
};

struct Pocket {
    Vec2 pos;
};

struct TableSpec {
    double w = 900, h = 450;   // cushion-face rectangle (logical)
    double ballR = 12;
    double pocketR = 22;
    std::vector<Pocket> pockets;
    static TableSpec standard();  // 6 pockets: 4 corners + 2 side midpoints
};

// ----- helpers -----------------------------------------------------------

// Minimum distance from point p to segment [a, b].
double segPointDist(const Vec2& a, const Vec2& b, const Vec2& p);

// True if the segment [a, b] passes within r of circle center c.
bool segHitsCircle(const Vec2& a, const Vec2& b, const Vec2& c, double r);

// Intersect segment [a, b] with the infinite line through lp with direction ld.
// Returns true and fills out when the crossing exists within the segment.
bool segLineHit(const Vec2& a, const Vec2& b, const Vec2& lp, const Vec2& ld,
                Vec2& out);

// Mirror point p across the line through lp with direction ld.
Vec2 reflectPoint(const Vec2& p, const Vec2& lp, const Vec2& ld);

// Angle between two vectors, degrees in [0, 180].
double degBetween(const Vec2& u, const Vec2& v);

double clamp01(double v);
double clampd(double v, double lo, double hi);

}  // namespace pool
