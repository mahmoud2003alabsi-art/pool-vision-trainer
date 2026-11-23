// vision.hpp — dependency-free computer vision: PNG I/O (stb), HSV color
// analysis, felt/table segmentation, ball blob detection and HSV
// classification (cue / solid / stripe / black + color name + number guess).
#pragma once
#include "pool/geom.hpp"
#include <string>
#include <vector>

namespace pool {

struct Image {
    int w = 0, h = 0;
    std::vector<unsigned char> px;  // RGB8, row-major, 3 bytes per pixel

    bool empty() const { return w <= 0 || h <= 0 || px.empty(); }
    void set(int x, int y, unsigned char r, unsigned char g, unsigned char b) {
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        unsigned char* p = &px[(static_cast<size_t>(y) * w + x) * 3];
        p[0] = r; p[1] = g; p[2] = b;
    }
};

struct Hsv { double h = 0; double s = 0; double v = 0; };  // h in [0,360)

Hsv rgbToHsv(unsigned char r, unsigned char g, unsigned char b);

// Throws std::runtime_error on failure.
Image loadImage(const std::string& path);
bool saveImage(const std::string& path, const Image& img);

struct Rect {
    double x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    double w() const { return x1 - x0; }
    double h() const { return y1 - y0; }
    bool contains(double x, double y) const {
        return x >= x0 && x <= x1 && y >= y0 && y <= y1;
    }
    Vec2 corner(int i) const;  // 0..3 clockwise from top-left
};

struct Detection {
    Vec2 px;          // centroid, image pixels
    Vec2 logical;     // mapped into logical table coordinates
    double rPx = 0;   // estimated radius in pixels
    std::string kind; // "cue" | "solid" | "stripe" | "black"
    std::string color;
    int number = -1;  // 0 = cue, 8 = black, else best color guess (1-7 / 9-15)
    double whiteRatio = 0, sat = 0, val = 0, hue = 0;
    double rimColored = 0;  // colored fraction of the 0.68R..0.95R ring:
                            // high for a stripe ball showing its white cap
    double area = 0;
};

struct DetectOptions {
    double feltHueMin = 55, feltHueMax = 185;  // green felt hue window
    double feltSatMin = 0.30, feltValMin = 0.10;
    double cueSatMax = 0.25, cueValMin = 0.72;  // white ball
    double blackValMax = 0.14;                  // black ball / pocket hole
    double stripeWhiteMin = 0.42;               // stripe ball white fraction
    double areaMin = 20;                        // px^2, ignore tiny blobs
    double areaLoFactor = 0.30, areaHiFactor = 2.4;  // around median blob area
    // Dominant-surface (non-green felt) match windows, calibrated on real
    // 8BP window captures: felt s 0.52..0.87 / v 0.58..0.81 around the
    // dominant bin, while the navy background (v~0.26) and dark UI chrome
    // (s~0.18) must stay outside.
    double feltTolS = 0.20, feltTolV = 0.30;
    // Manual overrides ("edit anything" workflow, set via tune/overlay ini):
    double forceBallR = 0;       // px, 0 = auto
    Rect forceRect{0, 0, 0, 0};  // px rect, w<=0 = auto-detect
    bool stripLines = true;      // morphological aim-line removal
};

// Felt quad for perspective tables (Pool Club style trapezoids): corners in
// image pixels, TL, TR, BR, BL order.
struct Quad {
    Vec2 tl, tr, br, bl;
};

// Bilinear mapping between a felt quad and the logical table. The inverse
// solves the bilinear system with a damped Newton iteration, so shots
// computed in logical space land exactly on the drawn table even when the
// game renders it with perspective.
Vec2 quadLogicalToPx(const Quad& q, const Vec2& l, double logicalW,
                     double logicalH);
Vec2 quadPxToLogical(const Quad& q, const Vec2& p, double logicalW,
                     double logicalH);

struct DetectResult {
    std::vector<Detection> balls;
    Rect feltRect;      // detected playing field, image pixels
    double ballRPx = 0; // median blob radius
    TableSpec table;    // logical table used for coordinate mapping
    bool hasQuad = false;  // perspective-corrected felt quad available
    Quad quadPx{};         // TL,TR,BR,BL in image pixels (hasQuad only)
};

// Detect the table and the balls. `logical` selects the logical table the
// pixel coordinates are mapped into (default: TableSpec::standard()).
DetectResult detectBalls(const Image& img, const DetectOptions& opt = {},
                         const TableSpec& logical = TableSpec::standard());

// Estimate the in-game aim direction from the white guideline radiating from
// the cue ball (games draw it while aiming). Returns true and fills dirPx
// (unit vector in image pixel axes, y down) when a line-like white ray is
// found; otherBallPx are pixel positions to skip while sampling the ray.
bool estimateAimDirPx(const Image& img, Vec2 cuePx, double ballRPx,
                      const std::vector<Vec2>& otherBallPx, Vec2& dirPx);

}  // namespace pool
