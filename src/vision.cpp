// vision.cpp — segmentation, blob detection and HSV ball classification.
//
// Pipeline:
//   1. RGB -> HSV; build the surface mask (green felt window; if no
//      table-sized green region exists, the dominant surface color).
//   2. Largest 4-connected surface component -> playing-field rect.
//   3. Inside the rect: "ball-ish" pixels = NOT surface.
//   4. 8-connected components over ball-ish pixels -> blobs.
//   5. Median blob area -> ball radius estimate; keep blobs of plausible size.
//   6. Exclude blobs centered on the six pocket anchors.
//   7. Classify each blob by HSV statistics inside the disc.
//   8. Map centroids to logical table coordinates.
#include "pool/vision.hpp"
#include "stb_image.h"
#include "stb_image_write.h"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <numeric>
#include <stdexcept>

namespace pool {

Hsv rgbToHsv(unsigned char r8, unsigned char g8, unsigned char b8) {
    double r = r8 / 255.0, g = g8 / 255.0, b = b8 / 255.0;
    double mx = std::max({r, g, b}), mn = std::min({r, g, b});
    double d = mx - mn;
    Hsv out;
    out.v = mx;
    out.s = mx > 1e-9 ? d / mx : 0.0;
    if (d < 1e-9) {
        out.h = 0;
    } else if (mx == r) {
        double h = 60.0 * std::fmod((g - b) / d, 6.0);
        out.h = h < 0 ? h + 360.0 : h;
    } else if (mx == g) {
        out.h = 60.0 * (((b - r) / d) + 2.0);
    } else {
        out.h = 60.0 * (((r - g) / d) + 4.0);
    }
    return out;
}

Image loadImage(const std::string& path) {
    int w = 0, h = 0, comp = 0;
    stbi_uc* data = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (!data) throw std::runtime_error("cannot load image: " + path +
                                        " (" + (stbi_failure_reason() ? stbi_failure_reason() : "?") + ")");
    Image img;
    img.w = w; img.h = h;
    size_t n = static_cast<size_t>(w) * h * 3;
    img.px.assign(data, data + n);
    stbi_image_free(data);
    return img;
}

bool saveImage(const std::string& path, const Image& img) {
    if (img.empty()) return false;
    return stbi_write_png(path.c_str(), img.w, img.h, 3, img.px.data(),
                          img.w * 3) != 0;
}

Vec2 Rect::corner(int i) const {
    switch (i & 3) {
        case 0: return {x0, y0};
        case 1: return {x1, y0};
        case 2: return {x1, y1};
        default: return {x0, y1};
    }
}

namespace {

// Iterative flood fill (4-connectivity), returns the bounding rect and area.
void floodFill(const std::vector<uint8_t>& mask, int w, int h, int sx, int sy,
               std::vector<uint8_t>& visited, Rect& rect, long& area) {
    std::vector<int> stack;
    stack.push_back(sy * w + sx);
    visited[sy * w + sx] = 1;
    int x0 = sx, x1 = sx, y0 = sy, y1 = sy;
    area = 0;
    while (!stack.empty()) {
        int idx = stack.back();
        stack.pop_back();
        int x = idx % w, y = idx / w;
        ++area;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
        const int dx[4] = {1, -1, 0, 0};
        const int dy[4] = {0, 0, 1, -1};
        for (int k = 0; k < 4; ++k) {
            int nx = x + dx[k], ny = y + dy[k];
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            int nidx = ny * w + nx;
            if (mask[nidx] && !visited[nidx]) {
                visited[nidx] = 1;
                stack.push_back(nidx);
            }
        }
    }
    rect = {static_cast<double>(x0), static_cast<double>(y0),
            static_cast<double>(x1 + 1), static_cast<double>(y1 + 1)};
}

// 8-connected labeling of the ball-ish mask. Returns blobs with centroid,
// area and radius (as if the blob were a disc).
struct Blob {
    double cx = 0, cy = 0;
    long area = 0;
    Rect bbox;
};

std::vector<Blob> findBlobs(const std::vector<uint8_t>& mask, int w, int h) {
    std::vector<Blob> out;
    std::vector<uint8_t> visited(mask.size(), 0);
    const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    std::vector<int> stack;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            if (!mask[idx] || visited[idx]) continue;
            Blob b;
            long sumX = 0, sumY = 0;
            int x0 = x, x1 = x, y0 = y, y1 = y;
            stack.push_back(idx);
            visited[idx] = 1;
            while (!stack.empty()) {
                int cur = stack.back();
                stack.pop_back();
                int cx = cur % w, cy = cur / w;
                ++b.area;
                sumX += cx;
                sumY += cy;
                if (cx < x0) x0 = cx;
                if (cx > x1) x1 = cx;
                if (cy < y0) y0 = cy;
                if (cy > y1) y1 = cy;
                for (int k = 0; k < 8; ++k) {
                    int nx = cx + dx[k], ny = cy + dy[k];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    int nidx = ny * w + nx;
                    if (mask[nidx] && !visited[nidx]) {
                        visited[nidx] = 1;
                        stack.push_back(nidx);
                    }
                }
            }
            b.cx = static_cast<double>(sumX) / b.area;
            b.cy = static_cast<double>(sumY) / b.area;
            b.bbox = {static_cast<double>(x0), static_cast<double>(y0),
                      static_cast<double>(x1 + 1), static_cast<double>(y1 + 1)};
            out.push_back(b);
        }
    }
    return out;
}

std::string colorName(double hue, double coloredVal) {
    if (hue < 0.0) return "unknown";  // no colored pixels: hue is garbage
    // Bins calibrated on measured 8 Ball Pool renderings: yellow ball 1
    // sits at hue 41..62, orange ball 5 at 15..41 (only ~10 deg apart),
    // purple spans 255..345 (measured 272 and 316), maroon is a dark red
    // (measured v 0.46).
    struct Range { const char* name; double lo, hi; };
    static const Range ranges[] = {
        {"red", 345.0, 360.0}, {"red", 0.0, 15.0},
        {"orange", 15.0, 41.0},
        {"yellow", 41.0, 62.0},
        {"green", 62.0, 165.0},
        {"blue", 165.0, 255.0},
        {"purple", 255.0, 345.0},
    };
    // Dark red = maroon (ball 7); 8BP renders it at v ~0.46.
    bool inRed = (hue >= 345.0 || hue < 15.0);
    if (inRed && coloredVal < 0.55) return "maroon";
    for (const auto& rg : ranges) {
        if (hue >= rg.lo && hue < rg.hi) return rg.name;
    }
    return "unknown";
}

int numberForColor(const std::string& color, bool stripe) {
    int base;
    if (color == "yellow") base = 1;
    else if (color == "blue") base = 2;
    else if (color == "red") base = 3;
    else if (color == "purple") base = 4;
    else if (color == "orange") base = 5;
    else if (color == "green") base = 6;
    else if (color == "maroon") base = 7;
    else base = -1;
    if (base < 0) return -1;
    return stripe ? base + 8 : base;
}

// Largest 4-connected component of `mask`; returns its bounding rect.
Rect largestComponent(const std::vector<uint8_t>& mask, int w, int h,
                      long& areaOut) {
    std::vector<uint8_t> visited(mask.size(), 0);
    Rect best{0, 0, 0, 0};
    long bestArea = 0;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] && !visited[i]) {
            Rect r;
            long a = 0;
            floodFill(mask, w, h, static_cast<int>(i % w),
                      static_cast<int>(i / w), visited, r, a);
            if (a > bestArea) { bestArea = a; best = r; }
        }
    }
    areaOut = bestArea;
    return best;
}

// A pool-table rect must be big enough and roughly 2:1 (games crop a bit,
// phone screenshots stretch) — guards against green/dominant UI accents.
bool plausibleTableRect(const Rect& r, int w, int h) {
    if (r.w() <= 0 || r.h() <= 0) return false;
    double aspect = r.w() / r.h();
    return r.w() * r.h() >= 0.06 * w * h && aspect >= 1.15 && aspect <= 2.8;
}

// Fallback for non-green tables (game screenshots with dark/gray felt):
// quantize RGB, take the dominant surface color, mask everything within a
// small tolerance and return the largest component's rect.
Rect dominantSurfaceRect(const Image& img, std::vector<uint8_t>& surface,
                         const DetectOptions& opt, Hsv* dhOut = nullptr) {
    const int w = img.w, h = img.h;
    const size_t n = static_cast<size_t>(w) * h;
    std::vector<long> hist(16 * 16 * 16, 0);
    for (size_t i = 0; i < n; ++i) {
        int r = img.px[i * 3] >> 4, g = img.px[i * 3 + 1] >> 4,
            b = img.px[i * 3 + 2] >> 4;
        ++hist[(r << 8) | (g << 4) | b];
    }
    // Tiered dominant-bin search: felts are saturated and mid-bright, while
    // the game window's dark UI chrome and navy background are the biggest
    // regions in windowed captures and must never win. Try (bright+saturated)
    // first, then mid, then the plain dominant bin.
    //
    // The count for each bin is SMOOTHED over its 26 RGB neighbors first:
    // a gradient felt straddles several adjacent 16-step bins while a flat
    // rail region lands in a single one, so raw counts let a smaller flat
    // region beat the felt (this mis-picked red rails on one synthetic
    // scene, then poisoned every median downstream). Summing the ridge
    // recovers the true dominant COLOR REGION regardless of bin split —
    // and makes the pick robust to gamma/HDR shifts on other screens.
    static const double kPassV[3] = {0.45, 0.30, 0.07};
    static const double kPassS[3] = {0.30, 0.22, 0.0};
    std::vector<long> smooth(16 * 16 * 16, 0);
    for (int bin = 0; bin < 16 * 16 * 16; ++bin) {
        long sum = 0;
        int r0 = (bin >> 8) & 15, g0 = (bin >> 4) & 15, b0 = bin & 15;
        for (int dr = -1; dr <= 1; ++dr)
            for (int dg = -1; dg <= 1; ++dg)
                for (int db = -1; db <= 1; ++db) {
                    int r = r0 + dr, g = g0 + dg, b = b0 + db;
                    if (r < 0 || g < 0 || b < 0 || r > 15 || g > 15 ||
                        b > 15)
                        continue;
                    sum += hist[(r << 8) | (g << 4) | b];
                }
        smooth[bin] = sum;
    }
    int bestBin = -1;
    long bestCnt = 0;
    for (int pass = 0; pass < 3 && bestBin < 0; ++pass) {
        for (int bin = 0; bin < 16 * 16 * 16; ++bin) {
            int r = ((bin >> 8) & 15) * 16 + 8;
            int g = ((bin >> 4) & 15) * 16 + 8;
            int b = (bin & 15) * 16 + 8;
            Hsv c = rgbToHsv(static_cast<unsigned char>(r),
                             static_cast<unsigned char>(g),
                             static_cast<unsigned char>(b));
            bool dark = c.v < kPassV[pass];
            bool white = c.s < 0.10 && c.v > 0.93;
            bool dull = c.s < kPassS[pass];
            if (dark || white || dull) continue;
            if (smooth[bin] > bestCnt) { bestCnt = smooth[bin]; bestBin = bin; }
        }
    }
    // Match in HSV: shaded felts vary a lot in brightness (vignette) and get
    // bluer (higher saturation) as they darken, while chroma direction stays
    // stable. Compare saturation+value with generous windows, require a small
    // absolute brightness floor (holes/background are near-black), and check
    // hue only when both colors are actually colored.
    Hsv dh = rgbToHsv(static_cast<unsigned char>(((bestBin >> 8) & 15) * 16 + 8),
                      static_cast<unsigned char>(((bestBin >> 4) & 15) * 16 + 8),
                      static_cast<unsigned char>((bestBin & 15) * 16 + 8));
    for (size_t i = 0; i < n; ++i) {
        Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1], img.px[i * 3 + 2]);
        if (c.v < 0.10) continue;  // holes, deep shadows, black background
        // White balls and the aim guideline must never count as surface:
        // on light felts their s/v sit inside the felt's match windows and
        // the cue ball would silently vanish from the detection.
        if (c.s < 0.12 && c.v > 0.72) continue;
        if (std::fabs(c.s - dh.s) > opt.feltTolS) continue;
        if (std::fabs(c.v - dh.v) > opt.feltTolV) continue;
        if (c.s >= 0.12 && dh.s >= 0.12) {
            double hd = std::fabs(c.h - dh.h);
            hd = std::min(hd, 360.0 - hd);
            if (hd > 35.0) continue;
        }
        surface[i] = 1;
    }
    if (dhOut) *dhOut = dh;
    long area = 0;
    return largestComponent(surface, w, h, area);
}

// Largest inscribed disc of a ballish region around `box` (chamfer 3-4
// distance transform, box grown so surrounding felt seeds the transform).
// Returns the radius (capped at `cap`) and the disc center, or -1.
double inscribedDisc(const std::vector<uint8_t>& m, int w, int h, const Rect& box,
                     double cap, Vec2& center) {
    int x0 = std::max(0, static_cast<int>(box.x0));
    int y0 = std::max(0, static_cast<int>(box.y0));
    int x1 = std::min(w - 1, static_cast<int>(box.x1));
    int y1 = std::min(h - 1, static_cast<int>(box.y1));
    if (x1 < x0 || y1 < y0) return -1.0;
    const int rw = x1 - x0 + 1, rh = y1 - y0 + 1;
    std::vector<float> dist(static_cast<size_t>(rw) * rh, 0.0f);
    auto at = [&](int x, int y) -> float& {
        return dist[static_cast<size_t>(y - y0) * rw + (x - x0)];
    };
    auto masked = [&](int x, int y) {
        return m[static_cast<size_t>(y) * w + x] != 0;
    };
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x) {
            if (!masked(x, y)) { at(x, y) = 0.0f; continue; }
            float d = 1e9f;
            if (x - 1 >= x0) d = std::min(d, at(x - 1, y) + 3.0f);
            if (y - 1 >= y0) d = std::min(d, at(x, y - 1) + 3.0f);
            if (x - 1 >= x0 && y - 1 >= y0)
                d = std::min(d, at(x - 1, y - 1) + 4.0f);
            if (x + 1 <= x1 && y - 1 >= y0)
                d = std::min(d, at(x + 1, y - 1) + 4.0f);
            at(x, y) = d;
        }
    double best = -1.0;
    for (int y = y1; y >= y0; --y)
        for (int x = x1; x >= x0; --x) {
            if (!masked(x, y)) continue;
            float d = at(x, y);
            if (x + 1 <= x1) d = std::min(d, at(x + 1, y) + 3.0f);
            if (y + 1 <= y1) d = std::min(d, at(x, y + 1) + 3.0f);
            if (x + 1 <= x1 && y + 1 <= y1)
                d = std::min(d, at(x + 1, y + 1) + 4.0f);
            if (x - 1 >= x0 && y + 1 <= y1)
                d = std::min(d, at(x - 1, y + 1) + 4.0f);
            // Write the backward result back: without this the right/down
            // boundary never propagates and every disc center drifts
            // toward the bottom-right of its ball by up to ~0.7R (the
            // depth ends up measured against the left/top boundary only).
            at(x, y) = d;
            float r = d / 3.0f;
            if (r > best && r <= cap) {
                best = r;
                center = {static_cast<double>(x), static_cast<double>(y)};
            }
        }
    return best;
}

}  // namespace

// ---- perspective-corrected mapping (Pool Club style trapezoid felts) ----
// (exported: declared in vision.hpp, used by tests and the overlay too)

Vec2 quadLogicalToPx(const Quad& q, const Vec2& l, double lw, double lh) {
    double u = clampd(l.x / lw, 0.0, 1.0);
    double v = clampd(l.y / lh, 0.0, 1.0);
    double w00 = (1 - u) * (1 - v), w10 = u * (1 - v);
    double w01 = (1 - u) * v, w11 = u * v;
    return {q.tl.x * w00 + q.tr.x * w10 + q.bl.x * w01 + q.br.x * w11,
            q.tl.y * w00 + q.tr.y * w10 + q.bl.y * w01 + q.br.y * w11};
}

Vec2 quadPxToLogical(const Quad& q, const Vec2& p, double lw, double lh) {
    // Newton on F(u,v) = bilinear(u,v) - p with an analytic Jacobian.
    double u = 0.5, v = 0.5;
    for (int it = 0; it < 12; ++it) {
        double a = 1 - u, b = 1 - v;
        double fx = q.tl.x * a * b + q.tr.x * u * b + q.bl.x * a * v +
                    q.br.x * u * v - p.x;
        double fy = q.tl.y * a * b + q.tr.y * u * b + q.bl.y * a * v +
                    q.br.y * u * v - p.y;
        double dxdu = (q.tr.x - q.tl.x) * b + (q.br.x - q.bl.x) * v;
        double dxdv = (q.bl.x - q.tl.x) * a + (q.br.x - q.tr.x) * u;
        double dydu = (q.tr.y - q.tl.y) * b + (q.br.y - q.bl.y) * v;
        double dydv = (q.bl.y - q.tl.y) * a + (q.br.y - q.tr.y) * u;
        double det = dxdu * dydv - dxdv * dydu;
        if (std::fabs(det) < 1e-9) break;
        double du = (fx * dydv - fy * dxdv) / det;
        double dv = (fy * dxdu - fx * dydu) / det;
        u = clampd(u - du, -0.2, 1.2);
        v = clampd(v - dv, -0.2, 1.2);
        if (std::fabs(du) + std::fabs(dv) < 1e-10) break;
    }
    return {clampd(u, 0.0, 1.0) * lw, clampd(v, 0.0, 1.0) * lh};
}

// Fit the felt's left/right edges as straight lines over the per-row spans
// of the surface mask. Perspective tables draw the felt as a trapezoid, so
// the inscribed rectangle hugs the narrow top edge and squeezes the whole
// coordinate mapping; the quad restores it. Flat tables degenerate to the
// rectangle (slopes ~0) and are rejected by the trapezoid gate.
bool fitTrapezoid(const std::vector<uint8_t>& surface, int w, int h,
                  const Rect& seed, Quad& q) {
    int y0 = std::max(0, static_cast<int>(seed.y0) - 2);
    int y1 = std::min(h - 1, static_cast<int>(seed.y1) + 2);
    int wx0 = std::max(0, static_cast<int>(seed.x0 - 0.30 * seed.w()) - 8);
    int wx1 = std::min(w - 1, static_cast<int>(seed.x1 + 0.30 * seed.w()) + 8);
    std::vector<double> xmin, xmax, ys;
    for (int y = y0; y <= y1; ++y) {
        int lo = -1, hi = -1;
        for (int x = wx0; x <= wx1; ++x)
            if (surface[static_cast<size_t>(y) * w + x]) {
                if (lo < 0) lo = x;
                hi = x;
            }
        if (lo < 0) continue;
        xmin.push_back(lo);
        xmax.push_back(hi);
        ys.push_back(y);
    }
    if (xmin.size() < 20) return false;
    std::vector<double> spans;
    for (size_t i = 0; i < xmin.size(); ++i)
        spans.push_back(xmax[i] - xmin[i]);
    std::vector<double> srt = spans;
    std::sort(srt.begin(), srt.end());
    double medSpan = srt[srt.size() / 2];
    double maxSpan = srt.back();
    // Least squares x = a + b*y over rows near the median span (felt body).
    double lsx = 0, lsy = 0, lsyy = 0, lsxy = 0, rsx = 0, rsxy = 0;
    long lcnt = 0, rcnt = 0;
    for (size_t i = 0; i < ys.size(); ++i) {
        if (spans[i] < 0.45 * medSpan) continue;
        double y = ys[i];
        lsx += xmin[i];
        lsy += y;
        lsyy += y * y;
        lsxy += xmin[i] * y;
        ++lcnt;
        rsx += xmax[i];
        rsxy += xmax[i] * y;
        ++rcnt;
    }
    if (lcnt < 20 || rcnt < 20) return false;
    auto fit = [](double sx, double sy, double syy, double sxy, long n,
                  double& a, double& b) {
        double d = static_cast<double>(n) * syy - sy * sy;
        if (std::fabs(d) < 1e-6) {
            a = sx / n;
            b = 0;
            return;
        }
        b = (static_cast<double>(n) * sxy - sx * sy) / d;
        a = (sx - b * sy) / n;
    };
    double la, lb, ra, rb;
    fit(lsx, lsy, lsyy, lsxy, lcnt, la, lb);
    fit(rsx, lsy, lsyy, rsxy, rcnt, ra, rb);
    if (std::fabs(lb) > 1.2 || std::fabs(rb) > 1.2) return false;
    // Top/bottom of the felt: first/last row with a real span. A low gate
    // (25%) is required here — a strong trapezoid's narrow edge can be well
    // below half of the wide edge, and the quad must still start there.
    double yTop = -1, yBot = -1;
    for (size_t i = 0; i < ys.size(); ++i) {
        if (spans[i] < 0.25 * maxSpan) continue;
        if (yTop < 0) yTop = ys[i];
        yBot = ys[i];
    }
    if (yTop < 0 || yBot - yTop < 0.4 * seed.h()) return false;
    Quad cand;
    cand.tl = {la + lb * yTop, yTop};
    cand.tr = {ra + rb * yTop, yTop};
    cand.bl = {la + lb * yBot, yBot};
    cand.br = {ra + rb * yBot, yBot};
    if (cand.tl.x >= cand.tr.x || cand.bl.x >= cand.br.x) return false;
    // Coverage gate: the fitted edges must track the felt mask itself. A
    // broken gradient mask produces edge fits that leave large parts of
    // the felt OUTSIDE the quad — those project logical points to wild
    // pixels (lines into the window corner and beyond) and must fall back
    // to the rectangle path. Real trapezoids (Pool Club) hug the mask.
    double outSum = 0, spanSum = 0;
    for (size_t i = 0; i < ys.size(); ++i) {
        if (spans[i] < 0.45 * medSpan) continue;
        double y = ys[i];
        double lo = la + lb * y, hi = ra + rb * y;
        outSum += std::max(0.0, lo - xmin[i]) +
                  std::max(0.0, xmax[i] - hi);
        spanSum += xmax[i] - xmin[i];
    }
    if (spanSum <= 0 || outSum > 0.08 * spanSum) return false;
    double topW = cand.tr.x - cand.tl.x, botW = cand.br.x - cand.bl.x;
    // Trapezoid gate: only accept when the felt actually widens toward the
    // bottom by more than 6% — flat tables keep the rectangle path.
    double wide = std::max(topW, botW), narrow = std::min(topW, botW);
    if (wide - narrow < 0.06 * seed.w()) return false;
    q = cand;
    return true;
}

DetectResult detectBalls(const Image& img, const DetectOptions& opt,
                         const TableSpec& logical) {
    if (img.empty()) throw std::runtime_error("empty image");
    const int w = img.w, h = img.h;
    const size_t n = static_cast<size_t>(w) * h;

    // 1. Green felt mask (classic tables + the synthetic demo).
    std::vector<uint8_t> surface(n, 0);
    for (size_t i = 0; i < n; ++i) {
        Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1], img.px[i * 3 + 2]);
        if (c.h >= opt.feltHueMin && c.h <= opt.feltHueMax &&
            c.s >= opt.feltSatMin && c.v >= opt.feltValMin)
            surface[i] = 1;
    }

    // 2. Largest green component -> candidate table rect. Game screenshots
    // with a non-green felt only match this window on small UI accents, so
    // the rect must actually look like a table; otherwise fall back to the
    // dominant surface color (dark/gray felts).
    long bestArea = 0;
    Rect best;
    bool usedFallback = false;
    Hsv feltHue{200.0, 0.7, 0.7};  // dominant felt color (fallback path)
    bool manual = opt.forceRect.w() > 10 && opt.forceRect.h() > 10;
    if (manual) {
        // Manual region ("edit anything" workflow): still build a surface
        // mask for the ballish pass via the dominant-color matcher, which
        // works on any felt color; the auto rect is ignored.
        std::fill(surface.begin(), surface.end(), 0);
        dominantSurfaceRect(img, surface, opt, &feltHue);
        best = opt.forceRect;
        usedFallback = true;
    } else {
        // In-game white aim lines and rail trays can cut the felt into
        // components; a lightly dilated copy bridges the thin gaps so the
        // largest component is the whole felt again. The original mask is
        // kept for the ball pass (dilation must not eat ball edges).
        std::vector<uint8_t> surfD = surface;
        for (int it = 0; it < 2; ++it) {
            std::vector<uint8_t> nxt = surfD;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    size_t i = static_cast<size_t>(y) * w + x;
                    if (surfD[i]) continue;
                    bool nb = false;
                    for (int dy = -1; dy <= 1 && !nb; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            int nx = x + dx, ny = y + dy;
                            if (nx < 0 || ny < 0 || nx >= w || ny >= h)
                                continue;
                            if (surfD[static_cast<size_t>(ny) * w + nx]) {
                                nb = true;
                                break;
                            }
                        }
                    if (nb) nxt[i] = 1;
                }
            surfD = std::move(nxt);
        }
        best = largestComponent(surfD, w, h, bestArea);
        if (bestArea < 100 || !plausibleTableRect(best, w, h)) {
            std::fill(surface.begin(), surface.end(), 0);
            best = dominantSurfaceRect(img, surface, opt, &feltHue);
            usedFallback = true;
            if (!plausibleTableRect(best, w, h))
                throw std::runtime_error(
                    "no table region found - crop the table area from the "
                    "screenshot and try again");
        }
    }

    // 2b. Adaptive re-threshold for gradient felts: 8BP renders the felt
    // with a strong brightness/saturation gradient (lit middle s~0.43,
    // dark rails edge s~0.88). One fixed tolerance keeps only part of the
    // felt, the fit collapses and balls at the far end drop out. Rebuild
    // the mask around the fitted core's medians with per-channel windows
    // wide enough to hold the whole gradient, then re-fit.
    {
        // Manual path (forceRect): the medians MUST come from the center of
        // the forced rect over ALL pixels — the middle of a table is always
        // felt, while the dominant-color mask may have locked onto the page
        // background (navy UI) and poisoned every window.
        Hsv med{200.0, 0.6, 0.6};
        bool medOk = false;
        if (manual) {
            int cx0 = static_cast<int>(best.x0 + 0.25 * best.w());
            int cx1 = static_cast<int>(best.x1 - 0.25 * best.w());
            int cy0 = static_cast<int>(best.y0 + 0.25 * best.h());
            int cy1 = static_cast<int>(best.y1 - 0.25 * best.h());
            std::vector<double> hs, ss2, vv;
            for (int y = cy0; y < cy1; y += 2)
                for (int x = cx0; x < cx1; x += 2) {
                    Hsv c = rgbToHsv(
                        img.px[(static_cast<size_t>(y) * w + x) * 3],
                        img.px[(static_cast<size_t>(y) * w + x) * 3 + 1],
                        img.px[(static_cast<size_t>(y) * w + x) * 3 + 2]);
                    hs.push_back(c.h); ss2.push_back(c.s); vv.push_back(c.v);
                }
            if (hs.size() > 200) {
                auto med3 = [](std::vector<double>& v) {
                    std::sort(v.begin(), v.end());
                    return v[v.size() / 2];
                };
                med = Hsv{med3(hs), med3(ss2), med3(vv)};
                medOk = true;
            }
        } else {
            long cnt = 0;
            double sh = 0, ss = 0, sv = 0;
            for (int y = static_cast<int>(best.y0);
                 y < static_cast<int>(best.y1); y += 2)
                for (int x = static_cast<int>(best.x0);
                     x < static_cast<int>(best.x1); x += 2) {
                    if (!surface[static_cast<size_t>(y) * w + x]) continue;
                    Hsv c = rgbToHsv(
                        img.px[(static_cast<size_t>(y) * w + x) * 3],
                        img.px[(static_cast<size_t>(y) * w + x) * 3 + 1],
                        img.px[(static_cast<size_t>(y) * w + x) * 3 + 2]);
                    sh += c.h; ss += c.s; sv += c.v; ++cnt;
                }
            if (cnt > 200) {
                med = Hsv{sh / cnt, ss / cnt, sv / cnt};
                medOk = true;
            }
        }

        if (medOk) {
            // Windows: wide enough to hold a real gradient felt (8BP spans
            // s 0.43..0.88 / v 0.46..0.87 measured, including the darker
            // cushion-shadow band — a narrow window carves that band out of
            // the surface and its fragments surface as rows of fake blue
            // balls). The v floor (0.30) still keeps the dark page
            // background out, and the black ball stays ballish (its body
            // reads v < 0.35).
            std::vector<uint8_t> surf2(n, 0);
            for (size_t i = 0; i < n; ++i) {
                Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1],
                                 img.px[i * 3 + 2]);
                double hd = std::fabs(c.h - med.h);
                hd = std::min(hd, 360.0 - hd);
                if (hd <= 25.0 && c.s >= std::max(0.10, med.s - 0.45) &&
                    c.s <= std::min(0.95, med.s + 0.45) &&
                    c.v >= 0.30 &&
                    c.v <= std::min(0.95, med.v + 0.45))
                    surf2[i] = 1;
            }
            for (int it = 0; it < 2; ++it) {
                std::vector<uint8_t> nxt = surf2;
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x) {
                        size_t i = static_cast<size_t>(y) * w + x;
                        if (nxt[i]) continue;
                        bool nb = false;
                        for (int dy = -1; dy <= 1 && !nb; ++dy)
                            for (int dx = -1; dx <= 1; ++dx) {
                                int nx = x + dx, ny = y + dy;
                                if (nx >= 0 && ny >= 0 && nx < w && ny < h &&
                                    surf2[static_cast<size_t>(ny) * w + nx]) {
                                    nb = true;
                                    break;
                                }
                            }
                        if (nb) nxt[i] = 1;
                    }
                surf2 = std::move(nxt);
            }
            long area2 = 0;
            Rect best2 = largestComponent(surf2, w, h, area2);
            Rect oldBest = best;
            bool overlaps = best2.x0 < oldBest.x1 && oldBest.x0 < best2.x1 &&
                            best2.y0 < oldBest.y1 && oldBest.y0 < best2.y1;
            if (manual) {
                // Forced boundary: the rect cannot leak — it is fixed by
                // the caller — so adopt any meaningful mask improvement to
                // keep ballish correct on gradient felts.
                if (area2 > bestArea * 1.10 && overlaps) {
                    surface.swap(surf2);
                    best = opt.forceRect;
                    bestArea = area2;
                    feltHue = med;
                }
            } else {
                // Growth gate: the re-threshold must REFINE the same felt,
                // not leak into the surrounding page. Per-side fraction: a
                // real gradient felt can recover a cut edge by ~10-14% of
                // that side, while page leaks grow by 50-85%. Cap at 15%.

                double growFrac = 0;
                {
                    double gx = std::max(oldBest.x0 - best2.x0,
                                         best2.x1 - oldBest.x1) /
                                std::max(1.0, oldBest.w());
                    double gy = std::max(oldBest.y0 - best2.y0,
                                         best2.y1 - oldBest.y1) /
                                std::max(1.0, oldBest.h());
                    growFrac = std::max(gx, gy);
                }
                bool rectGrew = best2.x1 - oldBest.x1 > 2 ||
                                best2.y1 - oldBest.y1 > 2 ||
                                oldBest.x0 - best2.x0 > 2 ||
                                oldBest.y0 - best2.y0 > 2;
                if (area2 > bestArea * 1.08 && overlaps && rectGrew &&
                    growFrac <= 0.15 && plausibleTableRect(best2, w, h)) {
                    surface.swap(surf2);
                    best = best2;
                    bestArea = area2;
                    feltHue = med;
                }
            }
        }
    }

    // Shrink a little to avoid the cushion/wood transition pixels.
    Rect field = best;
    field.x0 += 2; field.y0 += 2;
    field.x1 -= 2; field.y1 -= 2;

    // Median felt brightness (sampled) — feeds the adaptive white gates.
    double feltMedV = 0;
    {
        std::vector<double> vv;
        for (int y = static_cast<int>(field.y0); y < static_cast<int>(field.y1);
             y += 3)
            for (int x = static_cast<int>(field.x0);
                 x < static_cast<int>(field.x1); x += 3) {
                if (!surface[static_cast<size_t>(y) * w + x]) continue;
                vv.push_back(rgbToHsv(
                    img.px[(static_cast<size_t>(y) * w + x) * 3],
                    img.px[(static_cast<size_t>(y) * w + x) * 3 + 1],
                    img.px[(static_cast<size_t>(y) * w + x) * 3 + 2]).v);
            }
        if (!vv.empty()) {
            std::sort(vv.begin(), vv.end());
            feltMedV = vv[vv.size() / 2];
        } else {
            feltMedV = 0.6;
        }
    }

    // Perspective fit (green auto path only): Pool Club style felts are
    // trapezoids; the quad + bilinear mapping keeps every line on measure.
    Quad quad{};
    bool hasQuad = false;
    if (!manual && !usedFallback)
        hasQuad = fitTrapezoid(surface, w, h, best, quad);

    // 3. Ball-ish mask inside the field: anything not matching the surface.
    // With a quad, balls near the wide end of the trapezoid sit outside the
    // inscribed rectangle, so scan the quad rows instead.
    std::vector<uint8_t> ballish(n, 0);
    {
        int sy0 = hasQuad ? static_cast<int>(quad.tl.y)
                          : static_cast<int>(field.y0);
        int sy1 = hasQuad ? static_cast<int>(quad.bl.y)
                          : static_cast<int>(field.y1);
        for (int y = sy0; y < sy1; ++y) {
            int lo = static_cast<int>(field.x0);
            int hi = static_cast<int>(field.x1);
            if (hasQuad) {
                double t = (y - quad.tl.y) /
                           std::max(1.0, quad.bl.y - quad.tl.y);
                lo = static_cast<int>(quad.tl.x + (quad.bl.x - quad.tl.x) * t);
                hi = static_cast<int>(quad.tr.x + (quad.br.x - quad.tr.x) * t);
            }
            for (int x = lo; x < hi; ++x) {
                if (x < 0 || x >= w) continue;
                size_t i = static_cast<size_t>(y) * w + x;
                if (!surface[i]) ballish[i] = 1;
            }
        }
    }

    // 3b. Strip the in-game aim guideline: it physically merges with the
    // white cue ball and shreds the felt into pseudo-blobs. The guideline is
    // THIN, ELONGATED and LOW-DENSITY; balls are thick, round and dense even
    // with the strike dot / cue stick punching holes into them — so a white
    // pixel is removed only when its local white neighborhood is sparse AND
    // strongly anisotropic. Holey balls survive; lines die.
    if (opt.stripLines) {
        std::vector<uint8_t> white(n, 0);
        long whiteCnt = 0;
        for (int y = static_cast<int>(field.y0); y < static_cast<int>(field.y1); ++y) {
            for (int x = static_cast<int>(field.x0); x < static_cast<int>(field.x1); ++x) {
                size_t i = static_cast<size_t>(y) * w + x;
                if (!ballish[i]) continue;
                Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1], img.px[i * 3 + 2]);
                if (c.s < 0.25 && c.v > 0.70) { white[i] = 1; ++whiteCnt; }
            }
        }
        const int R = 6;
        const double kDisc = static_cast<double>(R * R) * 3.14159265358979323846;
        for (int y = static_cast<int>(field.y0); y < static_cast<int>(field.y1); ++y) {
            for (int x = static_cast<int>(field.x0); x < static_cast<int>(field.x1); ++x) {
                size_t i = static_cast<size_t>(y) * w + x;
                if (!white[i]) continue;
                double cnt = 0, mx = 0, my = 0, sxx = 0, sxy = 0, syy = 0;
                for (int dy = -R; dy <= R; ++dy) {
                    int ny = y + dy;
                    if (ny < 0 || ny >= h) continue;
                    for (int dx = -R; dx <= R; ++dx) {
                        if (dx * dx + dy * dy > R * R) continue;
                        int nx = x + dx;
                        if (nx < 0 || nx >= w) continue;
                        if (!white[static_cast<size_t>(ny) * w + nx]) continue;
                        ++cnt;
                        mx += nx;
                        my += ny;
                        sxx += static_cast<double>(nx) * nx;
                        sxy += static_cast<double>(nx) * ny;
                        syy += static_cast<double>(ny) * ny;
                    }
                }
                bool strip = cnt < 8;  // isolated speck
                if (!strip) {
                    mx /= cnt;
                    my /= cnt;
                    double vxx = sxx / cnt - mx * mx;
                    double vxy = sxy / cnt - mx * my;
                    double vyy = syy / cnt - my * my;
                    double tr = vxx + vyy;
                    double det = vxx * vyy - vxy * vxy;
                    double disc = std::sqrt(std::max(0.0, tr * tr * 0.25 - det));
                    double l1 = tr * 0.5 + disc;
                    double l2 = std::max(tr * 0.5 - disc, 0.5);
                    double density = cnt / kDisc;
                    double aniso = l1 / l2;
                    if (density < 0.5 && aniso > 4.0) strip = true;
                }
                if (strip) ballish[i] = 0;
            }
        }
    }

    // 4. Blobs.
    std::vector<Blob> blobs = findBlobs(ballish, w, h);

    // 5. Ball radius estimate.
    //    Green path: median blob area (uniform felts -> blobs are the balls).
    //    Fallback path (game screenshots): gradient felts create many noise
    //    blobs that skew the median, so derive the expected radius from the
    //    detected rect size instead (logical ballR scales with rect width).
    double minDim = std::min(field.w(), field.h());
    double pocketExclPre = 0.07 * minDim;
    // Pocket anchors: quad corners + edge midpoints (TL,TC,TR,BL,BC,BR) or
    // the rectangle corners when no perspective fit was accepted.
    Vec2 anchors[6];
    if (hasQuad) {
        anchors[0] = quad.tl;
        anchors[1] = {(quad.tl.x + quad.tr.x) / 2, quad.tl.y};
        anchors[2] = quad.tr;
        anchors[3] = quad.bl;
        anchors[4] = {(quad.bl.x + quad.br.x) / 2, quad.bl.y};
        anchors[5] = quad.br;
    } else {
        anchors[0] = {field.x0, field.y0};
        anchors[1] = {(field.x0 + field.x1) / 2, field.y0};
        anchors[2] = {field.x1, field.y0};
        anchors[3] = {field.x0, field.y1};
        anchors[4] = {(field.x0 + field.x1) / 2, field.y1};
        anchors[5] = {field.x1, field.y1};
    }
    auto nearAnyAnchor = [&](const Blob& b, double rad) {
        for (const auto& a : anchors)
            if (std::hypot(b.cx - a.x, b.cy - a.y) < rad) return true;
        return false;
    };
    double ballR;
    if (opt.forceBallR > 1.0) {
        ballR = opt.forceBallR;  // manual calibration wins over both paths
    } else if (usedFallback) {
        ballR = field.w() / logical.w * logical.ballR;
        if (ballR < 2.0) ballR = 2.0;
    } else {
        std::vector<long> areas;
        for (const auto& b : blobs) {
            if (b.area < opt.areaMin) continue;
            if (nearAnyAnchor(b, pocketExclPre)) continue;
            areas.push_back(b.area);
        }
        if (areas.empty())
            throw std::runtime_error("no ball-like blobs found inside the table");
        std::sort(areas.begin(), areas.end());
        // 25th percentile: single balls dominate the small end, while merged
        // clusters (Pool Club style) inflate the median and made every
        // circle oversize. Flat-table games are all singles, so p25 == median.
        double p25A = areas[areas.size() / 4];
        ballR = std::sqrt(p25A / 3.14159265358979323846);
        // Clamp against the table-size expectation (rect/quad width scales
        // the logical ballR) so noise specks and merged blobs cannot drift.
        double scaleW = hasQuad
                            ? ((quad.tr.x - quad.tl.x) +
                               (quad.br.x - quad.bl.x)) / 2
                            : field.w();
        double expectedR = scaleW / logical.w * logical.ballR;
        if (expectedR > 2.0)
            ballR = clampd(ballR, 0.55 * expectedR, 1.45 * expectedR);
    }

    // Final exclusion radius around the pocket anchors. The kept-blob window
    // below is expressed against the ball disc area (green path: the disc of
    // the median blob, i.e. exactly the previous behavior).
    double pocketExcl = std::max(pocketExclPre, 3.04 * ballR);
    double refArea = 3.14159265358979323846 * ballR * ballR;

    // 6+7. Keep plausible blobs, skip pockets, classify.
    DetectResult res;
    res.feltRect = best;
    res.ballRPx = ballR;
    res.table = logical;
    res.hasQuad = hasQuad;
    res.quadPx = quad;

    // Snap pocket anchors to the real pocket holes. Gradient felts make the
    // detected rect cover the core felt only, so rect corners can sit a few
    // pixels off the actual pockets (games draw dark round holes there).
    // anchors[k] and table.pockets[k] share the same TL,TC,TR,BL,BC,BR order.
    {
        const double snapR = std::max(ballR, 0.055 * minDim);
        for (int k = 0; k < 6 && k < static_cast<int>(res.table.pockets.size()); ++k) {
            Vec2 a = anchors[k];
            int sx0 = static_cast<int>(clampd(a.x - snapR, 0, w - 1.0));
            int sx1 = static_cast<int>(clampd(a.x + snapR, 0, w - 1.0));
            int sy0 = static_cast<int>(clampd(a.y - snapR, 0, h - 1.0));
            int sy1 = static_cast<int>(clampd(a.y + snapR, 0, h - 1.0));
            long cnt = 0;
            double sx = 0, sy = 0;
            for (int y = sy0; y <= sy1; ++y) {
                for (int x = sx0; x <= sx1; ++x) {
                    size_t i = static_cast<size_t>(y) * w + x;
                    Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1],
                                     img.px[i * 3 + 2]);
                    if (c.v < 0.13) { ++cnt; sx += x; sy += y; }
                }
            }
            if (cnt < 30) continue;
            Vec2 s{sx / cnt, sy / cnt};
            if (s.dist(a) > snapR) continue;
            // Partial pull (60%) so a stray dark patch cannot drag a pocket.
            Vec2 moved{a.x * 0.4 + s.x * 0.6, a.y * 0.4 + s.y * 0.6};
            anchors[k] = moved;
            res.table.pockets[k].pos =
                hasQuad ? quadPxToLogical(quad, moved, logical.w, logical.h)
                        : Vec2{(moved.x - field.x0) / field.w() * logical.w,
                               (moved.y - field.y0) / field.h() * logical.h};
        }
    }

    struct Raw { Blob b; };
    std::vector<Raw> kept;
    int salvaged = 0;

    // Inscribed-disc probes shared by the cluster splitter and the merged-
    // blob salvage path: find a ball-sized disc inside a (possibly merged)
    // blob, gate it against felt fragments and gray UI junk, keep only
    // discs distinct from the parts already found, and erase the accepted
    // disc from `mask` so the next probe moves on. The geometric gate
    // (>= 1.35R from every accepted part) blocks shadow/line remnants that
    // hug a real ball; touching balls sit ~2R apart and pass. Returns the
    // number of accepted discs appended to `parts`.
    auto probeDiscs = [&](std::vector<uint8_t>& mask, const Blob& src,
                          int maxProbes, std::vector<Blob>& parts) -> int {
        Rect grown{src.bbox.x0 - ballR - 2, src.bbox.y0 - ballR - 2,
                   src.bbox.x1 + ballR + 2, src.bbox.y1 + ballR + 2};
        int accepted = 0;
        for (int probe = 0; probe < maxProbes; ++probe) {
            Vec2 sc{};
            double sr = inscribedDisc(mask, w, h, grown, ballR, sc);
            if (sr < 0.72 * ballR) break;
            bool good = true;
            {
                long felt = 0, tot = 0;
                double sumS = 0, sumV = 0;
                int ir = static_cast<int>(std::max(1.0, sr * 0.8));
                for (int yy = static_cast<int>(sc.y) - ir;
                     yy <= static_cast<int>(sc.y) + ir; ++yy) {
                    for (int xx = static_cast<int>(sc.x) - ir;
                         xx <= static_cast<int>(sc.x) + ir; ++xx) {
                        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                        double dx = xx - sc.x, dy = yy - sc.y;
                        if (dx * dx + dy * dy > ir * ir) continue;
                        ++tot;
                        Hsv cc = rgbToHsv(img.px[(static_cast<size_t>(yy) * w + xx) * 3],
                                          img.px[(static_cast<size_t>(yy) * w + xx) * 3 + 1],
                                          img.px[(static_cast<size_t>(yy) * w + xx) * 3 + 2]);
                        sumS += cc.s;
                        sumV += cc.v;
                        double hd = std::fabs(cc.h - feltHue.h);
                        hd = std::min(hd, 360.0 - hd);
                        if (std::fabs(cc.s - feltHue.s) < 0.32 &&
                            std::fabs(cc.v - feltHue.v) < 0.32 &&
                            (hd < 20.0 || cc.s < 0.12)) ++felt;
                    }
                }
                if (tot > 0 && felt > tot * 0.6) good = false;  // felt fragment
                // Gray/dim discs are UI junk, not balls (balls are
                // white, colored or black).
                if (tot > 0 && sumS / tot < 0.12 && sumV / tot < 0.72)
                    good = false;
            }
            if (!good) {
                // Erase the rejected disc so the next probe can move on.
                int er = static_cast<int>(sr) + 1;
                for (int yy = static_cast<int>(sc.y) - er;
                     yy <= static_cast<int>(sc.y) + er; ++yy)
                    for (int xx = static_cast<int>(sc.x) - er;
                         xx <= static_cast<int>(sc.x) + er; ++xx) {
                        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                        double dx = xx - sc.x, dy = yy - sc.y;
                        if (dx * dx + dy * dy <= er * er)
                            mask[static_cast<size_t>(yy) * w + xx] = 0;
                    }
                continue;
            }
            // Geometric gate: touching balls sit ~2R apart, so anything
            // closer than 1.35R to an accepted part is its shadow, an aim-
            // line remnant or a sub-radius fragment, never another ball.
            for (const auto& pv : parts)
                if (std::hypot(pv.cx - sc.x, pv.cy - sc.y) < 1.35 * ballR) {
                    good = false;
                    break;
                }
            if (!good) {
                int er = static_cast<int>(sr) + 1;
                for (int yy = static_cast<int>(sc.y) - er;
                     yy <= static_cast<int>(sc.y) + er; ++yy)
                    for (int xx = static_cast<int>(sc.x) - er;
                         xx <= static_cast<int>(sc.x) + er; ++xx) {
                        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                        double dx = xx - sc.x, dy = yy - sc.y;
                        if (dx * dx + dy * dy <= er * er)
                            mask[static_cast<size_t>(yy) * w + xx] = 0;
                    }
                continue;
            }
            // Accept a further probe only if it is a DIFFERENT ball:
            // far enough from the earlier parts of this blob (touching
            // balls sit ~2 radii apart) or clearly colored. Elongated white
            // structures (the dilated cue-fallback mask) otherwise
            // yield chains of overlapping "balls" 10 px apart.
            double meanSat = 0;
            long satN = 0;
            {
                int sir = static_cast<int>(std::max(1.0, sr));
                for (int yy = static_cast<int>(sc.y) - sir;
                     yy <= static_cast<int>(sc.y) + sir; ++yy)
                    for (int xx = static_cast<int>(sc.x) - sir;
                         xx <= static_cast<int>(sc.x) + sir; ++xx) {
                        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                        double ddx = xx - sc.x, ddy = yy - sc.y;
                        if (ddx * ddx + ddy * ddy > sir * sir) continue;
                        meanSat += rgbToHsv(
                            img.px[(static_cast<size_t>(yy) * w + xx) * 3],
                            img.px[(static_cast<size_t>(yy) * w + xx) * 3 + 1],
                            img.px[(static_cast<size_t>(yy) * w + xx) * 3 + 2]).s;
                        ++satN;
                    }
            }
            meanSat = satN > 0 ? meanSat / satN : 0.0;
            bool distinct = true;
            for (const auto& pv : parts)
                if (std::hypot(pv.cx - sc.x, pv.cy - sc.y) <
                        2.0 * ballR &&
                    meanSat < 0.15) {
                    distinct = false;
                    break;
                }
            if (!distinct) {
                int er = static_cast<int>(sr) + 1;
                for (int yy = static_cast<int>(sc.y) - er;
                     yy <= static_cast<int>(sc.y) + er; ++yy)
                    for (int xx = static_cast<int>(sc.x) - er;
                         xx <= static_cast<int>(sc.x) + er; ++xx) {
                        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                        double ddx = xx - sc.x, ddy = yy - sc.y;
                        if (ddx * ddx + ddy * ddy <= er * er)
                            mask[static_cast<size_t>(yy) * w + xx] = 0;
                    }
                continue;
            }
            ++salvaged;
            ++accepted;
            Blob cc = src;
            cc.cx = sc.x;
            cc.cy = sc.y;
            cc.area = static_cast<long>(3.14159265358979323846 * sr * sr);
            cc.bbox = Rect{sc.x - sr, sc.y - sr, sc.x + sr, sc.y + sr};
            parts.push_back(cc);
            // Erase a FULL ball radius: a smaller eraser re-finds a
            // second disc inside the same ball (two detections 13px
            // apart), which then poisons the cue isolation vote.
            int er = static_cast<int>(sr) + 1;
            for (int yy = static_cast<int>(sc.y) - er;
                 yy <= static_cast<int>(sc.y) + er; ++yy)
                for (int xx = static_cast<int>(sc.x) - er;
                     xx <= static_cast<int>(sc.x) + er; ++xx) {
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                    double dx = xx - sc.x, dy = yy - sc.y;
                    if (dx * dx + dy * dy <= er * er)
                        mask[static_cast<size_t>(yy) * w + xx] = 0;
                }
        }
        return accepted;
    };

    for (const auto& b : blobs) {
        Blob c = b;
        bool ok = c.area >= opt.areaMin;
        if (ok)
            ok = c.area >= refArea * opt.areaLoFactor &&
                 c.area <= refArea * opt.areaHiFactor;
        if (ok) {
            // Pocket-shadow crescents: dark saturated-blue blobs that hug
            // the pocket holes (8BP renders them at v 0.22-0.27, r 8-10).
            // Real balls never read that dark at blob scale — even the
            // black 8 ball keeps a brighter disk mean.
            double rBlob = std::sqrt(
                static_cast<double>(c.area) / 3.14159265358979323846);
            long vTot = 0;
            double vSum = 0;
            int vx0 = std::max(0, static_cast<int>(c.bbox.x0)),
                vx1 = std::min(w - 1, static_cast<int>(c.bbox.x1));
            int vy0 = std::max(0, static_cast<int>(c.bbox.y0)),
                vy1 = std::min(h - 1, static_cast<int>(c.bbox.y1));
            for (int yy = vy0; yy <= vy1; yy += 2)
                for (int xx = vx0; xx <= vx1; xx += 2) {
                    size_t vi = static_cast<size_t>(yy) * w + xx;
                    if (!ballish[vi]) continue;
                    vSum += rgbToHsv(
                        img.px[vi * 3], img.px[vi * 3 + 1],
                        img.px[vi * 3 + 2]).v;
                    ++vTot;
                }
            double meanV = vTot > 0 ? vSum / vTot : 1.0;
            if (meanV < 0.30 && rBlob < 0.75 * ballR) ok = false;
            if (c.area < refArea * 0.45 && rBlob < 0.75 * ballR) ok = false;
        }
        if (ok) {
            // Text, name plates and elongated UI are not balls (all paths;
            // photo shadows attached under balls stay within these bounds).
            // Game screenshots additionally drop hollow rings/icons by fill.
            double bw = std::max(1.0, c.bbox.w()), bh = std::max(1.0, c.bbox.h());
            double aspect = bw / bh;
            if (aspect < 0.45 || aspect > 2.25) ok = false;
            else if (usedFallback) {
                double extent = static_cast<double>(c.area) / (bw * bh);
                if (extent < 0.35) ok = false;
            }
        }
        std::vector<Blob> parts;
        if (ok) {
            // Touching-ball clusters (post-break racks, balls glued
            // together) can pass the area window as ONE blob: two glued
            // balls cover 2x a ball disc, inside areaHiFactor (2.4), and
            // then classify from blended colors with a center between two
            // real balls. Any plausible blob meaningfully larger than one
            // ball disc is probed for several full-size inscribed discs;
            // two or more accepted discs split it into real balls, a
            // single disc keeps the original blob unchanged. The probe
            // runs on a COPY of the mask: its erasures must not leak into
            // the cue-fallback pass below when the split does not stick.
            double rBlobOk = std::sqrt(
                static_cast<double>(c.area) / 3.14159265358979323846);
            if (rBlobOk > 1.25 * ballR) {
                std::vector<uint8_t> splitMask = ballish;
                int maxProbes = std::max(
                    3, std::min(10,
                                static_cast<int>(c.area / refArea) + 2));
                probeDiscs(splitMask, c, maxProbes, parts);
                if (parts.size() < 2) parts.clear();
                if (parts.empty()) parts.push_back(c);
            } else {
                parts.push_back(c);
            }
        } else {
            // Merged structures — cue + stick, cue + aim line + ghost disc
            // + an object ball, touching clusters — still contain ball-
            // sized inscribed discs. Probe count scales with the blob's
            // ball multiples so a multi-ball rack splits fully; the
            // stick/line remainder is dropped. 8BP draws the cue stick
            // glued to the ball, so the merged blob routinely exceeds
            // 5x a ball area; 16x still rejects page-scale junk, and a
            // blob that size must yield at least TWO balls or it is a UI
            // plate, not a cluster.
            if (c.area > refArea * 16.0) continue;
            if (salvaged >= 32) continue;  // junk images explode otherwise
            // A blob this large must look like a union of round balls to
            // be trusted as a cluster: any union of discs fills at most
            // ~0.79 of its bbox, while a filled rectangle / UI name plate
            // reaches ~1.0 and can host several inscribed discs inside its
            // ring band even after the center is erased.
            if (c.area > refArea * 12.0) {
                double bw2 = std::max(1.0, c.bbox.w());
                double bh2 = std::max(1.0, c.bbox.h());
                if (c.area / (bw2 * bh2) > 0.85) continue;
            }
            int maxProbes = std::max(
                3, std::min(12, static_cast<int>(c.area / refArea) + 2));
            probeDiscs(ballish, c, maxProbes, parts);
            if (c.area > refArea * 12.0 && parts.size() < 2) continue;
        }
        for (const auto& pb : parts) {
            const Blob& c = pb;
            bool nearPocket = false;
            for (const auto& a : anchors)
                if (std::hypot(c.cx - a.x, c.cy - a.y) < pocketExcl) { nearPocket = true; break; }
            if (nearPocket) continue;
            bool dup = false;

            for (const auto& k : kept) {
                // 1.05 * ballR: same-ball duplicates from aim-line splits sit
                // closer than this; real touching balls sit ~2 radii apart.
                if (std::hypot(k.b.cx - c.cx, k.b.cy - c.cy) < 1.05 * ballR) {
                    dup = true;
                    break;
                }
            }
            if (dup) continue;
            kept.push_back({c});
        }
    }

    for (const auto& raw : kept) {
        const auto& b = raw.b;
        double r = std::sqrt(b.area / 3.14159265358979323846);
        double ir = std::max(1.0, r * 0.85);  // sampling radius inside the disc
        long whiteCnt = 0, blackCnt = 0, colCnt = 0, rimTot = 0, rimCol = 0;
        long litCnt = 0;
        double litX = 0, litY = 0;
        double sumS = 0, sumV = 0, sumColV = 0;
        int hist[24] = {0};
        long total = 0;
        // Adaptive white floor: on the dark half of a gradient felt the
        // white ball renders at v~0.54, far below the fixed 0.72 gate, and
        // the cue disappears. White stays = low saturation + brighter than
        // the felt, clamped under the calibrated default.
        double whiteVMin = std::min(opt.cueValMin,
                                    std::max(0.55, feltMedV + 0.10));
        // Center disc for the cue decision: when the blob radius runs large
        // (merged clusters), the full disc bleeds into the felt and a pure
        // white cue ball drops below the 0.80 gate. The ball CENTER is
        // always white for a real cue ball, colored for a stripe ball.
        long cwhite = 0, ctot = 0;
        int x0 = static_cast<int>(b.cx - ir), x1 = static_cast<int>(b.cx + ir);
        int y0 = static_cast<int>(b.cy - ir), y1 = static_cast<int>(b.cy + ir);
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                if (x < 0 || y < 0 || x >= w || y >= h) continue;
                double dx = x - b.cx, dy = y - b.cy;
                double d2 = dx * dx + dy * dy;
                if (d2 > ir * ir) continue;
                size_t i = static_cast<size_t>(y) * w + x;
                Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1], img.px[i * 3 + 2]);
                ++total;
                sumS += c.s;
                sumV += c.v;
                bool white = c.s < opt.cueSatMax && c.v > whiteVMin;
                bool black = c.v < opt.blackValMax;
                if (white) ++whiteCnt;
                if (black) ++blackCnt;
                // Rim ring 0.68R..0.95R: white for a real cue ball, colored
                // for a stripe ball flipped white-cap-up (the stripe wraps
                // the silhouette edge around the cap).
                double rimLo = ir * ir * 0.46, rimHi = ir * ir * 0.90;
                if (d2 >= rimLo && d2 <= rimHi) {
                    ++rimTot;
                    if (c.s > 0.30 && c.v > 0.20) ++rimCol;
                }
                if (c.s > 0.30 && c.v > 0.20) {
                    ++colCnt;
                    sumColV += c.v;
                    ++hist[static_cast<int>(c.h / 15.0) % 24];
                }
                // Lit colored pixels carry the true ball color; the
                // shadowed half reads 10-25 deg toward red/brown and once
                // hijacked the hue histogram (yellow read as orange).
                if (c.s > 0.35 && c.v > 0.50) {
                    ++litCnt;
                    // circular mean: accumulate the unit vector
                    double rad = c.h * 3.14159265358979323846 / 180.0;
                    litX += std::cos(rad);
                    litY += std::sin(rad);
                }
            }
        }
        if (total == 0) continue;
        int cir = static_cast<int>(std::max(1.0, r * 0.60));
        for (int y = static_cast<int>(b.cy) - cir;
             y <= static_cast<int>(b.cy) + cir; ++y) {
            for (int x = static_cast<int>(b.cx) - cir;
                 x <= static_cast<int>(b.cx) + cir; ++x) {
                if (x < 0 || y < 0 || x >= w || y >= h) continue;
                double dx = x - b.cx, dy = y - b.cy;
                if (dx * dx + dy * dy > cir * cir) continue;
                size_t i = static_cast<size_t>(y) * w + x;
                Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1],
                                 img.px[i * 3 + 2]);
                ++ctot;
                if (c.s < opt.cueSatMax && c.v > whiteVMin) ++cwhite;
            }
        }
        Detection d;
        d.px = {b.cx, b.cy};
        d.rPx = r;
        d.area = static_cast<double>(b.area);
        d.whiteRatio = static_cast<double>(whiteCnt) / total;
        double centerWhite = ctot > 0 ? static_cast<double>(cwhite) / ctot : 0.0;
        d.rimColored = rimTot > 0 ? static_cast<double>(rimCol) / rimTot : 0.0;
        d.sat = sumS / total;
        d.val = sumV / total;
        int hueBin = static_cast<int>(std::max_element(hist, hist + 24) - hist);
        d.hue = (hueBin + 0.5) * 15.0;
        // Prefer the lit-pixel circular mean when enough lit pixels exist.
        if (litCnt >= total * 0.10 && litCnt > 4) {
            double lh = std::atan2(litY, litX) * 180.0 /
                        3.14159265358979323846;
            if (lh < 0) lh += 360.0;
            d.hue = lh;
        }
        // A disc with (almost) no colored pixels has no meaningful hue —
        // an empty histogram reads bin 0 ("red") and once fabricated a
        // "red stripe 11" out of a pure-white ghost disc. Mark it invalid;
        // colorName() maps it to "unknown".
        if (colCnt < total * 0.02) d.hue = -1.0;
        double coloredVal = colCnt > 0 ? sumColV / colCnt : 0.0;
        double blackRatio = static_cast<double>(blackCnt) / total;

        if (blackRatio > 0.55) {
            d.kind = "black"; d.color = "black"; d.number = 8;
        } else if ((d.whiteRatio > 0.80 || centerWhite > 0.72) &&
                   d.rimColored < 0.25) {
            // 0.72 center gate: the ball center is white for a real cue
            // ball even when a salvaged/merged blob sits a few px off, and
            // colored for stripe/solid balls (their band crosses the
            // center). The rim gate rejects a stripe ball flipped
            // white-cap-up: its cap looks like a cue disc but the stripe
            // colors the silhouette edge, which a real cue never does.
            d.kind = "cue"; d.color = "white"; d.number = 0;
        } else if (d.whiteRatio > opt.stripeWhiteMin) {
            d.kind = "stripe";
            d.color = colorName(d.hue, coloredVal);
            d.number = numberForColor(d.color, true);
        } else {
            d.kind = "solid";
            d.color = colorName(d.hue, coloredVal);
            d.number = numberForColor(d.color, false);
        }
        if (d.color == "unknown" && d.kind != "cue" && d.kind != "black") {
            d.kind = "unknown";
            d.number = -1;
        }

        // Felt-texture fragments (shadow bands, gradient clumps) carry the
        // felt's own color — real balls never do. Blue 8BP balls sit ~25deg
        // away from the felt hue, so they survive this gate.
        double fhd = std::fabs(d.hue - feltHue.h);
        fhd = std::min(fhd, 360.0 - fhd);
        if (usedFallback && std::fabs(d.sat - feltHue.s) < 0.25 &&
            std::fabs(d.val - feltHue.v) < 0.30 && fhd < 15.0)
            continue;

        // Logical mapping: perspective-corrected through the felt quad when
        // available, affine rect stretch otherwise.
        d.logical = hasQuad
                        ? quadPxToLogical(quad, d.px, logical.w, logical.h)
                        : Vec2{(b.cx - field.x0) / field.w() * logical.w,
                               (b.cy - field.y0) / field.h() * logical.h};
        res.balls.push_back(d);
    }

    // Cue fallback: when the blob pipeline loses the cue ball (the strike
    // dot and the cue stick shred its white disc into thin fragments that
    // the line-strip removes), look for the deepest remaining white core.
    // Only a ball-sized white disc reaches depth >= 4px; the aim line and
    // the ghost-ball ring are thin (depth <= 2) and cannot compete.
    {
        bool haveCue = false;
        for (const auto& d : res.balls) haveCue |= d.kind == "cue";
        if (!haveCue) {
            std::vector<uint8_t> wm(n, 0);
            double wmVMin = std::max(0.55, feltMedV);
            for (int y = static_cast<int>(field.y0); y < static_cast<int>(field.y1); ++y) {
                for (int x = static_cast<int>(field.x0); x < static_cast<int>(field.x1); ++x) {
                    size_t i = static_cast<size_t>(y) * w + x;
                    if (!ballish[i]) continue;
                    Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1],
                                     img.px[i * 3 + 2]);
                    // Adaptive floor: a cue on the dark half of a gradient
                    // felt renders at v~0.54; the old fixed 0.70 gate lost
                    // it. The line-crossed disc is then shredded — dilate
                    // the mask so the fragments rejoin into a solid ball.
                    if (c.s < 0.30 && c.v > wmVMin) wm[i] = 1;
                }
            }
            for (int it = 0; it < 2; ++it) {
                std::vector<uint8_t> nxt = wm;
                for (int y = static_cast<int>(field.y0);
                     y < static_cast<int>(field.y1); ++y)
                    for (int x = static_cast<int>(field.x0);
                         x < static_cast<int>(field.x1); ++x) {
                        size_t i = static_cast<size_t>(y) * w + x;
                        if (wm[i]) continue;
                        bool nb = false;
                        for (int dy = -1; dy <= 1 && !nb; ++dy)
                            for (int dx = -1; dx <= 1; ++dx) {
                                int nx = x + dx, ny = y + dy;
                                if (nx < 0 || ny < 0 || nx >= w || ny >= h)
                                    continue;
                                if (wm[static_cast<size_t>(ny) * w + nx]) {
                                    nb = true;
                                    break;
                                }
                            }
                        if (nb) nxt[i] = 1;
                    }
                wm.swap(nxt);
            }
            const float INF = 1e9f;
            // Iterate deepest-first: when a candidate turns out to be the
            // white band of an already-detected ball, erase it and re-probe.
            for (int iter = 0; iter < 4; ++iter) {
            std::vector<float> depth(n, 0.0f);
            for (int y = static_cast<int>(field.y0); y < static_cast<int>(field.y1); ++y) {
                for (int x = static_cast<int>(field.x0); x < static_cast<int>(field.x1); ++x) {
                    size_t i = static_cast<size_t>(y) * w + x;
                    if (!wm[i]) { depth[i] = 0; continue; }
                    float d = INF;
                    if (x - 1 >= 0) d = std::min(d, depth[i - 1] + 3.0f);
                    if (y - 1 >= 0) d = std::min(d, depth[i - w] + 3.0f);
                    if (x - 1 >= 0 && y - 1 >= 0) d = std::min(d, depth[i - w - 1] + 4.0f);
                    if (x + 1 < w && y - 1 >= 0) d = std::min(d, depth[i - w + 1] + 4.0f);
                    depth[i] = d >= INF ? INF : d;
                }
            }
            double bestDepth = 0;
            Vec2 bestC{};
            for (int y = static_cast<int>(field.y0); y < static_cast<int>(field.y1); ++y) {
                for (int x = static_cast<int>(field.x0); x < static_cast<int>(field.x1); ++x) {
                    size_t i = static_cast<size_t>(y) * w + x;
                    if (!wm[i]) continue;
                    float d = depth[i];
                    if (x + 1 < w) d = std::min(d, depth[i + 1] + 3.0f);
                    if (y + 1 < h) d = std::min(d, depth[i + w] + 3.0f);
                    if (x + 1 < w && y + 1 < h) d = std::min(d, depth[i + w + 1] + 4.0f);
                    if (x - 1 >= 0 && y + 1 < h) d = std::min(d, depth[i + w - 1] + 4.0f);
                    depth[i] = d;
                    if (d / 3.0f > bestDepth) {
                        bestDepth = d / 3.0f;
                        bestC = {static_cast<double>(x), static_cast<double>(y)};
                    }
                }
            }
            if (bestDepth < 4.0) break;
            bool nearOther = false;
            for (const auto& d : res.balls)
                if (std::hypot(d.px.x - bestC.x, d.px.y - bestC.y) < 1.2 * ballR)
                    nearOther = true;
            if (nearOther) {
                int ix = static_cast<int>(bestC.x), iy = static_cast<int>(bestC.y);
                int ir = static_cast<int>(ballR) + 2;
                for (int yy = iy - ir; yy <= iy + ir; ++yy)
                    for (int xx = ix - ir; xx <= ix + ir; ++xx) {
                        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                        double dx = xx - bestC.x, dy = yy - bestC.y;
                        if (dx * dx + dy * dy <= ir * ir)
                            wm[static_cast<size_t>(yy) * w + xx] = 0;
                    }
                continue;  // probe the next-deepest white region
            }
            {
                double wr = 0.7 * ballR;
                long wCnt = 0, tot = 0;
                double sumV = 0;
                int ix0 = static_cast<int>(bestC.x - wr), ix1 = static_cast<int>(bestC.x + wr);
                int iy0 = static_cast<int>(bestC.y - wr), iy1 = static_cast<int>(bestC.y + wr);
                for (int yy = iy0; yy <= iy1; ++yy) {
                    for (int xx = ix0; xx <= ix1; ++xx) {
                        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                        double dx = xx - bestC.x, dy = yy - bestC.y;
                        if (dx * dx + dy * dy > wr * wr) continue;
                        size_t i = static_cast<size_t>(yy) * w + xx;
                        Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1],
                                         img.px[i * 3 + 2]);
                        ++tot;
                        sumV += c.v;
                        if (c.s < 0.30 &&
                            c.v > std::max(0.55, feltMedV))
                            ++wCnt;
                    }
                }
                // Rim guard for the fallback too: a deep white core can
                // be a stripe ball's cap; a colored rim means it is not the
                // cue. Erase the cap disc and probe the next white region.
                {
                    long rimT = 0, rimC = 0;
                    double rr0 = ballR * ballR * 0.46, rr1 = ballR * ballR * 0.90;
                    int ax0 = static_cast<int>(bestC.x - ballR),
                        ax1 = static_cast<int>(bestC.x + ballR);
                    int ay0 = static_cast<int>(bestC.y - ballR),
                        ay1 = static_cast<int>(bestC.y + ballR);
                    for (int yy = ay0; yy <= ay1; ++yy)
                        for (int xx = ax0; xx <= ax1; ++xx) {
                            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                            double ddx = xx - bestC.x, ddy = yy - bestC.y;
                            double dd2 = ddx * ddx + ddy * ddy;
                            if (dd2 < rr0 || dd2 > rr1) continue;
                            size_t ri = static_cast<size_t>(yy) * w + xx;
                            Hsv c = rgbToHsv(img.px[ri * 3], img.px[ri * 3 + 1],
                                             img.px[ri * 3 + 2]);
                            ++rimT;
                            if (c.s > 0.30 && c.v > 0.20) ++rimC;
                        }
                    if (rimT > 0 && rimC >= rimT * 0.25) {
                        int ix = static_cast<int>(bestC.x),
                            iy = static_cast<int>(bestC.y);
                        int ir2 = static_cast<int>(ballR) + 2;
                        for (int yy = iy - ir2; yy <= iy + ir2; ++yy)
                            for (int xx = ix - ir2; xx <= ix + ir2; ++xx) {
                                if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                                double ddx = xx - bestC.x, ddy = yy - bestC.y;
                                if (ddx * ddx + ddy * ddy <= ir2 * ir2)
                                    wm[static_cast<size_t>(yy) * w + xx] = 0;
                            }
                        continue;  // probe the next-deepest white region
                    }
                }
                if (tot > 0 && wCnt >= tot * 0.40) {
                    Detection d;
                    d.px = bestC;
                    d.rPx = ballR;
                    d.area = 3.14159265358979323846 * ballR * ballR;
                    d.whiteRatio = static_cast<double>(wCnt) / tot;
                    d.val = sumV / tot;
                    d.kind = "cue";
                    d.color = "white";
                    d.number = 0;
                    d.logical.x = (bestC.x - field.x0) / field.w() * logical.w;
                    d.logical.y = (bestC.y - field.y0) / field.h() * logical.h;
                    res.balls.push_back(d);
                    break;  // cue found
                }
            }
            break;
            }
        }
    }

    // Exactly one cue. White stats alone cannot always tell the cue from a
    // white object ball inside a cluster (Pool Club draws both as white
    // discs), but the CUE ALWAYS SITS ALONE while aiming. Choose the
    // white-ish ball with the largest nearest-neighbor distance; demote the
    // rest back to stripe/solid by their hue.
    {
        int n = static_cast<int>(res.balls.size());
        auto nnDist = [&](int i) {
            double best = 1e18;
            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                double d = std::hypot(res.balls[i].px.x - res.balls[j].px.x,
                                      res.balls[i].px.y - res.balls[j].px.y);
                if (d < best) best = d;
            }
            return best;
        };
        int bestCue = -1;
        double bestIso = -1;
        for (int i = 0; i < n; ++i) {
            const auto& d = res.balls[i];
            // Rim gate again: a white-cap stripe must never win the cue
            // isolation vote even when it is the most isolated white ball.
            bool whiteish = d.rimColored < 0.25 &&
                            (d.kind == "cue" || d.kind == "unknown" ||
                             d.whiteRatio > 0.45);
            if (!whiteish) continue;
            double iso = nnDist(i);
            if (iso > bestIso) {
                bestIso = iso;
                bestCue = i;
            }
        }
        // Tie-break by purity: a ghost disc drawn ON an object ball can be
        // as white and as isolated as the cue — but the ball behind it
        // tints it, so among candidates within 10% of the best isolation
        // the lowest saturation wins.
        if (bestCue >= 0 && bestIso > 2.0 * ballR) {
            double bestSat = res.balls[bestCue].sat;
            for (int i = 0; i < n; ++i) {
                const auto& d = res.balls[i];
                bool whiteish = d.rimColored < 0.25 &&
                                (d.kind == "cue" || d.kind == "unknown" ||
                                 d.whiteRatio > 0.45);
                if (!whiteish || i == bestCue) continue;
                if (nnDist(i) < 0.90 * bestIso) continue;
                if (d.sat < bestSat - 0.005) {
                    bestSat = d.sat;
                    bestCue = i;
                }
            }
        }
        if (bestCue >= 0 && bestIso > 2.0 * ballR) {
            for (int i = 0; i < n; ++i) {
                auto& d = res.balls[i];
                if (i == bestCue) {
                    d.kind = "cue";
                    d.color = "white";
                    d.number = 0;
                    continue;
                }
                if (d.kind == "cue" || d.kind == "unknown") {
                    // Demote: white-heavy -> stripe, else solid by hue.
                    // colorName() returns "unknown" for an invalid hue
                    // (no colored pixels), and a demoted ball without a
                    // color is not a stripe/solid at all.
                    if (d.whiteRatio > opt.stripeWhiteMin) {
                        d.kind = "stripe";
                        d.color = colorName(d.hue, d.val);
                        d.number = numberForColor(d.color, true);
                    } else {
                        d.kind = "solid";
                        d.color = colorName(d.hue, d.val);
                        d.number = numberForColor(d.color, false);
                    }
                    if (d.color == "unknown") {
                        d.kind = "unknown";
                        d.number = -1;
                    }
                }
            }
        } else {
            // No isolated white ball; keep the first classifier cue (if any)
            // and demote the remaining extras.
            int first = -1;
            for (int i = 0; i < n; ++i) {
                if (res.balls[i].kind != "cue") continue;
                if (first < 0) {
                    first = i;
                    continue;
                }
                res.balls[i].kind = "unknown";
                res.balls[i].number = -1;
            }
        }
    }

    // 8b. Merge same-ball split pairs: on gradient felts a ball whose
    // shaded half fell out of the widened surface mask can survive as two
    // blobs ~14 px apart (lit half + dark half). Two detections within
    // 1.5R of each other that agree on kind or color are ONE ball — keep
    // the larger. Genuinely touching balls sit ~2R apart and differ.
    {
        double mergeR = 1.5 * ballR;
        bool mergedPair = true;
        while (mergedPair) {
            mergedPair = false;
            for (size_t i = 0; i < res.balls.size(); ++i)
                for (size_t j = i + 1; j < res.balls.size(); ++j) {
                    const auto& a = res.balls[i];
                    const auto& b = res.balls[j];
                    if (std::hypot(a.px.x - b.px.x, a.px.y - b.px.y) >
                        mergeR)
                        continue;
                    // Never merge a cue detection into anything: the cue
                    // identity carries the whole aim pipeline, and a white
                    // neighbor half a radius away is a different ball.
                    if (a.kind == "cue" || b.kind == "cue") continue;
                    bool kin = a.kind == b.kind;
                    bool cin = a.color == b.color && a.color != "unknown";
                    if (!kin && !cin) continue;
                    size_t keep = a.rPx >= b.rPx ? i : j;
                    size_t drop = keep == i ? j : i;
                    res.balls.erase(res.balls.begin() +
                                    static_cast<long>(drop));
                    mergedPair = true;
                    break;
                }
        }
    }

    res.feltMedV = feltMedV;
    return res;
}

bool estimateAimDirPx(const Image& img, Vec2 cuePx, double ballRPx,
                      const std::vector<Vec2>& otherBallPx, Vec2& dirPx) {
    const int w = img.w, h = img.h;
    const double r0 = std::max(2.0, ballRPx * 1.25);
    const double r1 = std::max(r0 + 6.0, ballRPx * 9.0);
    constexpr double kPi = 3.14159265358979323846;
    auto isWhiteish = [&](double x, double y) {
        int xi = static_cast<int>(x), yi = static_cast<int>(y);
        if (xi < 0 || yi < 0 || xi >= w || yi >= h) return false;
        size_t i = static_cast<size_t>(yi) * w + xi;
        Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1], img.px[i * 3 + 2]);
        return c.s < 0.30 && c.v > 0.65;
    };
    auto inOtherBall = [&](double x, double y) {
        for (const auto& p : otherBallPx)
            if (std::hypot(x - p.x, y - p.y) < ballRPx * 0.95) return true;
        return false;
    };
    auto rayScore = [&](double angDeg) {
        double a = angDeg * kPi / 180.0;
        double dx = std::cos(a), dy = std::sin(a);
        int cnt = 0;
        for (double r = r0; r <= r1; r += 1.5) {
            double x = cuePx.x + dx * r, y = cuePx.y + dy * r;
            if (inOtherBall(x, y)) break;
            if (isWhiteish(x, y)) ++cnt;
        }
        return cnt;
    };
    int bestA = -1, bestCnt = 0;
    for (int a = 0; a < 360; a += 3) {
        int c = rayScore(static_cast<double>(a));
        if (c > bestCnt) { bestCnt = c; bestA = a; }
    }
    if (bestA < 0 || bestCnt < 5) return false;
    int fine = bestA;
    for (int a = bestA - 3; a <= bestA + 3; ++a) {
        int c = rayScore(static_cast<double>((a + 360) % 360));
        if (c > bestCnt) { bestCnt = c; fine = (a + 360) % 360; }
    }
    double a = fine * kPi / 180.0;
    dirPx = Vec2{std::cos(a), std::sin(a)};
    return true;
}

// ---------------------------------------------------------------------------
// Live cue stick: sweep rays of stick-colored pixels from the cue ball.
// The 8BP stick renders as a tan/brown shaft (hue 12..55) with dark edges,
// glued to the ball and rotating around it — its angle IS the aim angle and
// its pull-back distance is the power.
CueStick findCueStick(const Image& img, Vec2 cuePx, double ballRPx,
                      const Rect& field) {
    CueStick out;
    const int w = img.w, h = img.h;
    if (img.empty() || ballRPx <= 0) return out;
    constexpr double kPi = 3.14159265358979323846;
    auto stickPx = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= w || y >= h) return false;
        size_t i = static_cast<size_t>(y) * w + x;
        Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1], img.px[i * 3 + 2]);
        if (c.h >= 12.0 && c.h <= 55.0 && c.s > 0.20 && c.v > 0.25)
            return true;  // wood shaft (the 8BP stick reads hue 26..55)
        // Dark shaft edge / shaft over dark UI — still BROWN (hue 5..60).
        // Without the hue window the navy game background (hue ~210-260,
        // v~0.26) extends every ray to the screen edge and fake sticks
        // appear in every direction.
        if (c.h >= 5.0 && c.h <= 60.0 && c.s < 0.55 && c.v > 0.06 &&
            c.v < 0.32)
            return true;
        return false;
    };
    const double rStart = ballRPx * 1.45, rMax = ballRPx * 26.0;
    // The 8BP stick runs beyond the felt into the window header (the
    // player drags it across the whole window), so rays are bounded by
    // the IMAGE, not the felt. Measured on real captures: the felt edge
    // cut the visible stick to ~3R and the detector missed it.
    const double m = 4.0;
    (void)field;
    // Thick rays: the shaft is only 3-4 px wide, so a single-pixel ray
    // samples it intermittently and the run collapses. Each step probes
    // the center plus two perpendicular offsets (+/-2 px) — a hit on any
    // of the three counts.
    auto rayHit = [&](double x, double y, double dx, double dy) {
        double px = -dy, py = dx;  // perpendicular unit
        for (double off : {0.0, -2.0, 2.0}) {
            double sx = x + px * off, sy = y + py * off;
            if (stickPx(static_cast<int>(sx), static_cast<int>(sy)))
                return true;
        }
        return false;
    };
    auto rayLen = [&](double aDeg, double* tipR) {
        double a = aDeg * kPi / 180.0;
        double dx = std::cos(a), dy = std::sin(a);
        double len = 0, gap = 0;
        bool started = false;
        for (double r = rStart; r <= rMax; r += 1.5) {
            double x = cuePx.x + dx * r, y = cuePx.y + dy * r;
            if (x < -m || y < -m || x > w + m || y > h + m)
                break;
            if (rayHit(x, y, dx, dy)) {
                if (!started && r > rStart + ballRPx * 14.0) break;
                started = true;
                len = r - rStart + ballRPx;
                gap = 0;
            } else if (started) {
                gap += 1.5;
                if (gap > ballRPx * 1.2) break;  // stick ended
            }
        }
        if (tipR && started) {
            // walk back to the first sustained hit for the tip
            int hits = 0;
            for (double r = rStart; r <= rStart + ballRPx * 14.0; r += 1.5) {
                double x = cuePx.x + dx * r, y = cuePx.y + dy * r;
                if (rayHit(x, y, dx, dy)) {
                    if (++hits >= 3) { *tipR = r; break; }
                } else {
                    hits = 0;
                }
            }
        }
        return len;
    };
    // Coarse sweep, 2 deg steps.
    int bestA = -1;
    double bestLen = 0;
    std::vector<double> lens(180, 0.0);
    for (int ai = 0; ai < 180; ++ai) {
        double len = rayLen(ai * 2.0, nullptr);
        lens[ai] = len;
        if (len > bestLen) {
            bestLen = len;
            bestA = ai;
        }
    }
    if (bestA < 0 || bestLen < ballRPx * 5.0) return out;
    // The stick is straight: neighbors within +-2 deg must agree, so a
    // lone rail edge or a wood-colored UI strip cannot win.
    double agree = 0;
    for (int d = -2; d <= 2; ++d)
        agree = std::max(agree, lens[(bestA + d + 180) % 180]);
    if (agree < bestLen * 0.55) return out;
    // Fine sweep, 0.5 deg steps.
    double bestFine = bestA * 2.0, bestScore = -1;
    for (double f = bestA * 2.0 - 2.0; f <= bestA * 2.0 + 2.0; f += 0.5) {
        double len = rayLen(f, nullptr);
        if (len > bestScore) {
            bestScore = len;
            bestFine = f;
        }
    }
    double a = bestFine * kPi / 180.0;
    // The stick extends from the ball at angle a; the aim direction runs
    // from the stick INTO the ball and beyond.
    out.dir = {-std::cos(a), -std::sin(a)};
    double tipR = 0;
    rayLen(bestFine, &tipR);
    if (tipR <= 0) return out;
    out.tip = {cuePx.x + std::cos(a) * tipR, cuePx.y + std::sin(a) * tipR};
    out.pullPx = std::max(0.0, tipR - ballRPx);
    out.found = true;
    // A full draw pulls the tip back by roughly 8 ball radii.
    out.power = std::min(1.0, out.pullPx / (8.0 * ballRPx));
    return out;
}

// The 8BP power slider: a tall saturated-orange bar left of the felt with a
// bright knob. 0 = knob at the top (idle), 1 = knob at the bottom (full
// draw). Returns -1 when the bar or the knob is not visible.
double findPowerSlider(const Image& img, const Rect& field, double ballRPx) {
    const int w = img.w, h = img.h;
    if (img.empty() || ballRPx <= 0) return -1.0;
    int x1 = static_cast<int>(field.x0) - static_cast<int>(ballRPx * 1.2);
    int x0 = x1 - static_cast<int>(ballRPx * 12.0);
    if (x1 <= x0 || x1 <= 0) return -1.0;
    if (x0 < 0) x0 = 0;
    if (x1 >= w) x1 = w - 1;
    auto cls = [&](int x, int y) {
        // 0 = not bar, 1 = bar body (bright orange), 2 = knob (whiter).
        // The knob must count as bar so it does not SPLIT the column run.
        if (x < 0 || y < 0 || x >= w || y >= h) return 0;
        size_t i = static_cast<size_t>(y) * w + x;
        Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1], img.px[i * 3 + 2]);
        if (c.h >= 10.0 && c.h <= 45.0 && c.s > 0.45 && c.v > 0.40) return 1;
        if (c.h >= 10.0 && c.h <= 60.0 && c.s > 0.10 && c.s < 0.45 &&
            c.v > 0.85)
            return 2;
        return 0;
    };
    // Per-column longest bar run (single pass over the strip).
    std::vector<int> runLen(w, 0), runY0(w, -1), runY1(w, -1);
    for (int x = x0; x <= x1; ++x) {
        int run = 0, y0 = -1, brun = 0, by0 = -1, by1 = -1;
        for (int y = 0; y < h; ++y) {
            int cl = cls(x, y);
            if (cl) {
                if (run == 0) y0 = y;
                ++run;
                if (run > brun) {
                    brun = run;
                    by0 = y0;
                    by1 = y;
                }
            } else {
                run = 0;
            }
        }
        runLen[x] = brun;
        runY0[x] = by0;
        runY1[x] = by1;
    }
    int bestCol = -1, bestRun = 0;
    for (int x = x0; x <= x1; ++x)
        if (runLen[x] > bestRun) {
            bestRun = runLen[x];
            bestCol = x;
        }
    if (bestCol < 0 || bestRun < ballRPx * 4.0) return -1.0;
    int bestY0 = runY0[bestCol], bestY1 = runY1[bestCol];
    // Bar width: neighbouring columns with a comparably long run. The
    // bar is narrow (5-8 px) and surrounded by the dark page background —
    // sampling medians beyond the bar edge would poison them with black.
    int bx0 = bestCol, bx1 = bestCol;
    while (bx0 - 1 >= x0 && runLen[bx0 - 1] >= bestRun * 0.6) --bx0;
    while (bx1 + 1 <= x1 && runLen[bx1 + 1] >= bestRun * 0.6) ++bx1;
    // Bar body statistics (body class only) from the bar columns.
    long nS = 0;
    double sumS = 0, sumV = 0;
    for (int y = bestY0; y <= bestY1; ++y)
        for (int x = bx0; x <= bx1; ++x) {
            size_t i = static_cast<size_t>(y) * w + x;
            Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1],
                             img.px[i * 3 + 2]);
            bool body = c.h >= 10.0 && c.h <= 45.0 && c.s > 0.45 &&
                        c.v > 0.40;
            if (!body) continue;
            sumS += c.s;
            sumV += c.v;
            ++nS;
        }
    if (nS < 10) return -1.0;
    double medS = sumS / nS, medV = sumV / nS;
    // Knob: the row band whose pixels are distinctly whiter than the bar
    // body (saturation drop). Some skins show no marker while idle —
    // report unknown (-1) then, never a fake number.
    int bRun = 0, bRun0 = -1, knobRun = 0, run0 = -1;
    for (int y = bestY0; y <= bestY1; ++y) {
        bool b = false;
        for (int x = bx0; x <= bx1; ++x) {
            size_t i = static_cast<size_t>(y) * w + x;
            Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1],
                             img.px[i * 3 + 2]);
            if (c.v > medV + 0.02 && c.s < medS - 0.15) {
                b = true;
                break;
            }
        }
        if (b) {
            if (run0 < 0) run0 = y;
            ++knobRun;
            if (knobRun > bRun) {
                bRun = knobRun;
                bRun0 = run0;
            }
        } else {
            knobRun = 0;
            run0 = -1;
        }
    }
    if (bRun < 2) return -1.0;
    double barH = static_cast<double>(bestY1 - bestY0);
    if (barH < 1.0) return -1.0;
    double knobC = bRun0 + bRun * 0.5;
    double frac = (knobC - bestY0) / barH;
    if (frac < 0) frac = 0;
    if (frac > 1) frac = 1;
    return frac;
}

}  // namespace pool