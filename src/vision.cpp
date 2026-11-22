// vision.cpp — segmentation, blob detection and HSV ball classification.
//
// Pipeline:
//   1. RGB -> HSV; build the felt mask (green hue window).
//   2. Largest 4-connected felt component -> playing-field rect.
//   3. Inside the rect: "ball-ish" pixels = NOT felt.
//   4. 8-connected components over ball-ish pixels -> blobs.
//   5. Median blob area -> ball radius estimate; keep blobs of plausible size.
//   6. Exclude blobs centered on the six pocket anchors.
//   7. Classify each blob by HSV statistics inside the disc.
//   8. Map centroids to logical table coordinates.
#include "pool/vision.hpp"
#include "stb_image.h"
#include "stb_image_write.h"
#include <algorithm>
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
    struct Range { const char* name; double lo, hi; };
    static const Range ranges[] = {
        {"red", 345.0, 360.0}, {"red", 0.0, 15.0},
        {"orange", 15.0, 40.0},
        {"yellow", 40.0, 70.0},
        {"green", 70.0, 165.0},
        {"blue", 165.0, 255.0},
        {"purple", 255.0, 320.0},
        {"red", 320.0, 345.0},  // magenta-ish reads as the red family here
    };
    // Dark red = maroon (ball 7).
    bool inRed = (hue >= 345.0 || hue < 15.0);
    if (inRed && coloredVal < 0.45) return "maroon";
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

}  // namespace

DetectResult detectBalls(const Image& img, const DetectOptions& opt,
                         const TableSpec& logical) {
    if (img.empty()) throw std::runtime_error("empty image");
    const int w = img.w, h = img.h;
    const size_t n = static_cast<size_t>(w) * h;

    // 1. Felt mask.
    std::vector<uint8_t> felt(n, 0);
    for (size_t i = 0; i < n; ++i) {
        Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1], img.px[i * 3 + 2]);
        if (c.h >= opt.feltHueMin && c.h <= opt.feltHueMax &&
            c.s >= opt.feltSatMin && c.v >= opt.feltValMin)
            felt[i] = 1;
    }

    // 2. Largest felt component -> table rect.
    std::vector<uint8_t> visited(n, 0);
    Rect best{0, 0, 0, 0};
    long bestArea = 0;
    for (size_t i = 0; i < n; ++i) {
        if (felt[i] && !visited[i]) {
            Rect r;
            long a = 0;
            floodFill(felt, w, h, static_cast<int>(i % w),
                      static_cast<int>(i / w), visited, r, a);
            if (a > bestArea) { bestArea = a; best = r; }
        }
    }
    if (bestArea < 100)
        throw std::runtime_error("no table (felt) region found");

    // Shrink a little to avoid the cushion/wood transition pixels.
    Rect field = best;
    field.x0 += 2; field.y0 += 2;
    field.x1 -= 2; field.y1 -= 2;

    // 3. Ball-ish mask inside the field.
    std::vector<uint8_t> ballish(n, 0);
    for (int y = static_cast<int>(field.y0); y < static_cast<int>(field.y1); ++y) {
        for (int x = static_cast<int>(field.x0); x < static_cast<int>(field.x1); ++x) {
            size_t i = static_cast<size_t>(y) * w + x;
            if (!felt[i]) ballish[i] = 1;
        }
    }

    // 4. Blobs.
    std::vector<Blob> blobs = findBlobs(ballish, w, h);

    // 5. Median area of plausible blobs -> ball radius. Pocket blobs are
    // excluded from the statistics first (they sit on the six anchors) so
    // they cannot skew the median.
    double minDim = std::min(field.w(), field.h());
    double pocketExclPre = 0.07 * minDim;
    Vec2 anchors[6] = {
        {field.x0, field.y0}, {(field.x0 + field.x1) / 2, field.y0},
        {field.x1, field.y0}, {field.x0, field.y1},
        {(field.x0 + field.x1) / 2, field.y1}, {field.x1, field.y1}};
    auto nearAnyAnchor = [&](const Blob& b, double rad) {
        for (const auto& a : anchors)
            if (std::hypot(b.cx - a.x, b.cy - a.y) < rad) return true;
        return false;
    };
    std::vector<long> areas;
    for (const auto& b : blobs) {
        if (b.area < opt.areaMin) continue;
        if (nearAnyAnchor(b, pocketExclPre)) continue;
        areas.push_back(b.area);
    }
    if (areas.empty())
        throw std::runtime_error("no ball-like blobs found inside the table");
    std::sort(areas.begin(), areas.end());
    double medA = areas[areas.size() / 2];
    double ballR = std::sqrt(medA / 3.14159265358979323846);

    // Final exclusion radius around the pocket anchors.
    double pocketExcl = std::max(pocketExclPre, 3.04 * ballR);

    // 6+7. Keep plausible blobs, skip pockets, classify.
    DetectResult res;
    res.feltRect = best;
    res.ballRPx = ballR;
    res.table = logical;

    struct Raw { Blob b; };
    std::vector<Raw> kept;
    for (const auto& b : blobs) {
        if (b.area < opt.areaMin) continue;
        if (b.area < medA * opt.areaLoFactor || b.area > medA * opt.areaHiFactor)
            continue;
        bool nearPocket = false;
        for (const auto& a : anchors)
            if (std::hypot(b.cx - a.x, b.cy - a.y) < pocketExcl) { nearPocket = true; break; }
        if (nearPocket) continue;
        kept.push_back({b});
    }

    for (const auto& raw : kept) {
        const auto& b = raw.b;
        double r = std::sqrt(b.area / 3.14159265358979323846);
        double ir = std::max(1.0, r * 0.85);  // sampling radius inside the disc
        long whiteCnt = 0, blackCnt = 0, colCnt = 0;
        double sumS = 0, sumV = 0, sumColV = 0;
        int hist[24] = {0};
        long total = 0;
        int x0 = static_cast<int>(b.cx - ir), x1 = static_cast<int>(b.cx + ir);
        int y0 = static_cast<int>(b.cy - ir), y1 = static_cast<int>(b.cy + ir);
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                if (x < 0 || y < 0 || x >= w || y >= h) continue;
                double dx = x - b.cx, dy = y - b.cy;
                if (dx * dx + dy * dy > ir * ir) continue;
                size_t i = static_cast<size_t>(y) * w + x;
                Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1], img.px[i * 3 + 2]);
                ++total;
                sumS += c.s;
                sumV += c.v;
                bool white = c.s < opt.cueSatMax && c.v > opt.cueValMin;
                bool black = c.v < opt.blackValMax;
                if (white) ++whiteCnt;
                if (black) ++blackCnt;
                if (c.s > 0.30 && c.v > 0.20) {
                    ++colCnt;
                    sumColV += c.v;
                    ++hist[static_cast<int>(c.h / 15.0) % 24];
                }
            }
        }
        if (total == 0) continue;
        Detection d;
        d.px = {b.cx, b.cy};
        d.rPx = r;
        d.area = static_cast<double>(b.area);
        d.whiteRatio = static_cast<double>(whiteCnt) / total;
        d.sat = sumS / total;
        d.val = sumV / total;
        int hueBin = static_cast<int>(std::max_element(hist, hist + 24) - hist);
        d.hue = (hueBin + 0.5) * 15.0;
        double coloredVal = colCnt > 0 ? sumColV / colCnt : 0.0;
        double blackRatio = static_cast<double>(blackCnt) / total;

        if (blackRatio > 0.55) {
            d.kind = "black"; d.color = "black"; d.number = 8;
        } else if (d.whiteRatio > 0.80) {
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

        // Logical mapping (affine stretch of the felt rect onto the table).
        d.logical.x = (b.cx - field.x0) / field.w() * logical.w;
        d.logical.y = (b.cy - field.y0) / field.h() * logical.h;
        res.balls.push_back(d);
    }

    // Exactly one cue: keep the biggest white blob, demote the others.
    int cueCount = 0;
    int bestCue = -1;
    for (int i = 0; i < static_cast<int>(res.balls.size()); ++i) {
        if (res.balls[i].kind == "cue") {
            ++cueCount;
            if (bestCue < 0 || res.balls[i].area > res.balls[bestCue].area)
                bestCue = i;
        }
    }
    if (cueCount > 1) {
        for (int i = 0; i < static_cast<int>(res.balls.size()); ++i) {
            if (i == bestCue) continue;
            if (res.balls[i].kind == "cue") {
                res.balls[i].kind = "unknown";
                res.balls[i].number = -1;
            }
        }
    }

    return res;
}

}  // namespace pool
