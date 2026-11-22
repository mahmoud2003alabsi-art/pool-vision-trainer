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
    static const double kPassV[3] = {0.45, 0.30, 0.07};
    static const double kPassS[3] = {0.30, 0.22, 0.0};
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
            if (hist[bin] > bestCnt) { bestCnt = hist[bin]; bestBin = bin; }
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
            float r = d / 3.0f;
            if (r > best && r <= cap) {
                best = r;
                center = {static_cast<double>(x), static_cast<double>(y)};
            }
        }
    return best;
}

}  // namespace

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
        best = largestComponent(surface, w, h, bestArea);
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

    // Shrink a little to avoid the cushion/wood transition pixels.
    Rect field = best;
    field.x0 += 2; field.y0 += 2;
    field.x1 -= 2; field.y1 -= 2;

    // 3. Ball-ish mask inside the field: anything not matching the surface.
    std::vector<uint8_t> ballish(n, 0);
    for (int y = static_cast<int>(field.y0); y < static_cast<int>(field.y1); ++y) {
        for (int x = static_cast<int>(field.x0); x < static_cast<int>(field.x1); ++x) {
            size_t i = static_cast<size_t>(y) * w + x;
            if (!surface[i]) ballish[i] = 1;
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
    Vec2 anchors[6] = {
        {field.x0, field.y0}, {(field.x0 + field.x1) / 2, field.y0},
        {field.x1, field.y0}, {field.x0, field.y1},
        {(field.x0 + field.x1) / 2, field.y1}, {field.x1, field.y1}};
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
        double medA = areas[areas.size() / 2];
        ballR = std::sqrt(medA / 3.14159265358979323846);
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
            res.table.pockets[k].pos = {
                (moved.x - field.x0) / field.w() * logical.w,
                (moved.y - field.y0) / field.h() * logical.h};
        }
    }

    struct Raw { Blob b; };
    std::vector<Raw> kept;
    int salvaged = 0;
    for (const auto& b : blobs) {
        Blob c = b;
        bool ok = c.area >= opt.areaMin;
        if (ok)
            ok = c.area >= refArea * opt.areaLoFactor &&
                 c.area <= refArea * opt.areaHiFactor;
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
        if (!ok) {
            // Merged structures — cue + stick, cue + guideline, touching
            // clusters — still contain a ball-sized inscribed disc. Salvage
            // that disc and classify it; the stick/line remainder is dropped.
            // Gates: only ball+appendage sized blobs, and the salvaged disc
            // must not be felt-colored (shadow bands are full of discs).
            if (c.area > refArea * 5.0) continue;
            if (salvaged >= 12) continue;  // junk images explode otherwise
            Vec2 sc{};
            Rect grown{c.bbox.x0 - ballR - 2, c.bbox.y0 - ballR - 2,
                       c.bbox.x1 + ballR + 2, c.bbox.y1 + ballR + 2};
            double sr = inscribedDisc(ballish, w, h, grown, ballR, sc);
            if (sr < 0.72 * ballR) continue;
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
                if (tot > 0 && felt > tot * 0.6) continue;  // felt fragment
                // Gray/dim discs are UI junk, not balls (balls are white,
                // colored or black).
                if (tot > 0 && sumS / tot < 0.12 && sumV / tot < 0.72) continue;
            }
            ++salvaged;
            c.cx = sc.x;
            c.cy = sc.y;
            c.area = static_cast<long>(3.14159265358979323846 * sr * sr);
            c.bbox = Rect{sc.x - sr, sc.y - sr, sc.x + sr, sc.y + sr};
        }
        bool nearPocket = false;
        for (const auto& a : anchors)
            if (std::hypot(c.cx - a.x, c.cy - a.y) < pocketExcl) { nearPocket = true; break; }
        if (nearPocket) continue;
        bool dup = false;
        for (const auto& k : kept) {
            if (std::hypot(k.b.cx - c.cx, k.b.cy - c.cy) < 0.8 * ballR) {
                dup = true;
                break;
            }
        }
        if (dup) continue;
        kept.push_back({c});
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

        // Felt-texture fragments (shadow bands, gradient clumps) carry the
        // felt's own color — real balls never do. Blue 8BP balls sit ~25deg
        // away from the felt hue, so they survive this gate.
        double fhd = std::fabs(d.hue - feltHue.h);
        fhd = std::min(fhd, 360.0 - fhd);
        if (usedFallback && std::fabs(d.sat - feltHue.s) < 0.25 &&
            std::fabs(d.val - feltHue.v) < 0.30 && fhd < 15.0)
            continue;

        // Logical mapping (affine stretch of the felt rect onto the table).
        d.logical.x = (b.cx - field.x0) / field.w() * logical.w;
        d.logical.y = (b.cy - field.y0) / field.h() * logical.h;
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
            for (int y = static_cast<int>(field.y0); y < static_cast<int>(field.y1); ++y) {
                for (int x = static_cast<int>(field.x0); x < static_cast<int>(field.x1); ++x) {
                    size_t i = static_cast<size_t>(y) * w + x;
                    if (!ballish[i]) continue;
                    Hsv c = rgbToHsv(img.px[i * 3], img.px[i * 3 + 1],
                                     img.px[i * 3 + 2]);
                    if (c.s < 0.25 && c.v > 0.70) wm[i] = 1;
                }
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
                        if (c.s < 0.30 && c.v > 0.62) ++wCnt;
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

}  // namespace pool
