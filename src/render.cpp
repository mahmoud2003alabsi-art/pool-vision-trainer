// render.cpp — raster drawing + tiny 3x5 bitmap font.
#include "pool/render.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

namespace pool {

namespace {

// 3x5 font: 5 rows of 3 bits, MSB = leftmost pixel. Uppercase subset only.
// Keys: characters we actually render on annotated images.
uint8_t glyphRow(char c, int row) {
    // clang-format off
    static const std::map<char, std::array<uint8_t, 5>> G = {
        {'A', {0b010, 0b101, 0b111, 0b101, 0b101}},
        {'B', {0b110, 0b101, 0b110, 0b101, 0b110}},
        {'C', {0b011, 0b100, 0b100, 0b100, 0b011}},
        {'D', {0b110, 0b101, 0b101, 0b101, 0b110}},
        {'E', {0b111, 0b100, 0b110, 0b100, 0b111}},
        {'F', {0b111, 0b100, 0b110, 0b100, 0b100}},
        {'G', {0b011, 0b100, 0b101, 0b101, 0b011}},
        {'H', {0b101, 0b101, 0b111, 0b101, 0b101}},
        {'I', {0b111, 0b010, 0b010, 0b010, 0b111}},
        {'K', {0b101, 0b101, 0b110, 0b101, 0b101}},
        {'L', {0b100, 0b100, 0b100, 0b100, 0b111}},
        {'M', {0b101, 0b111, 0b111, 0b101, 0b101}},
        {'N', {0b101, 0b111, 0b111, 0b111, 0b101}},
        {'O', {0b010, 0b101, 0b101, 0b101, 0b010}},
        {'P', {0b110, 0b101, 0b110, 0b100, 0b100}},
        {'R', {0b110, 0b101, 0b110, 0b101, 0b101}},
        {'S', {0b011, 0b100, 0b010, 0b001, 0b110}},
        {'T', {0b111, 0b010, 0b010, 0b010, 0b010}},
        {'U', {0b101, 0b101, 0b101, 0b101, 0b111}},
        {'V', {0b101, 0b101, 0b101, 0b101, 0b010}},
        {'W', {0b101, 0b101, 0b111, 0b111, 0b101}},
        {'X', {0b101, 0b101, 0b010, 0b101, 0b101}},
        {'Y', {0b101, 0b101, 0b010, 0b010, 0b010}},
        {'0', {0b111, 0b101, 0b101, 0b101, 0b111}},
        {'1', {0b010, 0b110, 0b010, 0b010, 0b111}},
        {'2', {0b110, 0b001, 0b010, 0b100, 0b111}},
        {'3', {0b111, 0b001, 0b011, 0b001, 0b111}},
        {'4', {0b101, 0b101, 0b111, 0b001, 0b001}},
        {'5', {0b111, 0b100, 0b110, 0b001, 0b110}},
        {'6', {0b011, 0b100, 0b111, 0b101, 0b111}},
        {'7', {0b111, 0b001, 0b010, 0b010, 0b010}},
        {'8', {0b111, 0b101, 0b111, 0b101, 0b111}},
        {'9', {0b111, 0b101, 0b111, 0b001, 0b110}},
        {'=', {0b000, 0b000, 0b111, 0b000, 0b111}},
        {'-', {0b000, 0b000, 0b111, 0b000, 0b000}},
        {'.', {0b000, 0b000, 0b000, 0b000, 0b010}},
        {':', {0b000, 0b010, 0b000, 0b010, 0b000}},
        {'%', {0b101, 0b001, 0b010, 0b100, 0b101}},
        {' ', {0b000, 0b000, 0b000, 0b000, 0b000}},
    };
    // clang-format on
    auto it = G.find(c);
    if (it == G.end()) return 0;
    return it->second[row];
}

}  // namespace

void drawLine(Image& img, Vec2 a, Vec2 b, Rgb c, int thick, bool dashed,
              int dashLen) {
    double len = a.dist(b);
    if (len < 1e-9) return;
    int steps = static_cast<int>(len * 2);
    double dashPhase = 0;
    bool on = true;
    for (int i = 0; i <= steps; ++i) {
        double t = i / static_cast<double>(steps);
        Vec2 p = a + (b - a) * t;
        if (dashed) {
            // advance dash phase by distance per step
            double stepDist = len / steps;
            if (dashPhase >= dashLen) { on = !on; dashPhase = 0; }
            dashPhase += stepDist;
            if (!on) continue;
        }
        int half = thick / 2;
        for (int dy = -half; dy <= half; ++dy)
            for (int dx = -half; dx <= half; ++dx)
                img.set(static_cast<int>(p.x) + dx, static_cast<int>(p.y) + dy,
                        c.r, c.g, c.b);
    }
}

void drawCircle(Image& img, Vec2 c, double r, Rgb ccol, int thick, bool filled) {
    if (filled) {
        for (int y = static_cast<int>(c.y - r); y <= static_cast<int>(c.y + r); ++y)
            for (int x = static_cast<int>(c.x - r); x <= static_cast<int>(c.x + r); ++x) {
                double dx = x - c.x, dy = y - c.y;
                if (dx * dx + dy * dy <= r * r)
                    img.set(x, y, ccol.r, ccol.g, ccol.b);
            }
        return;
    }
    // Outline: walk the full circle, stamp "thick" pixels.
    int steps = static_cast<int>(2 * 3.14159265 * r * 2);
    if (steps < 8) steps = 8;
    int half = thick / 2;
    for (int i = 0; i < steps; ++i) {
        double ang = i * 2 * 3.14159265358979323846 / steps;
        int x = static_cast<int>(c.x + r * std::cos(ang));
        int y = static_cast<int>(c.y + r * std::sin(ang));
        for (int dy = -half; dy <= half; ++dy)
            for (int dx = -half; dx <= half; ++dx)
                img.set(x + dx, y + dy, ccol.r, ccol.g, ccol.b);
    }
}

void drawRect(Image& img, const Rect& r, Rgb c, int thick) {
    Vec2 a{r.x0, r.y0}, b{r.x1, r.y0}, c2{r.x1, r.y1}, d{r.x0, r.y1};
    drawLine(img, a, b, c, thick);
    drawLine(img, b, c2, c, thick);
    drawLine(img, c2, d, c, thick);
    drawLine(img, d, a, c, thick);
}

Vec2 textSize(const std::string& s, int scale) {
    int cols = 0;
    for (char ch : s) {
        (void)ch;
        cols += 4;  // 3 glyph columns + 1 spacing
    }
    if (!s.empty()) cols -= 1;
    return {static_cast<double>(cols * scale), 5.0 * scale};
}

void drawText(Image& img, Vec2 topLeft, const std::string& s, Rgb c, int scale) {
    double cx = topLeft.x, cy = topLeft.y;
    for (char ch : s) {
        char up = ch >= 'a' && ch <= 'z' ? static_cast<char>(ch - 'a' + 'A') : ch;
        for (int row = 0; row < 5; ++row) {
            uint8_t bits = glyphRow(up, row);
            for (int col = 0; col < 3; ++col) {
                if (bits & (0b100 >> col)) {
                    for (int sy = 0; sy < scale; ++sy)
                        for (int sx = 0; sx < scale; ++sx)
                            img.set(static_cast<int>(cx + col * scale + sx),
                                    static_cast<int>(cy + row * scale + sy),
                                    c.r, c.g, c.b);
                }
            }
        }
        cx += 4 * scale;
    }
}

void drawTextCentered(Image& img, Vec2 center, const std::string& s, Rgb c,
                      int scale) {
    Vec2 ts = textSize(s, scale);
    drawText(img, center - ts / 2.0, s, c, scale);
}

}  // namespace pool
