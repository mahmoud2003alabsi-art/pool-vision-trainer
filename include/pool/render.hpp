// render.hpp — drawing helpers on top of Image: lines, dashed lines,
// circles and a tiny built-in 3x5 bitmap font (uppercase ASCII subset).
#pragma once
#include "pool/vision.hpp"
#include <string>

namespace pool {

struct Rgb {
    unsigned char r = 255, g = 255, b = 255;
};

void drawLine(Image& img, Vec2 a, Vec2 b, Rgb c, int thick = 2,
              bool dashed = false, int dashLen = 7);
void drawCircle(Image& img, Vec2 c, double r, Rgb ccol, int thick = 2,
                bool filled = false);
void drawRect(Image& img, const Rect& r, Rgb c, int thick = 2);
void drawText(Image& img, Vec2 topLeft, const std::string& s, Rgb c,
              int scale = 2);
// Center the text block at `center`.
void drawTextCentered(Image& img, Vec2 center, const std::string& s, Rgb c,
                      int scale = 2);
Vec2 textSize(const std::string& s, int scale);

}  // namespace pool
