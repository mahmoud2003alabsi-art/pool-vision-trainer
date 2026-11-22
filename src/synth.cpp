// synth.cpp — synthetic table renderer.
#include "pool/synth.hpp"
#include "pool/render.hpp"
#include <cmath>
#include <random>

namespace pool {

Rgb ballColorRGB(const std::string& color) {
    if (color == "yellow") return {244, 205, 10};
    if (color == "blue") return {22, 68, 198};
    if (color == "red") return {208, 30, 36};
    if (color == "purple") return {112, 40, 170};
    if (color == "orange") return {235, 118, 20};
    if (color == "green") return {22, 140, 60};
    if (color == "maroon") return {100, 16, 24};  // dark red family, V < 0.45
    if (color == "black") return {16, 16, 19};
    return {245, 248, 250};  // white / cue
}

SynthScene demoScene() {
    SynthScene s;
    s.table = TableSpec::standard();  // 900 x 450, R = 12
    s.balls = {
        {0, "cue", "white", {380, 120}},
        {1, "solid", "yellow", {420, 330}},
        {3, "solid", "red", {520, 180}},
        {8, "black", "black", {700, 380}},
        {9, "stripe", "yellow", {300, 120}},
        {12, "stripe", "purple", {620, 350}},
    };
    return s;
}

SynthScene randomScene(unsigned seed) {
    std::mt19937 rng(seed);
    SynthScene s;
    s.table = TableSpec::standard();
    const double R = s.table.ballR;
    const char* solids[7] = {"yellow", "blue", "red", "purple",
                             "orange", "green", "maroon"};
    const char* stripes[7] = {"yellow", "blue", "red", "purple",
                              "orange", "green", "maroon"};

    auto rnd = [&](double lo, double hi) {
        return std::uniform_real_distribution<double>(lo, hi)(rng);
    };

    // Cue in the left half, object balls scattered on the right.
    SceneBall cue;
    cue.number = 0; cue.kind = "cue"; cue.color = "white";
    cue.pos = {rnd(3 * R, s.table.w * 0.45), rnd(3 * R, s.table.h - 3 * R)};
    s.balls.push_back(cue);

    int n = 4 + static_cast<int>(rng() % 4);  // 4..7 object balls
    for (int i = 0; i < n; ++i) {
        for (int tries = 0; tries < 200; ++tries) {
            Vec2 p{rnd(s.table.w * 0.30, s.table.w - 3 * R),
                   rnd(3 * R, s.table.h - 3 * R)};
            bool ok = true;
            for (const auto& b : s.balls)
                if (p.dist(b.pos) < 2.6 * R) { ok = false; break; }
            if (!ok) continue;
            SceneBall nb;
            bool stripe = (rng() % 2) == 1;
            int idx = static_cast<int>(rng() % 7);
            if (stripe) {
                nb.number = idx + 9;
                nb.kind = "stripe";
                nb.color = stripes[idx];
            } else {
                nb.number = idx + 1;
                nb.kind = "solid";
                nb.color = solids[idx];
            }
            nb.pos = p;
            s.balls.push_back(nb);
            break;
        }
    }
    // Ensure the black ball exists.
    SceneBall black;
    black.number = 8; black.kind = "black"; black.color = "black";
    for (int tries = 0; tries < 300; ++tries) {
        Vec2 p{rnd(s.table.w * 0.30, s.table.w - 3 * R),
               rnd(3 * R, s.table.h - 3 * R)};
        bool ok = true;
        for (const auto& b : s.balls)
            if (p.dist(b.pos) < 2.6 * R) { ok = false; break; }
        if (ok) { black.pos = p; s.balls.push_back(black); break; }
    }
    return s;
}

Image renderSynth(const SynthScene& scene, unsigned noiseSeed) {
    const TableSpec& t = scene.table;
    const int margin = 18;  // wood frame thickness in px
    Image img;
    img.w = static_cast<int>(t.w) + 2 * margin;
    img.h = static_cast<int>(t.h) + 2 * margin;
    img.px.assign(static_cast<size_t>(img.w) * img.h * 3, 0);

    std::mt19937 noiseRng(noiseSeed);

    // Wood border.
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x)
            img.set(x, y, 84, 52, 24);

    // Felt with per-pixel noise (keeps the classifier honest).
    const unsigned char fr = 34, fg = 138, fb = 84;
    for (int y = margin; y < img.h - margin; ++y) {
        for (int x = margin; x < img.w - margin; ++x) {
            int nz = static_cast<int>(noiseRng() % 13) - 6;
            auto cl = [&](int v) {
                return static_cast<unsigned char>(std::max(0, std::min(255, v)));
            };
            img.set(x, y, cl(fr + nz), cl(fg + nz), cl(fb + nz));
        }
    }

    // Logical -> pixel mapping (felt origin at margin).
    auto toPx = [&](Vec2 p) { return Vec2{p.x + margin, p.y + margin}; };

    // Pockets: dark circles slightly recessed into the felt.
    Rgb pocketCol{14, 14, 17};
    for (const auto& pk : t.pockets) {
        Vec2 c = toPx(pk.pos);
        // Corner pockets sit on the corner; nudge them a touch inward for
        // realism. Side pockets sit on the edge lines.
        drawCircle(img, c, t.pocketR, pocketCol, 1, true);
    }

    // Balls.
    const double R = t.ballR;
    for (const auto& b : scene.balls) {
        Vec2 c = toPx(b.pos);
        // Shadow (dark felt tone — the detector treats it as felt).
        drawCircle(img, c + Vec2{3, 4}, R * 1.02, {26, 96, 58}, 1, true);

        Rgb col = ballColorRGB(b.color);
        if (b.kind == "stripe") {
            // White disc with a horizontal color band (keeps the white
            // fraction comfortably above the classifier's stripe threshold).
            drawCircle(img, c, R, {244, 246, 248}, 1, true);
            double bandH = R * 0.80;
            for (int y = static_cast<int>(c.y - bandH);
                 y <= static_cast<int>(c.y + bandH); ++y) {
                for (int x = static_cast<int>(c.x - R); x <= static_cast<int>(c.x + R); ++x) {
                    double dx = x - c.x, dy = y - c.y;
                    if (dx * dx + dy * dy <= R * R && std::fabs(dy) <= bandH / 2)
                        img.set(x, y, col.r, col.g, col.b);
                }
            }
        } else {
            drawCircle(img, c, R, col, 1, true);
        }
        // Small white number disc on solid balls (like a printed number spot).
        if (b.kind == "solid")
            drawCircle(img, c, R * 0.30, {244, 246, 248}, 1, true);
        // Specular highlight.
        drawCircle(img, c - Vec2{R * 0.35, R * 0.35}, R * 0.16, {250, 252, 254},
                   1, true);
    }
    return img;
}

}  // namespace pool
