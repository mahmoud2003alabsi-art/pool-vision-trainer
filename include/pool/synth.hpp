// synth.hpp — synthetic pool table generator. Produces realistic top-down
// table images (felt with noise, wood border, pockets, solid/stripe balls)
// plus ground-truth scene data, used for the demo pipeline and CI tests.
#pragma once
#include "pool/geom.hpp"
#include "pool/render.hpp"
#include "pool/vision.hpp"
#include <string>
#include <vector>

namespace pool {

struct SceneBall {
    int number = 0;        // 0 = cue, 1..15, 8 = black
    std::string kind;      // cue | solid | stripe | black
    std::string color;
    Vec2 pos;
};

struct SynthScene {
    TableSpec table;
    std::vector<SceneBall> balls;
};

// Curated layout: a clean direct shot (cue -> red 3 -> bottom-right pocket)
// plus bank/kick opportunities for the other balls.
SynthScene demoScene();

// Random layout with rejection sampling (no overlaps, inside the field).
SynthScene randomScene(unsigned seed);

// Render the scene into an image (image is table + margin for the wood).
Image renderSynth(const SynthScene& scene, unsigned noiseSeed = 42);

// RGB values used for each ball family (also documents the classifier's
// color expectations).
Rgb ballColorRGB(const std::string& color);

}  // namespace pool
