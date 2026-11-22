// main.cpp — pool-trainer CLI.
//
//   pool-trainer demo   [--out DIR] [--seed N]
//       Generate a synthetic table, detect its balls, solve shots, write
//       table.png / table_annotated.png / result.json and a ground-truth
//       accuracy report.
//
//   pool-trainer analyze <image> [--out DIR] [--table WxH] [--max-rails N]
//       Detect balls on any top-down table image, solve shots, write
//       <image>_annotated.png / <image>_result.json.
//
//   pool-trainer version
#include "pool/jsonw.hpp"
#include "pool/render.hpp"
#include "pool/solver.hpp"
#include "pool/synth.hpp"
#include "pool/vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using namespace pool;

namespace {

struct Args {
    std::string cmd;
    std::string in;
    std::string out = "output";
    unsigned seed = 42;
    double tableW = 900, tableH = 450;
    int maxRails = 2;
    int top = 3;
};

Args parseArgs(int argc, char** argv) {
    Args a;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (s == "--out") a.out = need(s.c_str());
        else if (s == "--seed") a.seed = static_cast<unsigned>(std::stoul(need(s.c_str())));
        else if (s == "--table") {
            std::string v = need(s.c_str());
            size_t x = v.find('x');
            if (x == std::string::npos) {
                std::fprintf(stderr, "--table expects WxH (e.g. 900x450)\n");
                std::exit(2);
            }
            a.tableW = std::stod(v.substr(0, x));
            a.tableH = std::stod(v.substr(x + 1));
        } else if (s == "--max-rails") a.maxRails = std::stoi(need(s.c_str()));
        else if (s == "--top") a.top = std::stoi(need(s.c_str()));
        else if (s == "demo" || s == "analyze" || s == "version") a.cmd = s;
        else pos.push_back(s);
    }
    if (!pos.empty()) a.in = pos[0];
    return a;
}

Rgb shotColor(int i) {
    static const Rgb palette[3] = {
        {255, 214, 40},   // amber
        {255, 140, 30},   // orange
        {235, 80, 200},   // magenta
    };
    return palette[i & 3];
}

std::string labelFor(const Detection& d) {
    if (d.number == 0) return "CUE";
    if (d.number == 8) return "8";
    if (d.number > 0) return std::to_string(d.number);
    return "?";
}

// The full pipeline shared by demo and analyze.
int runPipeline(const Image& img, const TableSpec& table, const std::string& outDir,
                const std::string& baseName, int maxRails, int top,
                const SynthScene* groundTruth = nullptr) {
    fs::create_directories(outDir);

    DetectOptions dopt;
    DetectResult det;
    try {
        det = detectBalls(img, dopt, table);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "detection failed: %s\n", e.what());
        return 1;
    }

    TableSpec tbl = det.table;  // detected table (pockets snapped to real holes)
    std::cout << "table rect px: (" << det.feltRect.x0 << "," << det.feltRect.y0
              << ")-(" << det.feltRect.x1 << "," << det.feltRect.y1 << ")"
              << "  ballR_px=" << det.ballRPx << "\n";
    std::cout << "detected " << det.balls.size() << " balls:\n";
    for (const auto& d : det.balls) {
        std::printf("  %-5s %-8s num=%2d  logical=(%6.1f,%6.1f)  px=(%5.1f,%5.1f)"
                    "  white=%.2f val=%.2f hue=%.0f\n",
                    d.kind.c_str(), d.color.c_str(), d.number, d.logical.x,
                    d.logical.y, d.px.x, d.px.y, d.whiteRatio, d.val, d.hue);
    }

    // Build the solver ball list from detections.
    std::vector<Ball> balls;
    int cueIdx = -1;
    for (size_t i = 0; i < det.balls.size(); ++i) {
        const auto& d = det.balls[i];
        Ball b;
        b.id = d.number;
        b.kind = d.kind;
        b.color = d.color;
        b.pos = d.logical;
        b.r = tbl.ballR;
        if (d.kind == "cue" && cueIdx < 0) cueIdx = static_cast<int>(i);
        balls.push_back(b);
    }
    if (cueIdx < 0) {
        std::fprintf(stderr, "warning: no cue ball detected; only geometry reports\n");
    }

    Solver solver(tbl, balls, cueIdx, SolverOptions{maxRails, 80.0, 0.98});
    std::vector<Shot> shots = solver.allShots();

    std::cout << "\nshots found: " << shots.size() << " (top " << top << ")\n";
    for (int i = 0; i < static_cast<int>(shots.size()) && i < top; ++i) {
        const Shot& s = shots[i];
        std::string obj = s.objectIdx >= 0 && s.objectIdx < static_cast<int>(balls.size())
                              ? std::to_string(balls[s.objectIdx].id)
                              : "?";
        std::printf("  #%d %-6s obj=%-3s pocket=%d cut=%5.1f deg conf=%.3f",
                    i + 1, s.typeName.c_str(), obj.c_str(), s.pocketIdx + 1,
                    s.cutDeg, s.confidence);
        if (s.typeName == "kick") std::printf("  [hit object, no pocket]\n");
        else std::printf("\n");
    }

    // ----- annotated image -------------------------------------------------
    Image ann = img;
    drawRect(ann, det.feltRect, {90, 190, 90}, 1);
    // Pocket markers.
    for (const auto& pk : tbl.pockets) {
        // Logical -> px mapping uses the felt rect.
        double pxX = det.feltRect.x0 + pk.pos.x / tbl.w * det.feltRect.w();
        double pxY = det.feltRect.y0 + pk.pos.y / tbl.h * det.feltRect.h();
        drawCircle(ann, {pxX, pxY}, tbl.pocketR * det.feltRect.w() / tbl.w,
                   {220, 90, 90}, 2);
    }
    // Balls + labels.
    for (const auto& d : det.balls) {
        drawCircle(ann, d.px, d.rPx + 2, {80, 220, 255}, 2);
        drawTextCentered(ann, d.px + Vec2{0, -d.rPx - 10}, labelFor(d),
                         {80, 220, 255}, 2);
    }
    // Top shots: draw the best last so it sits on top; label each shot at
    // its ghost position (distinct per shot) to avoid overlapping text.
    int nDraw = std::min<int>(top, static_cast<int>(shots.size()));
    for (int rank = nDraw - 1; rank >= 0; --rank) {
        const Shot& s = shots[rank];
        Rgb col = shotColor(rank);

        auto toPx = [&](Vec2 logical) {
            return Vec2{det.feltRect.x0 + logical.x / tbl.w * det.feltRect.w(),
                        det.feltRect.y0 + logical.y / tbl.h * det.feltRect.h()};
        };

        // Cue path (dashed) + ghost circle.
        for (size_t k = 0; k + 1 < s.cuePath.size(); ++k)
            drawLine(ann, toPx(s.cuePath[k]), toPx(s.cuePath[k + 1]), col, 2,
                     true, 8);
        if (s.typeName != "kick")
            drawCircle(ann, toPx(s.ghost), tbl.ballR * det.feltRect.w() / tbl.w,
                       col, 2);
        // Object path (solid) + bounce markers.
        for (size_t k = 0; k + 1 < s.objPath.size(); ++k)
            drawLine(ann, toPx(s.objPath[k]), toPx(s.objPath[k + 1]), col, 2);
        for (size_t k = 1; k + 1 < s.objPath.size(); ++k) {
            Vec2 p = toPx(s.objPath[k]);
            drawCircle(ann, p, 4, col, 1, true);
        }
        // 90-degree tangent hint at the ghost (direct shots only).
        if (s.typeName == "direct") {
            Vec2 g = toPx(s.ghost);
            Vec2 tdir = s.tangentDir;
            double scale = tbl.ballR * det.feltRect.w() / tbl.w * 2.2;
            drawLine(ann, g - tdir * scale, g + tdir * scale, {150, 150, 160}, 1,
                     true, 5);
        }
        // Rank + stats label near the ghost (direct/bank) or near the first
        // bounce (kick), offset downward per rank to avoid collisions.
        Vec2 labelAt;
        if (s.typeName == "kick" && s.cuePath.size() >= 2)
            labelAt = toPx((s.cuePath[0] + s.cuePath[1]) / 2.0);
        else
            labelAt = toPx(s.ghost) + Vec2{0, tbl.ballR * det.feltRect.w() / tbl.w + 6};
        char buf[64];
        std::snprintf(buf, sizeof(buf), "#%d %s CUT=%.0f CONF=%.2f",
                      rank + 1, s.typeName.c_str(),
                      static_cast<double>(s.cutDeg), s.confidence);
        drawTextCentered(ann, labelAt + Vec2{0.0, 8.0 + 12.0 * rank}, buf, col, 2);
    }
    std::string annPath = (fs::path(outDir) / (baseName + "_annotated.png")).string();
    if (!saveImage(annPath, ann)) {
        std::fprintf(stderr, "failed to save %s\n", annPath.c_str());
        return 1;
    }

    // ----- result JSON -----------------------------------------------------
    JsonWriter j;
    j.beginObj()
        .kv("tool", "pool-vision-trainer")
        .kv("version", "1.0.0")
        .beginObj("table")
            .kvNum("w", tbl.w, 1)
            .kvNum("h", tbl.h, 1)
            .kvNum("ballR", tbl.ballR, 1)
            .kvNum("pocketR", tbl.pocketR, 1)
        .endObj()
        .beginObj("detection")
            .kvNum("felt_rect_x0", det.feltRect.x0, 1)
            .kvNum("felt_rect_y0", det.feltRect.y0, 1)
            .kvNum("felt_rect_x1", det.feltRect.x1, 1)
            .kvNum("felt_rect_y1", det.feltRect.y1, 1)
            .kvNum("ball_radius_px", det.ballRPx, 2)
        .endObj();
    j.beginArr("balls");
    for (const auto& d : det.balls) {
        j.beginObj()
            .kv("kind", d.kind)
            .kv("color", d.color)
            .kvInt("number", d.number)
            .kvRaw("logical", jsonVec2(d.logical.x, d.logical.y))
            .kvRaw("px", jsonVec2(d.px.x, d.px.y))
            .kvNum("radius_px", d.rPx, 2)
            .kvNum("white_ratio", d.whiteRatio, 3)
            .kvNum("sat", d.sat, 3)
            .kvNum("val", d.val, 3)
            .kvNum("hue", d.hue, 1)
        .endObj();
    }
    j.endArr();
    j.beginArr("shots");
    for (const auto& s : shots) {
        j.beginObj()
            .kv("type", s.typeName)
            .kvInt("object", s.objectIdx >= 0 && s.objectIdx < static_cast<int>(balls.size())
                                 ? balls[s.objectIdx].id : -1)
            .kvInt("pocket", s.pocketIdx + 1)
            .kvNum("cut_deg", s.cutDeg, 1)
            .kvNum("confidence", s.confidence, 3)
            .kvBool("feasible", s.feasible)
            .kvRaw("ghost", jsonVec2(s.ghost.x, s.ghost.y))
            .kvRaw("cue_path", jsonVec2List(s.cuePath))
            .kvRaw("object_path", jsonVec2List(s.objPath))
            .kvRaw("tangent_dir", jsonVec2(s.tangentDir.x, s.tangentDir.y))
        .endObj();
    }
    j.endArr();
    // Ground truth comparison (demo mode).
    if (groundTruth) {
        int matched = 0;
        j.beginObj("ground_truth");
        j.beginArr("balls");
        for (const auto& gb : groundTruth->balls) {
            double best = 1e30;
            const Detection* bd = nullptr;
            for (const auto& d : det.balls) {
                double dist = d.logical.dist(gb.pos);
                if (dist < best) { best = dist; bd = &d; }
            }
            bool ok = bd && best < tbl.ballR * 1.5 && bd->number == gb.number;
            if (ok) ++matched;
            j.beginObj()
                .kvInt("number", gb.number)
                .kv("kind", gb.kind)
                .kvRaw("pos", jsonVec2(gb.pos.x, gb.pos.y))
                .kvBool("detected", ok)
                .kvNum("error_px_logical", bd ? best : -1.0, 2)
            .endObj();
        }
        j.endArr();
        j.kvInt("matched", matched);
        j.kvInt("total", static_cast<long long>(groundTruth->balls.size()));
        j.endObj();
        std::printf("\nground truth: %d/%d balls matched\n", matched,
                    static_cast<int>(groundTruth->balls.size()));
    }
    j.endObj();

    std::string jsonPath = (fs::path(outDir) / (baseName + "_result.json")).string();
    FILE* f = std::fopen(jsonPath.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "failed to write %s\n", jsonPath.c_str());
        return 1;
    }
    std::fputs(j.str().c_str(), f);
    std::fputs("\n", f);
    std::fclose(f);
    std::printf("wrote %s\n", annPath.c_str());
    std::printf("wrote %s\n", jsonPath.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parseArgs(argc, argv);

    if (a.cmd == "version") {
        std::printf("pool-vision-trainer 1.0.0\n");
        return 0;
    }

    if (a.cmd == "demo") {
        SynthScene scene = demoScene();
        Image img = renderSynth(scene, a.seed);
        std::string tablePath = (fs::path(a.out) / "table.png").string();
        fs::create_directories(a.out);
        if (!saveImage(tablePath, img)) {
            std::fprintf(stderr, "failed to save %s\n", tablePath.c_str());
            return 1;
        }
        std::printf("wrote %s\n", tablePath.c_str());
        TableSpec table = scene.table;
        table.pockets = TableSpec::standard().pockets;
        return runPipeline(img, table, a.out, "table", a.maxRails, a.top, &scene);
    }

    if (a.cmd == "analyze") {
        if (a.in.empty()) {
            std::fprintf(stderr, "usage: pool-trainer analyze <image> [options]\n");
            return 2;
        }
        Image img;
        try {
            img = loadImage(a.in);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
        TableSpec table = TableSpec::standard();
        table.w = a.tableW;
        table.h = a.tableH;
        table.pockets = TableSpec::standard().pockets;
        fs::path p(a.in);
        std::string base = p.stem().string();
        return runPipeline(img, table, a.out, base, a.maxRails, a.top);
    }

    std::fprintf(stderr,
                 "usage:\n"
                 "  pool-trainer demo [--out DIR] [--seed N] [--max-rails N]\n"
                 "  pool-trainer analyze <image> [--out DIR] [--table WxH]"
                 " [--max-rails N] [--top N]\n"
                 "  pool-trainer version\n");
    return a.cmd.empty() ? (a.in.empty() ? 0 : 2) : 2;
}
