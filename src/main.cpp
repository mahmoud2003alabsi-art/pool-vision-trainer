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
//   pool-trainer simulate <image> [--out DIR] [--table WxH] [--power P]
//       [--contact DX,DY] [--dir DEG] [--shot N] [--max-rails N]
//       Detect balls, pick an aim (in-game guideline when visible, else the
//       best solved shot), run the full physics simulation with the given
//       power and cue contact point, and report the complete outcome:
//       every ball's path and rest position, potted balls, scratch warning
//       and the minimum power that pots. Writes <image>_sim_annotated.png
//       and <image>_sim_result.json.
//
//   pool-trainer tune <image> [--out DIR] [--table WxH] [--rect x0,y0,x1,y1]
//       [--ballr N] [--tols S] [--tolv V] [--no-strip] [--save]
//       Manual calibration on a screenshot: apply overrides, print what
//       detection finds, optionally --save them into pool-overlay.ini for
//       the live overlay to consume.
//
//   pool-trainer version
#include "pool/jsonw.hpp"
#include "pool/physics.hpp"
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
    double power = -1.0;      // <0 = auto (minimum that pots)
    double contactX = 0, contactY = 0;  // cue contact point, ball-radius units
    double dirDeg = -999;     // manual aim direction (image axes, y down)
    int shotIdx = 1;          // 1-based solver shot index for simulate
    // tune overrides
    std::string rectOverride;  // "x0,y0,x1,y1" pixels
    double ballR = 0;          // px, 0 = auto
    double tolS = -1, tolV = -1;
    bool noStrip = false;
    bool saveIni = false;
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
        else if (s == "--power") a.power = std::stod(need(s.c_str()));
        else if (s == "--contact") {
            std::string v = need(s.c_str());
            size_t c = v.find(',');
            if (c == std::string::npos) {
                std::fprintf(stderr, "--contact expects DX,DY in [-1,1] (e.g. 0,-0.6)\n");
                std::exit(2);
            }
            a.contactX = std::stod(v.substr(0, c));
            a.contactY = std::stod(v.substr(c + 1));
        } else if (s == "--dir") a.dirDeg = std::stod(need(s.c_str()));
        else if (s == "--shot") a.shotIdx = std::stoi(need(s.c_str()));
        else if (s == "--rect") a.rectOverride = need(s.c_str());
        else if (s == "--ballr") a.ballR = std::stod(need(s.c_str()));
        else if (s == "--tols") a.tolS = std::stod(need(s.c_str()));
        else if (s == "--tolv") a.tolV = std::stod(need(s.c_str()));
        else if (s == "--no-strip") a.noStrip = true;
        else if (s == "--save") a.saveIni = true;
        else if (s == "demo" || s == "analyze" || s == "version" ||
                 s == "simulate" || s == "tune") a.cmd = s;
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

// Full-shot simulation: detect, aim (in-game guideline / solver shot /
// manual), simulate with power + cue contact point, report and draw
// everything that will happen.
int runSimulate(const Image& img, const TableSpec& tableIn,
                const std::string& outDir, const std::string& baseName,
                const Args& a) {
    fs::create_directories(outDir);
    DetectOptions dopt;
    DetectResult det;
    try {
        det = detectBalls(img, dopt, tableIn);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "detection failed: %s\n", e.what());
        return 1;
    }
    const TableSpec tbl = det.table;

    std::vector<Ball> balls;
    int cueIdx = -1;
    std::vector<Vec2> otherPx;
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
    for (size_t i = 0; i < det.balls.size(); ++i)
        if (static_cast<int>(i) != cueIdx) otherPx.push_back(det.balls[i].px);
    if (cueIdx < 0) {
        std::fprintf(stderr, "no cue ball detected - cannot simulate\n");
        return 1;
    }

    // Pixel <-> logical mapping (identical to the detection shrink).
    Rect f = det.feltRect;
    f.x0 += 2; f.y0 += 2; f.x1 -= 2; f.y1 -= 2;
    auto toPx = [&](Vec2 l) {
        return Vec2{f.x0 + l.x / tbl.w * f.w(), f.y0 + l.y / tbl.h * f.h()};
    };
    auto toLogicalDir = [&](Vec2 pxDir) {
        Vec2 d{pxDir.x * tbl.w / f.w(), pxDir.y * tbl.h / f.h()};
        return d.len() > 1e-12 ? d.norm() : Vec2{1, 0};
    };

    // Choose the aim direction.
    Vec2 dirL{};
    std::string aimSource;
    int targetId = -1;
    bool haveDir = false;
    if (a.dirDeg > -900) {
        double rad = a.dirDeg * 3.14159265358979323846 / 180.0;
        dirL = toLogicalDir(Vec2{std::cos(rad), std::sin(rad)});
        aimSource = "manual";
        haveDir = true;
    } else if (estimateAimDirPx(img, det.balls[cueIdx].px, det.ballRPx, otherPx,
                                dirL)) {  // dirL reused as px buffer here
        dirL = toLogicalDir(dirL);
        aimSource = "in-game guideline";
        haveDir = true;
    }
    if (!haveDir) {
        Solver solver(tbl, balls, cueIdx, SolverOptions{a.maxRails, 80.0, 0.98});
        std::vector<Shot> shots = solver.allShots();
        if (shots.empty()) {
            std::fprintf(stderr, "no aim line visible and no feasible shot found\n");
            return 1;
        }
        size_t idx = static_cast<size_t>(std::max(1, a.shotIdx) - 1);
        if (idx >= shots.size()) idx = 0;
        const Shot& s = shots[idx];
        dirL = (s.ghost - balls[cueIdx].pos).norm();
        targetId = s.objectIdx >= 0 ? balls[s.objectIdx].id : -1;
        aimSource = "solver shot #" + std::to_string(a.shotIdx);
    }

    double spinF = clampd(a.contactY, -1.0, 1.0);  // +Y follow, -Y draw
    double spinS = clampd(a.contactX, -1.0, 1.0);  // X side english

    // Power: user value or the minimum that pots.
    double power = a.power;
    bool autoPower = power < 0;
    if (autoPower) {
        power = minPotPower(tbl, balls, dirL, spinF, spinS, targetId);
        if (power < 0) {
            std::printf("no pot possible along this aim even at 100%% power "
                        "- showing the full-power outcome anyway\n");
            power = 1.0;
        }
    }
    power = clampd(power, 0.05, 1.0);

    Vec2 cueVel = dirL * (PhysicsParams{}.maxShotSpeed * power);
    SimResult sim = simulate(tbl, balls, cueVel, spinF, spinS);

    // ----- console report ---------------------------------------------------
    constexpr double kPi = 3.14159265358979323846;
    double aimDeg = std::atan2(dirL.y, dirL.x) * 180.0 / kPi;
    std::string contactLbl =
        spinF > 0.25 ? "follow" : (spinF < -0.25 ? "draw" : "stun");
    if (spinS > 0.25) contactLbl += " + right english";
    else if (spinS < -0.25) contactLbl += " + left english";
    std::printf("\n===== SIMULATION =====\n");
    std::printf("aim: %.1f deg (%s) | power: %.2f%s | contact: (%.1f,%.1f) %s\n",
                aimDeg, aimSource.c_str(), power, autoPower ? " (auto-min)" : "",
                spinS, spinF, contactLbl.c_str());
    if (sim.firstContact >= 0)
        std::printf("first contact: ball %d\n", sim.firstContact);
    else
        std::printf("first contact: none - the aim misses every ball\n");
    if (sim.potted.empty()) {
        std::printf("potted: none\n");
    } else {
        std::printf("potted:");
        for (int id : sim.potted) {
            std::printf(" %d", id);
            for (const auto& e : sim.events)
                if (e.kind == "pot" && e.ball == id)
                    std::printf(" (pocket %d)", e.other + 1);
        }
        std::printf("\n");
    }
    if (sim.cueScratch)
        std::printf("WARNING: cue ball scratches - use draw/stun, softer power "
                    "or another aim\n");
    std::printf("cue rests at logical (%.0f,%.0f) after %.2f s\n",
                sim.final[cueIdx].pos.x, sim.final[cueIdx].pos.y, sim.duration);

    // ----- annotated image --------------------------------------------------
    Image ann = img;
    drawRect(ann, det.feltRect, {90, 190, 90}, 1);
    Vec2 cuePx = det.balls[cueIdx].px;
    Vec2 pd{dirL.x * f.w() / tbl.w, dirL.y * f.h() / tbl.h};
    pd = pd.len() > 1e-12 ? pd.norm() : Vec2{1, 0};
    double rayLen = std::hypot(f.w(), f.h());
    drawLine(ann, cuePx, cuePx + pd * rayLen, {255, 255, 255}, 1, true, 10);
    static const Rgb simPal[6] = {
        {255, 255, 255}, {255, 210, 40}, {60, 200, 255},
        {120, 255, 120}, {255, 120, 80}, {230, 120, 255}};
    for (size_t i = 0; i < sim.paths.size(); ++i) {
        const Rgb col = simPal[i % 6];
        for (size_t k = 0; k + 1 < sim.paths[i].size(); ++k)
            drawLine(ann, toPx(sim.paths[i][k]), toPx(sim.paths[i][k + 1]), col,
                     1, true, 7);
        const auto& fb = sim.final[i];
        if (fb.active)
            drawCircle(ann, toPx(fb.pos), balls[i].r * f.w() / tbl.w, col, 1,
                       true);
    }
    for (const auto& e : sim.events) {
        if (e.kind != "pot" || e.other < 0 ||
            e.other >= static_cast<int>(tbl.pockets.size()))
            continue;
        drawCircle(ann, toPx(tbl.pockets[e.other].pos), 6, {255, 255, 0}, 2);
    }
    std::string annPath =
        (fs::path(outDir) / (baseName + "_sim_annotated.png")).string();
    if (!saveImage(annPath, ann)) {
        std::fprintf(stderr, "failed to save %s\n", annPath.c_str());
        return 1;
    }

    // ----- result JSON ------------------------------------------------------
    std::string pottedStr;
    for (size_t i = 0; i < sim.potted.size(); ++i)
        pottedStr += (i ? "," : "") + std::to_string(sim.potted[i]);
    JsonWriter j;
    j.beginObj()
        .kv("tool", "pool-vision-trainer")
        .kv("kind", "simulation")
        .beginObj("aim")
            .kv("source", aimSource)
            .kvNum("deg", aimDeg, 1)
            .kvNum("power", power, 2)
            .kvNum("contact_x", spinS, 2)
            .kvNum("contact_y", spinF, 2)
        .endObj()
        .beginObj("outcome")
            .kvInt("first_contact", sim.firstContact)
            .kv("potted", pottedStr)
            .kvBool("cue_scratch", sim.cueScratch)
            .kvNum("cue_rest_x", sim.final[cueIdx].pos.x, 1)
            .kvNum("cue_rest_y", sim.final[cueIdx].pos.y, 1)
            .kvNum("duration_s", sim.duration, 2)
        .endObj()
        .beginArr("balls_final");
    for (size_t i = 0; i < sim.final.size(); ++i) {
        j.beginObj()
            .kvInt("id", balls[i].id)
            .kvNum("x", sim.final[i].pos.x, 1)
            .kvNum("y", sim.final[i].pos.y, 1)
            .kvBool("active", sim.final[i].active)
        .endObj();
    }
    j.endArr().beginArr("events");
    for (const auto& e : sim.events) {
        j.beginObj()
            .kvNum("t", e.t, 3)
            .kv("kind", e.kind)
            .kvInt("ball", e.ball)
            .kvInt("other", e.other)
            .kvRaw("pos", jsonVec2(e.pos.x, e.pos.y, 1))
        .endObj();
    }
    j.endArr().endObj();
    std::string jsonPath =
        (fs::path(outDir) / (baseName + "_sim_result.json")).string();
    FILE* fj = std::fopen(jsonPath.c_str(), "w");
    if (!fj) {
        std::fprintf(stderr, "failed to write %s\n", jsonPath.c_str());
        return 1;
    }
    std::fputs(j.str().c_str(), fj);
    std::fputs("\n", fj);
    std::fclose(fj);
    std::printf("wrote %s\n", annPath.c_str());
    std::printf("wrote %s\n", jsonPath.c_str());
    return 0;
}

}  // namespace


// Manual calibration ("edit anything"): apply overrides to the detection,
// report everything found and optionally persist them into pool-overlay.ini.
int runTune(const Image& img, const TableSpec& tableIn, const std::string& outDir,
            const std::string& baseName, const Args& a) {
    fs::create_directories(outDir);
    DetectOptions dopt;
    if (a.tolS >= 0) dopt.feltTolS = a.tolS;
    if (a.tolV >= 0) dopt.feltTolV = a.tolV;
    dopt.stripLines = !a.noStrip;
    if (a.ballR > 1) dopt.forceBallR = a.ballR;
    if (!a.rectOverride.empty()) {
        double v[4];
        int k = 0;
        size_t pos = 0;
        while (k < 4) {
            size_t comma = a.rectOverride.find(',', pos);
            v[k++] = std::stod(a.rectOverride.substr(
                pos, comma == std::string::npos ? std::string::npos : comma - pos));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        dopt.forceRect = Rect{v[0], v[1], v[2], v[3]};
    }
    DetectResult det;
    try {
        det = detectBalls(img, dopt, tableIn);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "detection failed: %s\n", e.what());
        return 1;
    }
    std::vector<Ball> balls;
    int cueIdx = -1;
    for (size_t i = 0; i < det.balls.size(); ++i) {
        const auto& d = det.balls[i];
        Ball b;
        b.id = d.number; b.kind = d.kind; b.color = d.color;
        b.pos = d.logical; b.r = det.table.ballR;
        if (d.kind == "cue" && cueIdx < 0) cueIdx = static_cast<int>(i);
        balls.push_back(b);
    }
    std::printf("rect px: (%.0f,%.0f)-(%.0f,%.0f)  ballR_px=%.1f  balls=%d\n",
                det.feltRect.x0, det.feltRect.y0, det.feltRect.x1, det.feltRect.y1,
                det.ballRPx, static_cast<int>(det.balls.size()));
    for (const auto& d : det.balls)
        std::printf("  %-7s %-12s px=(%.0f,%.0f) white=%.2f\n", d.kind.c_str(),
                    d.color.c_str(), d.px.x, d.px.y, d.whiteRatio);
    int shotCount = 0;
    if (cueIdx >= 0 && balls.size() >= 2) {
        Solver solver(det.table, balls, cueIdx, SolverOptions{a.maxRails, 80.0, 0.98});
        shotCount = static_cast<int>(solver.allShots().size());
    }
    std::printf("shots: %d\n", shotCount);

    // Persist everything the overlay consumes. Rect stored as fractions of
    // the image size so the live overlay can map it onto its own capture.
    if (a.saveIni) {
        double fx0 = det.feltRect.x0 / img.w, fy0 = det.feltRect.y0 / img.h;
        double fx1 = det.feltRect.x1 / img.w, fy1 = det.feltRect.y1 / img.h;
        std::string iniPath = (fs::path(a.out) / "pool-overlay.ini").string();
        FILE* f = std::fopen(iniPath.c_str(), "w");
        if (!f) {
            std::fprintf(stderr, "failed to write %s\n", iniPath.c_str());
            return 1;
        }
        std::fprintf(f,
                     "# pool-overlay.ini - generated by: pool-trainer tune %s --save\n"
                     "window=8 ball pool\n"
                     "tableW=%.0f\ntableH=%.0f\nrails=%d\nshots=%d\nfps=12\n"
                     "guides=1\naimPreview=1\n"
                     "ballR=%.1f\nfeltTolS=%.2f\nfeltTolV=%.2f\nstripLines=%d\n"
                     "manualRect=%d\nrectX0=%.4f\nrectY0=%.4f\nrectX1=%.4f\nrectY1=%.4f\n",
                     baseName.c_str(), a.tableW, a.tableH, a.maxRails, a.top,
                     a.ballR > 1 ? a.ballR : 0.0, dopt.feltTolS, dopt.feltTolV,
                     dopt.stripLines ? 1 : 0,
                     a.rectOverride.empty() ? 0 : 1, fx0, fy0, fx1, fy1);
        std::fclose(f);
        std::printf("wrote %s\n", iniPath.c_str());
        std::printf("copy it next to pool-overlay.exe (or pass --config)\n");
    }
    return 0;
}

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

    if (a.cmd == "simulate") {
        if (a.in.empty()) {
            std::fprintf(stderr, "usage: pool-trainer simulate <image> [options]\n");
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
        return runSimulate(img, table, a.out, base, a);
    }

    if (a.cmd == "tune") {
        if (a.in.empty()) {
            std::fprintf(stderr, "usage: pool-trainer tune <image> [options]\n");
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
        return runTune(img, table, a.out, base, a);
    }

    std::fprintf(stderr,
                 "usage:\n"
                 "  pool-trainer demo [--out DIR] [--seed N] [--max-rails N]\n"
                 "  pool-trainer analyze <image> [--out DIR] [--table WxH]"
                 " [--max-rails N] [--top N]\n"
                 "  pool-trainer simulate <image> [--out DIR] [--table WxH]"
                 " [--power P] [--contact DX,DY] [--dir DEG] [--shot N]"
                 " [--max-rails N]\n"
                 "  pool-trainer tune <image> [--rect x0,y0,x1,y1] [--ballr N]"
                 " [--tols S] [--tolv V] [--no-strip] [--save]\n"
                 "  pool-trainer version\n");
    return a.cmd.empty() ? (a.in.empty() ? 0 : 2) : 2;
}
