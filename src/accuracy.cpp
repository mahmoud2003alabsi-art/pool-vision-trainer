// accuracy.cpp — measured-physics bridge + success probability.
#include "pool/accuracy.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace pool {
namespace accuracy {

double metersPerUnit() { return 0.0254 * 100.0 / 900.0; }  // 2.8222 mm

double unitsPerMeter() { return 1.0 / metersPerUnit(); }

double rollingDecel(double muRolling) {
    return muRolling * 9.81 / metersPerUnit();
}

double slidingDecel(double muSliding) {
    return muSliding * 9.81 / metersPerUnit();
}

double caromAngleDeg(double cutDeg) {
    // theta = atan( sin(phi)cos(phi) / (sin^2(phi) + 2/5) )
    double phi = clampd(cutDeg, 0.0, 90.0) * 3.14159265358979323846 / 180.0;
    double s = std::sin(phi), c = std::cos(phi);
    return std::atan((s * c) / (s * s + 0.4)) * 180.0 /
           3.14159265358979323846;
}

double stunCaromAngleDeg(double cutDeg) {
    return 90.0 - clampd(cutDeg, 0.0, 90.0);
}

Vec2 cueDepartureDir(const Vec2& cueDir, const Vec2& lineOfCenters) {
    Vec2 v0 = cueDir.len2() > 1e-12 ? cueDir.norm() : Vec2{1, 0};
    Vec2 n = lineOfCenters.len2() > 1e-12 ? lineOfCenters.norm() : Vec2{0, 1};
    // Stun component: perpendicular to the line of centers.
    Vec2 tang = v0 - n * v0.dot(n);
    // Rolling continuation: a naturally rolling cue ball keeps 2/5 of its
    // speed along the original path (Dr. Dave's 30-degree-rule derivation;
    // the (2/5) term is the same one that peaks the carom at ~33.7 deg).
    Vec2 roll = v0 * 0.4;
    Vec2 out = tang + roll;
    if (out.len2() < 1e-12) return v0;  // degenerate: keep the path
    return out.norm();
}

double aimToleranceDeg(double targetUnits, double distanceUnits) {
    if (targetUnits <= 0 || distanceUnits <= 0) return 0;
    return std::atan(targetUnits / distanceUnits) * 180.0 /
           3.14159265358979323846;
}

namespace {

// Ball radius in mm: 2.25 in balls => 57.15 mm diameter.
double ballRadiusMm() { return 28.575; }

struct Sample {
    double aimDeg, power, spinF, spinS;
};

// One perturbed execution of the shot under the noise model.
Sample perturb(const Vec2& aimDir, double power, double spinF, double spinS,
               const NoiseModel& nm, std::mt19937& rng) {
    Sample s;
    std::normal_distribution<double> g01(0.0, 1.0);
    double sigmaU = nm.speedSigma / metersPerUnit();  // units/s
    s.aimDeg = g01(rng) * nm.aimSigmaDeg;
    double speed = PhysicsParams{}.maxShotSpeed *
                   clampd(power, 0.05, 1.0);
    speed += g01(rng) * sigmaU;
    s.power = clampd(speed / PhysicsParams{}.maxShotSpeed, 0.05, 1.0);
    double tipSigma = nm.tipSigmaMm / ballRadiusMm();  // ball-radius units
    s.spinF = clampd(spinF + g01(rng) * tipSigma, -1.0, 1.0);
    s.spinS = clampd(spinS + g01(rng) * tipSigma, -1.0, 1.0);
    return s;
}

}  // namespace

double shotSuccessProbability(const TableSpec& t,
                              const std::vector<Ball>& balls, int cueIdx,
                              const Vec2& aimDir, double power, double spinF,
                              double spinS, int targetId, const NoiseModel& nm,
                              const PhysicsParams& p) {
    if (cueIdx < 0 || cueIdx >= static_cast<int>(balls.size())) return 0.0;
    if (targetId < 0) return 0.0;
    Vec2 dir = aimDir.len2() > 1e-12 ? aimDir.norm() : Vec2{1, 0};

    std::mt19937 rng(nm.seed);
    int ok = 0, total = std::max(1, nm.samples);
    for (int k = 0; k < total; ++k) {
        Sample s = perturb(dir, power, spinF, spinS, nm, rng);
        double rad = s.aimDeg * 3.14159265358979323846 / 180.0;
        Vec2 d{dir.x * std::cos(rad) - dir.y * std::sin(rad),
               dir.x * std::sin(rad) + dir.y * std::cos(rad)};
        SimResult r = simulate(t, balls,
                               d * (PhysicsParams{}.maxShotSpeed * s.power),
                               s.spinF, s.spinS, p);
        bool potted = std::find(r.potted.begin(), r.potted.end(), targetId) !=
                      r.potted.end();
        if (potted && !r.cueScratch) ++ok;
    }
    return static_cast<double>(ok) / static_cast<double>(total);
}

}  // namespace accuracy
}  // namespace pool
