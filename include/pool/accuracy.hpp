// accuracy.hpp — measured-physics bridge and success-probability layer.
//
// The solver/physics engine works in logical units (900x450 for a 9ft
// table). This layer converts published, measured billiards constants
// into those units, fixes the rolling cue ball departure direction (the
// 30-degree rule, not the naive 90-degree tangent) and turns a candidate
// shot into a success probability by Monte-Carlo over the standard
// execution-noise model from computational pool research.
//
// All constants come from published sources documented in RESEARCH.md:
//   - rolling friction coefficient 0.0127 (high-speed imaging, AJP 2009)
//   - sliding friction coefficient 0.21 (Mathavan et al. 2014)
//   - execution noise: aim 0.125 deg, speed 0.075 m/s, tip 0.5 mm
//     (Computational Pool, AI Magazine 2002 / Greenspan IEEE 2008)
//   - cue ball carom: theta = atan(sin(phi)cos(phi) / (sin^2(phi) + 2/5))
//     peak 33.7 deg at cut ~28 deg (Alciatore / pooltool 30-degree rule)
// Pure math, no dependencies, deterministic.
#pragma once
#include "pool/geom.hpp"
#include "pool/physics.hpp"
#include <string>
#include <vector>

namespace pool {
namespace accuracy {

// ---- unit bridge (logical units <-> SI) --------------------------------

// 9ft table: playing field 100 in wide mapped to 900 logical units
// => 1 unit = 2.8222 mm.
double metersPerUnit();
double unitsPerMeter();

// Rolling deceleration in logical units/s^2 from a friction coefficient
// (default mu = 0.0127, the measured rolling value). Reference: ~44.15.
double rollingDecel(double muRolling = 0.0127);

// Sliding deceleration in logical units/s^2 (default mu = 0.21, the
// measured ball-cloth sliding value). Reference: ~730.
double slidingDecel(double muSliding = 0.21);

// ---- cue ball departure direction --------------------------------------

// Rolling cue ball carom angle (30-degree rule) relative to the ORIGINAL
// cue path, for a cut angle phi in degrees. Peak ~33.7 deg at phi ~28 deg.
double caromAngleDeg(double cutDeg);

// Stun (stop-shot) cue departure angle relative to the original path:
// exactly along the tangent of the line of centers => 90 - cut.
double stunCaromAngleDeg(double cutDeg);

// Physical departure direction of a ROLLING cue ball after contact:
// stun component (perpendicular to the line of centers) + the 2/5 forward
// roll continuation along the original path. For a pure stun delivery use
// the tangential component alone (see stunCaromAngleDeg).
Vec2 cueDepartureDir(const Vec2& cueDir, const Vec2& lineOfCenters);

// ---- pocket aiming tolerance --------------------------------------------

// Full angular window (degrees) that an effective target of
// `targetUnits` diameter subtends at `distanceUnits`: the aim tolerance.
// Reference: a 3-in effective pocket mouth at 5 ft => ~2.86 deg.
double aimToleranceDeg(double targetUnits, double distanceUnits);

// ---- success probability -------------------------------------------------

// Standard execution-noise model from computational pool research.
struct NoiseModel {
    double aimSigmaDeg = 0.125;   // aim angle std, degrees
    double speedSigma = 0.075;    // cue speed std, m/s
    double tipSigmaMm = 0.5;      // tip contact offset std, mm
    int samples = 200;            // Monte-Carlo shots
    unsigned seed = 987654321u;   // deterministic RNG seed
};

// Probability (0..1) that playing the shot defined by `aimDir` (any
// length), `power` (0..1) and the cue contact point (`spinF`/`spinS`,
// ball-radius units) pots ball `targetId` WITHOUT scratching, under the
// execution noise model. Deterministic for a fixed seed on a given
// platform.
double shotSuccessProbability(const TableSpec& t,
                              const std::vector<Ball>& balls, int cueIdx,
                              const Vec2& aimDir, double power, double spinF,
                              double spinS, int targetId,
                              const NoiseModel& n = NoiseModel(),
                              const PhysicsParams& p = PhysicsParams());

}  // namespace accuracy
}  // namespace pool
