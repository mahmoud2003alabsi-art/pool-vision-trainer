// solver.hpp — shot solving: ghost-ball direct shots, bank and kick shots
// solved by the mirror/reflection (table-folding) method, path obstruction
// checks and shot scoring. Pure math, no dependencies.
//
// Rail indices:
//   0 = top (y = 0), 1 = right (x = w), 2 = bottom (y = h), 3 = left (x = 0)
#pragma once
#include "pool/geom.hpp"
#include <string>
#include <vector>

namespace pool {

struct Shot {
    enum class Type { Direct, Bank, Kick };
    Type type = Type::Direct;
    std::string typeName;      // "direct" | "bank" | "kick"
    int objectIdx = -1;        // index into the ball list
    int pocketIdx = -1;        // index into table.pockets (Kick: -1)
    double cutDeg = 0;         // cut angle at contact
    double confidence = 0;     // 0..1, higher = better
    bool feasible = true;
    std::string note;          // rejection reason when infeasible
    Vec2 ghost;                // cue ball center position at contact
    std::vector<Vec2> cuePath; // cue ball center path
    std::vector<Vec2> objPath; // object ball center path (empty for kicks)
    Vec2 tangentDir;           // cue departure direction after contact (90-deg rule)
};

struct SolverOptions {
    int maxRails = 2;          // max cushion bounces for bank/kick enumeration
    double maxCutDeg = 80.0;   // reject direct/bank cuts above this
    double clearance = 0.98;   // obstruction clearance multiplier (of 2R)
};

class Solver {
public:
    // `cueIdx` is the index of the cue ball in `balls` (-1 disables cue shots).
    Solver(const TableSpec& table, const std::vector<Ball>& balls, int cueIdx,
           const SolverOptions& opt = SolverOptions());

    // All feasible shots (direct + bank + kick), best confidence first.
    std::vector<Shot> allShots() const;

    // Individual solvers. On failure they return feasible=false with a note.
    Shot solveDirect(int objIdx, int pocketIdx) const;
    Shot solveBank(int objIdx, int pocketIdx, const std::vector<int>& rails) const;
    Shot solveKick(int objIdx, const std::vector<int>& rails) const;

private:
    TableSpec t_;
    std::vector<Ball> balls_;
    int cueIdx_;
    SolverOptions opt_;

    // True when no ball other than the two ignored ones is within ~2R of
    // the segment (a ball traveling along that segment would hit it).
    bool pathClear(const Vec2& a, const Vec2& b, int ignoreA, int ignoreB) const;

    // Bounce point must lie on the rail extent and not inside a pocket mouth.
    bool bounceOk(const Vec2& p, int rail) const;

    // Point must lie within the logical playing field.
    bool inField(const Vec2& p) const;

    static void railLine(const TableSpec& t, int rail, Vec2& lp, Vec2& ld);

    // Fold `P` across `rails` (last rail first) and return all folded images:
    // result[0] = P, result[N] = fully folded image.
    std::vector<Vec2> foldImages(const Vec2& P, const std::vector<int>& rails) const;

    // Walk the folded straight line forward through the rails, producing the
    // real-world bounce points. Returns false when any crossing is invalid.
    bool unfoldPath(const Vec2& from, const std::vector<Vec2>& imgs,
                    const std::vector<int>& rails, std::vector<Vec2>& pts,
                    std::string& err) const;
};

// Enumerate rail sequences (no consecutive repeats) up to maxRails.
std::vector<std::vector<int>> railSequences(int maxRails);

}  // namespace pool
