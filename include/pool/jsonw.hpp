// jsonw.hpp — tiny JSON writer with indentation and comma management.
#pragma once
#include "pool/geom.hpp"
#include <sstream>
#include <string>
#include <vector>

namespace pool {

class JsonWriter {
public:
    JsonWriter& beginObj(const std::string& key = "");
    JsonWriter& endObj();
    JsonWriter& beginArr(const std::string& key = "");
    JsonWriter& endArr();
    JsonWriter& kv(const std::string& key, const std::string& value);
    JsonWriter& kv(const std::string& key, const char* value);
    JsonWriter& kvNum(const std::string& key, double value, int prec = 3);
    JsonWriter& kvInt(const std::string& key, long long value);
    JsonWriter& kvBool(const std::string& key, bool value);
    // Pre-serialized JSON (e.g. a numbers array) under `key`.
    JsonWriter& kvRaw(const std::string& key, const std::string& raw);
    std::string str() const { return os_.str(); }

private:
    void comma_();
    void key_(const std::string& key);
    std::ostringstream os_;
    std::vector<bool> isObj_;
    std::vector<bool> needComma_;
};

std::string jsonEscape(const std::string& s);
std::string jsonVec2(double x, double y, int prec = 2);
std::string jsonVec2List(const std::vector<Vec2>& pts, int prec = 2);
std::string jsonNum(double v, int prec = 3);

}  // namespace pool
