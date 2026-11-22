// jsonw.cpp — JSON writer implementation.
#include "pool/jsonw.hpp"
#include <cmath>
#include <cstdio>

namespace pool {

std::string jsonEscape(const std::string& s) {
    std::string out;
    for (char ch : s) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                    out += buf;
                } else {
                    out += ch;
                }
        }
    }
    return out;
}

std::string jsonNum(double v, int prec) {
    if (!std::isfinite(v)) return "null";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return buf;
}

std::string jsonVec2(double x, double y, int prec) {
    return "[" + jsonNum(x, prec) + "," + jsonNum(y, prec) + "]";
}

std::string jsonVec2List(const std::vector<Vec2>& pts, int prec) {
    std::string out = "[";
    for (size_t i = 0; i < pts.size(); ++i) {
        if (i) out += ",";
        out += jsonVec2(pts[i].x, pts[i].y, prec);
    }
    out += "]";
    return out;
}

void JsonWriter::comma_() {
    if (needComma_.empty()) return;
    if (needComma_.back()) os_ << ",";
    needComma_.back() = true;
}

void JsonWriter::key_(const std::string& key) {
    if (!key.empty()) os_ << "\"" << jsonEscape(key) << "\":";
}

JsonWriter& JsonWriter::beginObj(const std::string& key) {
    comma_();
    key_(key);
    os_ << "{";
    isObj_.push_back(true);
    needComma_.push_back(false);
    return *this;
}

JsonWriter& JsonWriter::endObj() {
    os_ << "}";
    isObj_.pop_back();
    needComma_.pop_back();
    return *this;
}

JsonWriter& JsonWriter::beginArr(const std::string& key) {
    comma_();
    key_(key);
    os_ << "[";
    isObj_.push_back(false);
    needComma_.push_back(false);
    return *this;
}

JsonWriter& JsonWriter::endArr() {
    os_ << "]";
    isObj_.pop_back();
    needComma_.pop_back();
    return *this;
}

JsonWriter& JsonWriter::kv(const std::string& key, const std::string& value) {
    comma_();
    key_(key);
    os_ << "\"" << jsonEscape(value) << "\"";
    return *this;
}

JsonWriter& JsonWriter::kv(const std::string& key, const char* value) {
    return kv(key, std::string(value));
}

JsonWriter& JsonWriter::kvNum(const std::string& key, double value, int prec) {
    comma_();
    key_(key);
    os_ << jsonNum(value, prec);
    return *this;
}

JsonWriter& JsonWriter::kvInt(const std::string& key, long long value) {
    comma_();
    key_(key);
    os_ << value;
    return *this;
}

JsonWriter& JsonWriter::kvBool(const std::string& key, bool value) {
    comma_();
    key_(key);
    os_ << (value ? "true" : "false");
    return *this;
}

JsonWriter& JsonWriter::kvRaw(const std::string& key, const std::string& raw) {
    comma_();
    key_(key);
    os_ << raw;
    return *this;
}

}  // namespace pool
