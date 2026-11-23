// menu_ui.hpp — pure ImGui-2019-style settings menu model.
// No platform headers: geometry, hit-testing, value mutation and ini text
// are plain C++ so they are unit-testable anywhere; the Win32 renderer in
// pool_overlay.cpp only draws the rects this model computes and feeds mouse
// events in. Layout mirrors the reference "Settings" window: title bar with
// a close X, left tab column, right content rows (checkbox / slider / combo
// / button) with the label to the right of each control.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace mui {

enum CtrlType { CT_Check, CT_Slider, CT_Combo, CT_Button, CT_Label };

struct Ctrl {
    CtrlType type = CT_Label;
    const char* label = "";
    bool* bval = nullptr;                 // CT_Check
    float* fval = nullptr;                // CT_Slider
    int* ival = nullptr;                  // CT_Combo index / CT_Button id
    float fmin = 0, fmax = 1;
    const char* const* options = nullptr; // CT_Combo
    int nopt = 0;
    const char* fmt = "%.2f";
};

struct Tab {
    const char* name;
    std::vector<Ctrl> ctrls;
};

struct Metrics {
    float w = 640;
    float titleH = 30, tabW = 116, rowH = 27, pad = 12;
    float tabRowH = 30;
    float contentX(float px) const { return px + tabW + 8; }
    float contentY(float py) const { return py + titleH + pad; }
    float contentW() const { return w - tabW - 2 * pad - 4; }
    float panelH(const Tab& t) const {
        float rows = static_cast<float>(t.ctrls.size());
        float h = titleH + pad + rows * rowH + pad;
        float minH = titleH + pad + 6 * tabRowH + pad;  // tab column height
        return h > minH ? h : minH;
    }
};

struct State {
    int activeTab = 0;
    bool open = true;
    float px = 16, py = 36;  // panel top-left within the overlay window
    bool draggingPanel = false;
    float dragOffX = 0, dragOffY = 0;
    bool dragSlider = false;
    int dragTab = -1, dragCtrl = -1;
    int comboTab = -1, comboCtrl = -1;  // open dropdown
};

struct Rect {
    float x = 0, y = 0, w = 0, h = 0;
    bool contains(float px, float py) const {
        return px >= x && px <= x + w && py >= y && py <= y + h;
    }
};

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline Rect panelRect(const Metrics& m, const State& s, const Tab& t) {
    return {s.px, s.py, m.w, m.panelH(t)};
}
inline Rect titleRect(const Metrics& m, const State& s) {
    return {s.px, s.py, m.w, m.titleH};
}
inline Rect closeRect(const Metrics& m, const State& s) {
    return {s.px + m.w - 34, s.py + 2, 32, m.titleH - 4};
}
inline Rect tabRect(const Metrics& m, const State& s, int i) {
    return {s.px + 8, s.py + m.titleH + 6 + i * m.tabRowH, m.tabW - 16,
            m.tabRowH - 4};
}
inline Rect rowRect(const Metrics& m, const State& s, int row) {
    return {m.contentX(s.px), m.contentY(s.py) + row * m.rowH, m.contentW(),
            m.rowH};
}
// Track (the interactive bar) occupies ~72% of the row; label sits to its
// right, matching the reference layout.
inline Rect trackRect(const Metrics& m, const State& s, int row) {
    Rect r = rowRect(m, s, row);
    return {r.x, r.y + 4, r.w * 0.72f, r.h - 8};
}
// Dropdown popup for an open combo: listed right under its row.
inline Rect popRect(const Metrics& m, const State& s, int row, int nopt) {
    Rect r = rowRect(m, s, row);
    return {r.x + 4, r.y + r.h, r.w * 0.72f - 8, nopt * m.rowH};
}

enum Hit { H_None, H_Title, H_Close, H_Tab, H_Ctrl, H_Option, H_Panel };

struct HitResult {
    Hit hit = H_None;
    int tab = -1, ctrl = -1, option = -1;
    Rect rect;
};

inline HitResult hitTest(const Metrics& m, const State& s,
                         const std::vector<Tab>& tabs, float mx, float my) {
    HitResult hr;
    if (s.activeTab < 0 || s.activeTab >= static_cast<int>(tabs.size()))
        return hr;
    const Tab& t = tabs[static_cast<size_t>(s.activeTab)];
    // Open dropdown option first (it may overflow the panel).
    if (s.comboTab == s.activeTab && s.comboCtrl >= 0 &&
        s.comboCtrl < static_cast<int>(t.ctrls.size())) {
        const Ctrl& c = t.ctrls[static_cast<size_t>(s.comboCtrl)];
        Rect pr = popRect(m, s, s.comboCtrl, c.nopt);
        if (pr.contains(mx, my)) {
            hr.hit = H_Option;
            hr.tab = s.comboTab;
            hr.ctrl = s.comboCtrl;
            hr.option = static_cast<int>((my - pr.y) / m.rowH);
            hr.rect = pr;
            return hr;
        }
    }
    if (closeRect(m, s).contains(mx, my)) {
        hr.hit = H_Close;
        return hr;
    }
    if (titleRect(m, s).contains(mx, my)) {
        hr.hit = H_Title;
        return hr;
    }
    for (int i = 0; i < static_cast<int>(tabs.size()); ++i)
        if (tabRect(m, s, i).contains(mx, my)) {
            hr.hit = H_Tab;
            hr.tab = i;
            return hr;
        }
    for (int i = 0; i < static_cast<int>(t.ctrls.size()); ++i) {
        Rect r = rowRect(m, s, i);
        if (r.contains(mx, my)) {
            hr.hit = H_Ctrl;
            hr.ctrl = i;
            hr.rect = r;
            return hr;
        }
    }
    if (panelRect(m, s, t).contains(mx, my)) hr.hit = H_Panel;
    return hr;
}

// Mouse press: toggles checks, fires buttons, picks options, starts panel /
// slider drags. Returns true when the click is consumed by the menu.
inline bool press(const Metrics& m, State& s, std::vector<Tab>& tabs,
                  float mx, float my) {
    if (s.activeTab < 0 || s.activeTab >= static_cast<int>(tabs.size()))
        return false;
    Tab& t = tabs[static_cast<size_t>(s.activeTab)];
    HitResult hr = hitTest(m, s, tabs, mx, my);
    switch (hr.hit) {
        case H_Close:
            s.open = false;
            return true;
        case H_Title:
            s.draggingPanel = true;
            s.dragOffX = mx - s.px;
            s.dragOffY = my - s.py;
            return true;
        case H_Tab:
            s.activeTab = hr.tab;
            s.comboTab = s.comboCtrl = -1;
            return true;
        case H_Option: {
            if (hr.tab >= 0 && hr.ctrl >= 0) {
                Ctrl& c = tabs[static_cast<size_t>(hr.tab)]
                              .ctrls[static_cast<size_t>(hr.ctrl)];
                if (c.ival && hr.option >= 0 && hr.option < c.nopt)
                    *c.ival = hr.option;
                s.comboTab = s.comboCtrl = -1;
            }
            return true;
        }
        case H_Ctrl: {
            if (hr.ctrl < 0 ||
                hr.ctrl >= static_cast<int>(t.ctrls.size()))
                return true;
            Ctrl& c = t.ctrls[static_cast<size_t>(hr.ctrl)];
            s.comboTab = s.comboCtrl = -1;
            if (c.type == CT_Check && c.bval) {
                *c.bval = !*c.bval;
                return true;
            }
            if (c.type == CT_Button) return true;  // value read via ival poll
            if (c.type == CT_Combo && c.ival) {
                s.comboTab = s.activeTab;
                s.comboCtrl = hr.ctrl;
                return true;
            }
            if (c.type == CT_Slider && c.fval) {
                Rect tr = trackRect(m, s, hr.ctrl);
                float frac = clampf((mx - tr.x) / tr.w, 0.f, 1.f);
                *c.fval = c.fmin + frac * (c.fmax - c.fmin);
                s.dragSlider = true;
                s.dragTab = s.activeTab;
                s.dragCtrl = hr.ctrl;
                return true;
            }
            return true;  // labels swallow clicks inside the panel
        }
        case H_Panel:
            s.comboTab = s.comboCtrl = -1;
            return true;
        default:
            return false;  // outside the panel: let the game have it
    }
}

inline void release(State& s) {
    s.draggingPanel = false;
    s.dragSlider = false;
    s.dragTab = s.dragCtrl = -1;
}

// Mouse move with the button held: drags the panel or retargets sliders.
inline bool drag(const Metrics& m, State& s, std::vector<Tab>& tabs,
                 float mx, float my) {
    if (s.draggingPanel) {
        s.px = mx - s.dragOffX;
        s.py = my - s.dragOffY;
        return true;
    }
    if (s.dragSlider && s.dragTab >= 0 && s.dragTab < (int)tabs.size() &&
        s.dragCtrl >= 0) {
        Tab& t = tabs[static_cast<size_t>(s.dragTab)];
        if (s.dragCtrl < static_cast<int>(t.ctrls.size())) {
            Ctrl& c = t.ctrls[static_cast<size_t>(s.dragCtrl)];
            if (c.fval) {
                Rect tr = trackRect(m, s, s.dragCtrl);
                float frac = clampf((mx - tr.x) / tr.w, 0.f, 1.f);
                *c.fval = c.fmin + frac * (c.fmax - c.fmin);
                return true;
            }
        }
    }
    return false;
}

// ---- tiny ini text (k=v lines, # comments) ----
struct IniEntry { std::string key, val; };

inline std::string iniSerialize(const std::vector<IniEntry>& es) {
    std::string out;
    for (const auto& e : es)
        out += e.key + "=" + e.val + "\n";
    return out;
}

inline std::vector<IniEntry> iniParse(const std::string& text) {
    std::vector<IniEntry> es;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t eol = text.find('\n', pos);
        std::string line =
            text.substr(pos, eol == std::string::npos ? std::string::npos
                                                      : eol - pos);
        pos = eol == std::string::npos ? text.size() : eol + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        IniEntry e;
        e.key = line.substr(0, eq);
        e.val = line.substr(eq + 1);
        while (!e.key.empty() && e.key.back() == ' ') e.key.pop_back();
        es.push_back(e);
    }
    return es;
}

}  // namespace mui
