// test_menu.cpp — unit tests for the pure settings-menu model
// (overlay/menu_ui.hpp): layout bounds, hit-testing, checkbox toggles,
// slider press/drag math, combo dropdown, panel drag, close button and the
// tiny ini text round trip.
#include "menu_ui.hpp"

#include <cmath>
#include <cstdio>

static int gFail = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
            ++gFail;                                                         \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                \
    do {                                                                     \
        if (std::fabs((a) - (b)) > (eps)) {                                  \
            std::printf("FAIL %s:%d  %.4f != %.4f\n", __FILE__, __LINE__,    \
                        static_cast<double>(a), static_cast<double>(b));     \
            ++gFail;                                                         \
        }                                                                    \
    } while (0)

using namespace mui;

static bool showGuides = true, aimPreview = true, stripLines = true,
            manualRect = false;
static float ballR = 0, tolS = 0.20f, tolV = 0.30f, rx0 = 0, ry0 = 0, rx1 = 1,
             ry1 = 1, fps = 12;
static float rails = 2, shots = 3;  // sliders bind float mirrors
static int winIdx = 0, preset = 0, saveClicks = 0;

static const char* kWindows[] = {"Auto-detect", "8 Ball Pool", "Pool Live Tour"};
static const char* kPresets[] = {"8 Ball Pool 900x530", "Generic 2:1",
                                 "Compact 800x400"};

static std::vector<Tab> buildTabs() {
    std::vector<Tab> tabs;
    tabs.push_back({"AIM",
                    {{CT_Check, "Show guides", &showGuides},
                     {CT_Check, "Aim preview", &aimPreview},
                     {CT_Slider, "Rails", nullptr, &rails, nullptr, 0.f, 3.f},
                     {CT_Slider, "Shots", nullptr, &shots, nullptr, 1.f, 5.f}}});
    tabs.push_back({"DETECT",
                    {{CT_Check, "Strip aim line", &stripLines},
                     {CT_Slider, "Ball radius", nullptr, &ballR, nullptr, 0.f,
                      30.f},
                     {CT_Slider, "Felt tol S", nullptr, &tolS, nullptr, 0.05f,
                      0.6f},
                     {CT_Slider, "Felt tol V", nullptr, &tolV, nullptr, 0.05f,
                      0.6f}}});
    tabs.push_back({"TABLE",
                    {{CT_Combo, "Preset", nullptr, nullptr, &preset, 0, 0,
                      kPresets, 3},
                     {CT_Check, "Manual rect", &manualRect},
                     {CT_Slider, "Rect X0", nullptr, &rx0, nullptr, 0.f, 1.f},
                     {CT_Slider, "Rect Y0", nullptr, &ry0, nullptr, 0.f, 1.f},
                     {CT_Slider, "Rect X1", nullptr, &rx1, nullptr, 0.f, 1.f},
                     {CT_Slider, "Rect Y1", nullptr, &ry1, nullptr, 0.f, 1.f}}});
    tabs.push_back({"GAME",
                    {{CT_Combo, "Window", nullptr, nullptr, &winIdx, 0, 0,
                      kWindows, 3}}});
    tabs.push_back({"MISC",
                    {{CT_Slider, "FPS", nullptr, &fps, nullptr, 4.f, 30.f},
                     {CT_Button, "Save settings", nullptr, nullptr,
                      &saveClicks}}});
    return tabs;
}

int main() {
    std::vector<Tab> tabs = buildTabs();
    Metrics m;
    State s;

    // 1. Layout: every tab's rows fit inside the panel; tabs fit too.
    for (size_t ti = 0; ti < tabs.size(); ++ti) {
        Rect pr = panelRect(m, s, tabs[ti]);
        for (size_t i = 0; i < tabs[ti].ctrls.size(); ++i) {
            Rect r = rowRect(m, s, static_cast<int>(i));
            CHECK(r.x >= pr.x && r.x + r.w <= pr.x + pr.w + 0.5f);
            CHECK(r.y >= pr.y && r.y + r.h <= pr.y + pr.h + 0.5f);
        }
        Rect tr = tabRect(m, s, static_cast<int>(tabs.size()) - 1);
        CHECK(tr.y + tr.h <= pr.y + pr.h + 0.5f);
    }

    // 2. Checkbox toggle via click on the row.
    showGuides = true;
    CHECK(press(m, s, tabs, rowRect(m, s, 0).x + 30,
                rowRect(m, s, 0).y + 13));
    CHECK(!showGuides);
    press(m, s, tabs, rowRect(m, s, 0).x + 30, rowRect(m, s, 0).y + 13);
    CHECK(showGuides);

    // 3. Slider press sets the value from the click fraction.
    s.activeTab = 1;
    Rect tr = trackRect(m, s, 1);  // ball radius row
    ballR = -1;
    press(m, s, tabs, tr.x + tr.w * 0.25f, tr.y + tr.h / 2);
    CHECK_NEAR(ballR, 0.25f * 30.f, 1e-3);
    // Drag continues to move the value.
    drag(m, s, tabs, tr.x + tr.w * 0.75f, tr.y);
    release(s);
    CHECK_NEAR(ballR, 0.75f * 30.f, 1e-3);
    // After release the drag no longer moves it.
    drag(m, s, tabs, tr.x, tr.y);
    CHECK_NEAR(ballR, 0.75f * 30.f, 1e-3);

    // 4. Combo opens a dropdown; picking an option applies and closes it.
    s.activeTab = 3;
    winIdx = -1;
    Rect wr = rowRect(m, s, 0);
    CHECK(press(m, s, tabs, wr.x + 40, wr.y + 13));
    Rect pr = popRect(m, s, 0, 3);
    CHECK(press(m, s, tabs, pr.x + 20, pr.y + 2 * 27 + 10));  // option 2
    CHECK(winIdx == 2);
    // popup closed: a click at the same spot now lands on the panel itself
    // (swallowed) and must not change the selection
    press(m, s, tabs, pr.x + 20, pr.y + 2 * 27 + 10);
    CHECK(winIdx == 2);

    // 5. Close button hides the menu.
    Rect cr = closeRect(m, s);
    CHECK(press(m, s, tabs, cr.x + 8, cr.y + 8));
    CHECK(!s.open);
    s.open = true;

    // 6. Tab switch.
    s.activeTab = 0;
    Rect t2 = tabRect(m, s, 2);
    press(m, s, tabs, t2.x + 10, t2.y + 10);
    CHECK(s.activeTab == 2);

    // 7. Title-bar drag moves the panel.
    Rect tr2 = titleRect(m, s);
    float oldPx = s.px, oldPy = s.py;
    press(m, s, tabs, tr2.x + 40, tr2.y + 10);
    drag(m, s, tabs, oldPx + 90, oldPy + 60);
    release(s);
    CHECK_NEAR(s.px, oldPx + 50, 1e-3);
    CHECK_NEAR(s.py, oldPy + 50, 1e-3);

    // 8. Click outside the panel is not consumed.
    Rect prNow = panelRect(m, s, tabs[static_cast<size_t>(s.activeTab)]);
    CHECK(!press(m, s, tabs, prNow.x + prNow.w + 100,
                 prNow.y + prNow.h + 100));

    // 9. Manual-rect checkbox in TABLE tab toggles from a fresh scan.
    s.activeTab = 2;
    manualRect = false;
    press(m, s, tabs, rowRect(m, s, 1).x + 20, rowRect(m, s, 1).y + 13);
    CHECK(manualRect);

    // 10. ini round trip through the tiny serializer/parser.
    std::vector<IniEntry> es = {
        {"tableW", "900"}, {"feltTolV", "0.30"}, {"manualRect", "1"},
        {"window", "8 Ball Pool"}};
    std::string text = iniSerialize(es);
    text = "# comment line\n" + text + "\nbroken line without equals\n";
    auto back = iniParse(text);
    CHECK(back.size() == es.size());
    bool ok = true;
    for (size_t i = 0; i < es.size() && i < back.size(); ++i)
        ok = ok && back[i].key == es[i].key && back[i].val == es[i].val;
    CHECK(ok);

    // 11. Slider clamping stays inside [fmin, fmax].
    s.activeTab = 1;
    tr = trackRect(m, s, 2);  // tolS
    press(m, s, tabs, tr.x - 500, tr.y);  // far left
    CHECK(tolS >= 0.05f);
    drag(m, s, tabs, tr.x + tr.w + 500, tr.y);
    release(s);
    CHECK(tolS <= 0.6f);

    if (gFail == 0) {
        std::printf("ALL MENU TESTS PASSED\n");
        return 0;
    }
    std::printf("%d MENU TEST FAILURES\n", gFail);
    return 1;
}
