"""How many pixels tall is a tag character for a given camera, altitude and view?

The CONOPS gives the tag font as "85pt (~3cm) or greater". 85 pt is the font's
em size (85/72 in = 3.0 cm), not the height of a capital letter. The example
tags in CONOPS Figure 2 use a Times-style serif, whose capitals and lining
digits are about 0.66 em tall, so the smallest guaranteed character is about
2.0 cm tall. This is an interpretation of the font size, not an official rule.

Usage:
    python geometry.py                       # default table
    python geometry.py --width-px 1920 --hfov 80 --alt 10 20 30 40
"""
import argparse
import math

EM_M = 85 / 72 * 0.0254          # 85 pt em size in metres (~0.030 m)
CAP_HEIGHT_EM = 0.662            # Times New Roman capHeight / unitsPerEm (1356/2048)
CHAR_H_M = EM_M * CAP_HEIGHT_EM  # ~0.0199 m

# Pixel heights at which the synthetic experiment should be read (see README).
READ_THRESHOLDS = {"unreadable": 6, "marginal": 10, "reliable": 14}


def gsd_m_per_px(slant_range_m, hfov_deg, width_px):
    """Ground sample distance at the target, for a target facing the camera."""
    return 2 * slant_range_m * math.tan(math.radians(hfov_deg) / 2) / width_px


def char_px(slant_range_m, hfov_deg, width_px, view_angle_deg=0.0):
    """Character height in pixels.

    view_angle_deg is the angle between the camera ray and the tag face normal.
    Ear tags hang roughly vertically, so a nadir (straight-down) camera sees the
    tag almost edge-on (about 90 deg) and a 45 deg oblique camera sees it at
    about 45 deg. Foreshortening shrinks the visible height by cos(view_angle).
    """
    return CHAR_H_M * math.cos(math.radians(view_angle_deg)) / gsd_m_per_px(slant_range_m, hfov_deg, width_px)


def max_range_for(px_needed, hfov_deg, width_px, view_angle_deg=0.0):
    return CHAR_H_M * math.cos(math.radians(view_angle_deg)) * width_px / (2 * math.tan(math.radians(hfov_deg) / 2) * px_needed)


CAMERAS = [
    # name, horizontal pixels, horizontal FOV (deg). Illustrative options only.
    ("1080p wide (~80 deg)", 1920, 80),
    ("1080p normal (~50 deg)", 1920, 50),
    ("4K/12MP normal (~50 deg)", 4056, 50),
    ("1080p zoom (~15 deg)", 1920, 15),
    ("4K/12MP zoom (~15 deg)", 4056, 15),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width-px", type=int)
    ap.add_argument("--hfov", type=float)
    ap.add_argument("--alt", type=float, nargs="+", default=[5, 10, 20, 30, 50])
    ap.add_argument("--view", type=float, default=45.0,
                    help="angle between camera ray and tag face normal (deg); 45 = oblique camera")
    a = ap.parse_args()
    cams = [("custom", a.width_px, a.hfov)] if a.width_px and a.hfov else CAMERAS

    print(f"Character height assumed: {CHAR_H_M * 100:.2f} cm (85 pt em x {CAP_HEIGHT_EM} cap height)")
    print(f"View angle to tag face: {a.view:.0f} deg. Columns are altitude AGL; a vertical tag viewed at")
    print("this angle needs the camera tilted the same amount from nadir, so slant range = alt / cos(view).")
    print("With --view 0, read the columns as horizontal stand-off distance instead.\n")
    tilt = math.radians(a.view)
    head = f"{'camera':28s}" + "".join(f"{f'{h:g} m':>11s}" for h in a.alt)
    print(head + f"{'max range @14px':>17s}")
    for name, w, fov in cams:
        cells = []
        for h in a.alt:
            rng = h / math.cos(tilt)
            cells.append(f"{char_px(rng, fov, w, a.view):9.1f}px")
        print(f"{name:28s}" + "".join(f"{c:>11s}" for c in cells)
              + f"{max_range_for(READ_THRESHOLDS['reliable'], fov, w, a.view):14.1f} m")
    print("\nNadir camera (tag seen edge-on, ~85 deg): divide the 0 deg numbers by ~11; tags are effectively unreadable.")


if __name__ == "__main__":
    main()
