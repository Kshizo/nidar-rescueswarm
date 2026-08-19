#!/usr/bin/env python3
"""
NIDAR RescueSwarm - Post-Mission Survivor Map
=============================================
Renders the geotag report into a self-contained HTML plan view of the search
area: obstacles, sector split, drone pads, and every geotagged survivor with its
delivery state and WGS-84 coordinate.

    python3 mission_map.py [-o OUT.html]

The mission controller calls write_report() directly at the end of every run,
which drops a timestamped HTML + CSV pair into nidar_ws/mission_reports/ and
repoints latest.html at the newest one -- so every flight leaves an exportable
artefact behind without anyone remembering to run this by hand.

Reads /tmp/nidar_survivor_geotags.json (written by rescueswarm_mission.py) and,
when present, /tmp/world_layout.json for the obstacle field. If the layout also
carries survivor ground truth -- true only in simulation -- the map additionally
scores each geotag and draws the error scatter. At the real mission that block
is simply absent and everything else renders unchanged.
"""

import argparse
import csv
import json
import math
import os
from datetime import datetime

REPORT = "/tmp/nidar_survivor_geotags.json"
LAYOUT = "/tmp/world_layout.json"
DRONE_PADS = [(0.0, -5.0), (0.0, 5.0)]   # (East, North) in field frame

# Validated categorical slots 1-2 (all-pairs, both modes) + fixed status colors.
SERIES = {0: ("#2a78d6", "#3987e5"), 1: ("#eb6834", "#d95926")}


def load(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def match_ground_truth(survivors, truth):
    """Nearest-neighbour match, each truth point used at most once."""
    used, out = set(), {}
    for s in sorted(survivors, key=lambda x: x["id"]):
        cands = [g for g in truth if g["name"] not in used]
        if not cands:
            continue
        best = min(cands, key=lambda g: math.hypot(g["x"] - s["east"], g["y"] - s["north"]))
        d = math.hypot(best["x"] - s["east"], best["y"] - s["north"])
        if d < 3.0:
            used.add(best["name"])
            out[s["id"]] = {
                "name": best["name"], "x": best["x"], "y": best["y"], "z": best["z"],
                "type": best.get("location_type", "ground"),
                "horiz": d, "vert": s["alt"] - best["z"],
            }
    missed = [g for g in truth if g["name"] not in used]
    return out, missed


def build_svg(survivors, layout, gt, bounds):
    """Plan view. Field frame: +X East (right), +Y North (up)."""
    lo, hi = bounds
    span = hi - lo
    P = 34                      # padding inside viewBox
    S = 640                     # plot size
    VB = S + P * 2

    def sx(e):
        return P + (e - lo) / span * S

    def sy(n):
        return P + (hi - n) / span * S      # north up

    o = []
    o.append(f'<svg viewBox="0 0 {VB} {VB}" class="map" role="img" '
             f'aria-label="Plan view of the search area with geotagged survivors">')

    # search-area ground + grid
    o.append(f'<rect x="{P}" y="{P}" width="{S}" height="{S}" class="field"/>')
    step = 10
    g0 = int(math.ceil(lo / step) * step)
    for v in range(g0, int(hi) + 1, step):
        o.append(f'<line x1="{sx(v):.1f}" y1="{P}" x2="{sx(v):.1f}" y2="{P+S}" class="grid"/>')
        o.append(f'<line x1="{P}" y1="{sy(v):.1f}" x2="{P+S}" y2="{sy(v):.1f}" class="grid"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{P+S+16}" class="tick" text-anchor="middle">{v}</text>')
        o.append(f'<text x="{P-8}" y="{sy(v):.1f}" class="tick" text-anchor="end" dy="0.32em">{v}</text>')

    # sector divider (drone 0 west of E=0, drone 1 east)
    o.append(f'<line x1="{sx(0):.1f}" y1="{P}" x2="{sx(0):.1f}" y2="{P+S}" class="sector"/>')
    o.append(f'<text x="{sx(0)-10:.1f}" y="{P+15}" class="sector-lbl" text-anchor="end">WEST · DRONE 0</text>')
    o.append(f'<text x="{sx(0)+10:.1f}" y="{P+15}" class="sector-lbl" text-anchor="start">EAST · DRONE 1</text>')

    # obstacles
    for h in (layout or {}).get("houses", []):
        w = h.get("sx", 6.0) / span * S
        ht = h.get("sy", 6.0) / span * S
        o.append(f'<rect x="{sx(h["x"])-w/2:.1f}" y="{sy(h["y"])-ht/2:.1f}" '
                 f'width="{w:.1f}" height="{ht:.1f}" class="obs obs-house"><title>'
                 f'{h.get("name","house")} · {h.get("h",0):.1f} m tall</title></rect>')
    for t in (layout or {}).get("trees", []):
        r = t.get("canopy_r", 3.0) / span * S
        o.append(f'<circle cx="{sx(t["x"]):.1f}" cy="{sy(t["y"]):.1f}" r="{r:.1f}" '
                 f'class="obs obs-tree"><title>{t.get("name","tree")} · '
                 f'{t.get("h",0):.1f} m tall</title></circle>')
    for w in (layout or {}).get("towers", []):
        s = 2.0 / span * S
        o.append(f'<rect x="{sx(w["x"])-s/2:.1f}" y="{sy(w["y"])-s/2:.1f}" width="{s:.1f}" '
                 f'height="{s:.1f}" class="obs obs-tower"><title>{w.get("name","tower")} · '
                 f'{w.get("h",0):.1f} m tall</title></rect>')

    # physical launchpad deck (10x10 m shared surface both drones return to)
    lp = (layout or {}).get("launchpad")
    if lp:
        side = lp.get("size", 10.0) / span * S
        cx, cy = sx(lp.get("x", 0.0)), sy(lp.get("y", 0.0))
        o.append(f'<g class="lpad">'
                 f'<rect x="{cx-side/2:.1f}" y="{cy-side/2:.1f}" width="{side:.1f}" '
                 f'height="{side:.1f}" rx="2" class="lpad-deck"><title>Launchpad · '
                 f'{lp.get("size",10.0):.0f} x {lp.get("size",10.0):.0f} m</title></rect>'
                 f'<rect x="{cx-side/2+3:.1f}" y="{cy-side/2+3:.1f}" width="{side-6:.1f}" '
                 f'height="{side-6:.1f}" class="lpad-line"/>'
                 f'<line x1="{cx:.1f}" y1="{cy-side*0.18:.1f}" x2="{cx:.1f}" '
                 f'y2="{cy+side*0.18:.1f}" class="lpad-mark"/>'
                 f'<line x1="{cx-side*0.13:.1f}" y1="{cy:.1f}" x2="{cx+side*0.13:.1f}" '
                 f'y2="{cy:.1f}" class="lpad-mark"/>'
                 f'<text x="{cx:.1f}" y="{cy+side/2+11:.1f}" class="lpad-lbl" '
                 f'text-anchor="middle">LAUNCHPAD {lp.get("size",10.0):.0f}\u00d7'
                 f'{lp.get("size",10.0):.0f} m</text></g>')

    # spawn / return points, which sit on the pad's north & south edges
    for i, (pe, pn) in enumerate(DRONE_PADS):
        o.append(f'<g class="pad"><circle cx="{sx(pe):.1f}" cy="{sy(pn):.1f}" r="7" '
                 f'class="pad-ring" style="--s:{SERIES[i][0]}"/>'
                 f'<text x="{sx(pe)+12:.1f}" y="{sy(pn):.1f}" class="pad-lbl" dy="0.32em">'
                 f'PAD {i}</text></g>')

    # ground-truth ghosts + error connectors (simulation only)
    for s in survivors:
        t = gt.get(s["id"])
        if not t:
            continue
        o.append(f'<line x1="{sx(t["x"]):.1f}" y1="{sy(t["y"]):.1f}" '
                 f'x2="{sx(s["east"]):.1f}" y2="{sy(s["north"]):.1f}" class="errline"/>')
        o.append(f'<circle cx="{sx(t["x"]):.1f}" cy="{sy(t["y"]):.1f}" r="3" class="truth"/>')

    # survivors
    for s in survivors:
        drone = s["seen_by"][0] if s.get("seen_by") else 0
        t = gt.get(s["id"])
        roof = t and t["type"] == "rooftop"
        cls = "sv delivered" if s.get("delivered") else "sv pending"
        tip = (f'#{s["id"]} · {"rooftop" if roof else "ground"} · alt {s["alt"]:.2f} m · '
               f'{s["hits"]} hits · drone {"+".join(str(d) for d in s.get("seen_by", []))}'
               + (f' · error {t["horiz"]:.2f} m' if t else ''))
        o.append(f'<g class="{cls}" style="--s:{SERIES.get(drone,SERIES[0])[0]}">')
        o.append(f'<circle cx="{sx(s["east"]):.1f}" cy="{sy(s["north"]):.1f}" r="11" class="halo"/>')
        if roof:
            # square = elevated (rooftop); circle = ground. Shape carries the
            # distinction so it survives colorblind and greyscale reading.
            o.append(f'<rect x="{sx(s["east"])-6:.1f}" y="{sy(s["north"])-6:.1f}" width="12" '
                     f'height="12" rx="2" class="mark"><title>{tip}</title></rect>')
        else:
            o.append(f'<circle cx="{sx(s["east"]):.1f}" cy="{sy(s["north"]):.1f}" r="6.5" '
                     f'class="mark"><title>{tip}</title></circle>')
        o.append(f'<text x="{sx(s["east"]):.1f}" y="{sy(s["north"])-13:.1f}" class="sv-lbl" '
                 f'text-anchor="middle">{s["id"]}</text>')
        o.append('</g>')

    o.append(f'<rect x="{P}" y="{P}" width="{S}" height="{S}" class="frame"/>')
    o.append(f'<text x="{P+S/2}" y="{VB-4}" class="axis" text-anchor="middle">EAST (m)</text>')
    o.append(f'<text x="12" y="{P+S/2}" class="axis" text-anchor="middle" '
             f'transform="rotate(-90 12 {P+S/2})">NORTH (m)</text>')
    o.append('</svg>')
    return "\n".join(o)


def build_scatter(survivors, gt):
    """Error vectors in metres. Sub-metre error is invisible at field scale, so
    it gets its own axes rather than being implied by the map."""
    if not gt:
        return ""
    pts = [(s, gt[s["id"]]) for s in survivors if s["id"] in gt]
    if not pts:
        return ""
    lim = max(0.6, max(max(abs(s["east"] - t["x"]), abs(s["north"] - t["y"])) for s, t in pts) * 1.25)
    S, P = 210, 26
    VB = S + P * 2

    def px(v):
        return P + (v + lim) / (2 * lim) * S

    o = [f'<svg viewBox="0 0 {VB} {VB}" class="scatter" role="img" '
         f'aria-label="Geolocation error per survivor, in metres">']
    o.append(f'<rect x="{P}" y="{P}" width="{S}" height="{S}" class="field"/>')
    for r in (0.25, 0.5, 1.0):
        if r < lim:
            rr = r / (2 * lim) * S
            o.append(f'<circle cx="{px(0):.1f}" cy="{px(0):.1f}" r="{rr:.1f}" class="ring"/>')
            o.append(f'<text x="{px(0)+rr:.1f}" y="{px(0)-3:.1f}" class="tick">{r} m</text>')
    o.append(f'<line x1="{P}" y1="{px(0):.1f}" x2="{P+S}" y2="{px(0):.1f}" class="grid"/>')
    o.append(f'<line x1="{px(0):.1f}" y1="{P}" x2="{px(0):.1f}" y2="{P+S}" class="grid"/>')
    for s, t in pts:
        drone = s["seen_by"][0] if s.get("seen_by") else 0
        de, dn = s["east"] - t["x"], s["north"] - t["y"]
        o.append(f'<circle cx="{px(de):.1f}" cy="{px(-dn):.1f}" r="5" class="mark" '
                 f'style="--s:{SERIES.get(drone,SERIES[0])[0]}"><title>#{s["id"]} '
                 f'ΔE {de:+.2f} m · ΔN {dn:+.2f} m</title></circle>')
    o.append(f'<rect x="{P}" y="{P}" width="{S}" height="{S}" class="frame"/>')
    o.append('</svg>')
    return "\n".join(o)


def render(report, layout, out_path):
    survivors = sorted(report.get("survivors", []), key=lambda s: s["id"])
    truth = (layout or {}).get("survivors", [])
    gt, missed = match_ground_truth(survivors, truth) if truth else ({}, [])

    errs = [v["horiz"] for v in gt.values()]
    delivered = sum(1 for s in survivors if s.get("delivered"))
    both = sum(1 for s in survivors if len(s.get("seen_by", [])) > 1)

    # bounds from everything we draw, snapped out to a round number
    xs = [s["east"] for s in survivors] + [g["x"] for g in truth] + [p[0] for p in DRONE_PADS]
    ys = [s["north"] for s in survivors] + [g["y"] for g in truth] + [p[1] for p in DRONE_PADS]
    _lp = (layout or {}).get("launchpad")
    if _lp:
        _h = _lp.get("size", 10.0) / 2.0
        xs += [_lp.get("x", 0.0) - _h, _lp.get("x", 0.0) + _h]
        ys += [_lp.get("y", 0.0) - _h, _lp.get("y", 0.0) + _h]
    for k, dx, dy in (("houses", "sx", "sy"), ("trees", None, None), ("towers", None, None)):
        for ob in (layout or {}).get(k, []):
            xs.append(ob["x"]); ys.append(ob["y"])
    if not xs:
        xs, ys = [-25, 25], [-25, 25]
    m = max(abs(min(xs)), abs(max(xs)), abs(min(ys)), abs(max(ys))) + 4
    bound = math.ceil(m / 5) * 5
    bounds = (-bound, bound)

    stats = [("Survivors geotagged", str(len(survivors)),
              f"of {len(truth)} present" if truth else "confirmed tracks"),
             ("Kits delivered", str(delivered), "payload released"),
             ("Mean position error", f"{sum(errs)/len(errs):.2f} m" if errs else "—",
              f"worst {max(errs):.2f} m" if errs else "no ground truth"),
             ("Cross-confirmed", str(both), "seen by both drones")]

    stat_html = "\n".join(
        f'<div class="stat"><div class="stat-k">{k}</div>'
        f'<div class="stat-v">{v}</div><div class="stat-s">{s}</div></div>'
        for k, v, s in stats)

    rows = []
    for s in survivors:
        t = gt.get(s["id"])
        drone = s["seen_by"][0] if s.get("seen_by") else 0
        chip = ('<span class="chip ok">delivered</span>' if s.get("delivered")
                else '<span class="chip wait">pending</span>')
        kind = ("rooftop" if t and t["type"] == "rooftop" else "ground") if t else "—"
        err = f'{t["horiz"]:.2f}' if t else "—"
        coord = (f'{s["lat"]:.7f}, {s["lon"]:.7f}' if s.get("lat") is not None else "—")
        rows.append(
            f'<tr><td><span class="dot" style="--s:{SERIES.get(drone,SERIES[0])[0]}"></span>'
            f'<span class="num">{s["id"]}</span></td>'
            f'<td class="num geo">{coord}</td>'
            f'<td class="num">{s["alt"]:.2f}</td><td>{kind}</td>'
            f'<td class="num">{err}</td><td class="num">{s["hits"]}</td>'
            f'<td class="num">{"+".join(str(d) for d in s.get("seen_by", []))}</td>'
            f'<td>{chip}</td></tr>')

    missed_html = ""
    if missed:
        items = "".join(
            f'<li><span class="num">{g["name"]}</span> — E {g["x"]:.1f}, N {g["y"]:.1f}, '
            f'alt {g["z"]:.1f} m <span class="tag">{g.get("location_type","")}</span></li>'
            for g in missed)
        missed_html = (f'<section class="panel warn"><h2>Not found '
                       f'<span class="count">{len(missed)}</span></h2>'
                       f'<p class="note">Present in the world but never confirmed by either '
                       f'drone.</p><ul class="missed">{items}</ul></section>')

    scatter = build_scatter(survivors, gt)
    scatter_html = (f'<section class="panel"><h2>Position error</h2>'
                    f'<p class="note">Estimate minus truth, in metres. Rings mark 0.25, '
                    f'0.5 and 1 m.</p>{scatter}</section>') if scatter else ""

    gen = report.get("generated", datetime.now().isoformat(timespec="seconds"))

    return TEMPLATE.format(
        svg=build_svg(survivors, layout, gt, bounds),
        stats=stat_html, rows="\n".join(rows), missed=missed_html,
        scatter=scatter_html, generated=gen,
        area=f"{bound*2} × {bound*2} m")


TEMPLATE = """<title>Survivor Geotag Map</title>
<style>
  :root {{
    color-scheme: light;
    --ground:#eceff1; --surface:#fbfbfa; --raised:#ffffff;
    --ink:#111820; --muted:#5a6672; --faint:#8b96a1;
    --line:#dde2e6; --field:#f4f6f7; --grid:#e2e7ea;
    --obs:#c8d0d6; --obs-tree:#bfd0c4; --truth:#9aa5b0;
    --good:#0ca30c; --crit:#d03b3b;
    --s1:#2a78d6; --s2:#eb6834;
    --shadow:0 1px 2px rgba(16,24,32,.06), 0 8px 24px -12px rgba(16,24,32,.18);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --ground:#0b1015; --surface:#161c22; --raised:#1c242b;
      --ink:#e8ecef; --muted:#8c99a5; --faint:#69757f;
      --line:#28323a; --field:#121920; --grid:#222c34;
      --obs:#2d3841; --obs-tree:#2a3a33; --truth:#5d6a75;
      --good:#0ca30c; --crit:#d03b3b;
      --s1:#3987e5; --s2:#d95926;
      --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --ground:#0b1015; --surface:#161c22; --raised:#1c242b;
    --ink:#e8ecef; --muted:#8c99a5; --faint:#69757f;
    --line:#28323a; --field:#121920; --grid:#222c34;
    --obs:#2d3841; --obs-tree:#2a3a33; --truth:#5d6a75;
    --good:#0ca30c; --crit:#d03b3b;
    --s1:#3987e5; --s2:#d95926;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
  }}

  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--ground); color:var(--ink);
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    font-size:15px; line-height:1.55;
    -webkit-font-smoothing:antialiased;
  }}
  .num, .geo, .tick, .stat-v {{
    font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
    font-variant-numeric:tabular-nums;
  }}
  .wrap {{ max-width:1320px; margin:0 auto; padding:40px 24px 64px; }}

  header {{ display:flex; flex-wrap:wrap; align-items:baseline; gap:12px 20px;
           padding-bottom:20px; border-bottom:1px solid var(--line); }}
  h1 {{ font-size:26px; font-weight:640; letter-spacing:-.02em; margin:0;
       text-wrap:balance; }}
  .sub {{ color:var(--muted); font-size:13.5px; }}
  .eyebrow {{ font-size:11px; letter-spacing:.13em; text-transform:uppercase;
             color:var(--faint); font-weight:600; width:100%; }}

  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
           gap:1px; background:var(--line); border:1px solid var(--line);
           border-radius:10px; overflow:hidden; margin:24px 0 28px; }}
  .stat {{ background:var(--raised); padding:16px 18px; }}
  .stat-k {{ font-size:11px; letter-spacing:.1em; text-transform:uppercase;
            color:var(--faint); font-weight:600; }}
  .stat-v {{ font-size:27px; font-weight:600; letter-spacing:-.02em; margin-top:4px; }}
  .stat-s {{ font-size:12.5px; color:var(--muted); }}

  .cols {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:22px;
          align-items:start; }}
  @media (max-width:940px) {{ .cols {{ grid-template-columns:1fr; }} }}
  .side {{ display:flex; flex-direction:column; gap:22px; }}

  .panel {{ background:var(--raised); border:1px solid var(--line); border-radius:12px;
           padding:18px 20px 20px; box-shadow:var(--shadow); }}
  .panel h2 {{ font-size:13px; letter-spacing:.09em; text-transform:uppercase;
              color:var(--muted); font-weight:650; margin:0 0 4px; display:flex;
              align-items:center; gap:8px; }}
  .count {{ background:var(--crit); color:#fff; border-radius:999px; padding:1px 8px;
           font-size:11px; letter-spacing:0; }}
  .note {{ font-size:12.5px; color:var(--muted); margin:0 0 14px; }}

  svg {{ display:block; width:100%; height:auto; }}
  .field {{ fill:var(--field); }}
  .frame {{ fill:none; stroke:var(--line); stroke-width:1; }}
  .grid  {{ stroke:var(--grid); stroke-width:1; }}
  .ring  {{ fill:none; stroke:var(--grid); stroke-width:1; }}
  .tick  {{ fill:var(--faint); font-size:10.5px; }}
  .axis  {{ fill:var(--faint); font-size:10.5px; letter-spacing:.12em; }}
  .sector {{ stroke:var(--muted); stroke-width:1; stroke-dasharray:3 5; opacity:.55; }}
  .sector-lbl {{ fill:var(--faint); font-size:10px; letter-spacing:.11em; font-weight:600; }}
  .obs {{ fill:var(--obs); }}
  .obs-tree {{ fill:var(--obs-tree); }}
  .lpad-deck {{ fill:var(--obs); stroke:var(--muted); stroke-width:1; opacity:.55; }}
  .lpad-line {{ fill:none; stroke:var(--muted); stroke-width:1; opacity:.5;
                stroke-dasharray:4 3; }}
  .lpad-mark {{ stroke:var(--muted); stroke-width:2; opacity:.75; stroke-linecap:round; }}
  .lpad-lbl {{ fill:var(--faint); font-size:9px; letter-spacing:.11em; font-weight:600; }}
  .sw.pd {{ background:var(--obs); border:1px solid var(--muted); border-radius:2px; }}
  .planview {{ margin-bottom:22px; }}
  .pad-ring {{ fill:none; stroke:var(--s); stroke-width:1.6; stroke-dasharray:2.5 2.5; }}
  .pad-lbl {{ fill:var(--faint); font-size:9.5px; letter-spacing:.1em; font-weight:600; }}
  .truth {{ fill:none; stroke:var(--truth); stroke-width:1.4; }}
  .errline {{ stroke:var(--truth); stroke-width:1; opacity:.8; }}

  .sv .mark {{ fill:var(--s); stroke:var(--raised); stroke-width:2; }}
  .sv .halo {{ fill:var(--s); opacity:0; transition:opacity .12s; }}
  .sv:hover .halo, .sv:focus-within .halo {{ opacity:.17; }}
  .sv-lbl {{ fill:var(--muted); font-size:10px; font-weight:650;
            font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .sv.pending .mark {{ stroke:var(--crit); stroke-width:2.5; stroke-dasharray:3 2; }}
  .scatter .mark {{ fill:var(--s); stroke:var(--raised); stroke-width:2; }}

  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  .tablewrap {{ overflow-x:auto; }}
  th {{ text-align:left; font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
       color:var(--faint); font-weight:650; padding:0 10px 8px 0;
       border-bottom:1px solid var(--line); white-space:nowrap; }}
  td {{ padding:8px 10px 8px 0; border-bottom:1px solid var(--line); white-space:nowrap; }}
  tr:last-child td {{ border-bottom:none; }}
  .geo {{ font-size:12px; }}
  .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%;
         background:var(--s); margin-right:7px; vertical-align:baseline; }}
  .chip {{ display:inline-flex; align-items:center; gap:5px; font-size:11px;
          font-weight:600; padding:2px 9px; border-radius:999px; }}
  .chip::before {{ content:""; width:5px; height:5px; border-radius:50%; background:currentColor; }}
  .chip.ok {{ color:var(--good); background:color-mix(in srgb, var(--good) 13%, transparent); }}
  .chip.wait {{ color:var(--crit); background:color-mix(in srgb, var(--crit) 13%, transparent); }}

  .legend {{ display:flex; flex-wrap:wrap; gap:8px 20px; margin-top:16px;
            padding-top:14px; border-top:1px solid var(--line);
            font-size:12px; color:var(--muted); }}
  .lg {{ display:flex; align-items:center; gap:7px; }}
  .sw {{ width:11px; height:11px; border-radius:50%; background:var(--s); flex:none; }}
  .sw.sq {{ border-radius:2px; }}
  .sw.gh {{ background:none; border:1.4px solid var(--truth); }}

  .missed {{ list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:7px;
            font-size:13px; }}
  .missed li {{ padding-left:13px; border-left:2px solid var(--crit); }}
  .tag {{ color:var(--faint); font-size:11px; letter-spacing:.06em; text-transform:uppercase; }}
  footer {{ margin-top:30px; padding-top:16px; border-top:1px solid var(--line);
           color:var(--faint); font-size:12px; }}
  @media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; }} }}
</style>

<div class="wrap">
  <header>
    <span class="eyebrow">NIDAR RescueSwarm · post-mission report</span>
    <h1>Survivor Geotag Map</h1>
    <span class="sub">{area} search area · generated {generated}</span>
  </header>

  <div class="stats">{stats}</div>

  <section class="panel planview">
    <h2>Plan view</h2>
    <p class="note">Every marker is a position the swarm derived from its own camera
      and depth sensor. Grey rings show ground truth where available.</p>
    {svg}
    <div class="legend">
      <span class="lg"><span class="sw" style="--s:var(--s1)"></span>Found by drone 0</span>
      <span class="lg"><span class="sw" style="--s:var(--s2)"></span>Found by drone 1</span>
      <span class="lg"><span class="sw sq" style="--s:var(--muted)"></span>Rooftop</span>
      <span class="lg"><span class="sw" style="--s:var(--muted)"></span>Ground</span>
      <span class="lg"><span class="sw gh"></span>Ground truth</span>
      <span class="lg"><span class="sw pd"></span>Launchpad 10&times;10 m</span>
    </div>
  </section>

  <div class="cols">
    <div class="side">
      <section class="panel">
        <h2>Manifest</h2>
        <p class="note">WGS-84 coordinates for the operator handoff.</p>
        <div class="tablewrap"><table>
          <thead><tr><th>ID</th><th>Latitude, longitude</th><th>Alt</th><th>Site</th>
            <th>Err&nbsp;m</th><th>Hits</th><th>Drone</th><th>Kit</th></tr></thead>
          <tbody>{rows}</tbody>
        </table></div>
      </section>
    </div>

    <div class="side">
      {scatter}
      {missed}
    </div>
  </div>

  <footer>Positions fused from repeated detections; each track is the median of its
    estimates. Generated by mission_map.py from the mission geotag report.</footer>
</div>
"""


CSV_COLUMNS = ["id", "lat", "lon", "east", "north", "alt",
               "hits", "seen_by", "delivered"]


def write_csv(report, path):
    """Flat geotag table for GIS import / spreadsheets. One row per confirmed
    track, in the same order the manifest shows them."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for s in sorted(report.get("survivors", []), key=lambda r: r["id"]):
            w.writerow([
                s["id"],
                f"{s['lat']:.7f}" if s.get("lat") is not None else "",
                f"{s['lon']:.7f}" if s.get("lon") is not None else "",
                f"{s['east']:.2f}", f"{s['north']:.2f}", f"{s['alt']:.2f}",
                s.get("hits", ""),
                " ".join(str(d) for d in s.get("seen_by", [])),
                "yes" if s.get("delivered") else "no",
            ])
    return path


def write_report(report, layout, out_dir, stamp=None):
    """Emit the timestamped HTML + CSV pair for one run and repoint latest.html.

    Returns (html_path, csv_path). Never raises -- a reporting failure must not
    take down a mission that has already flown.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = os.path.join(out_dir, f"geotag_map_{stamp}.html")
        csv_path = os.path.join(out_dir, f"geotag_map_{stamp}.csv")

        with open(html_path, "w") as f:
            f.write(render(report, layout or {}, html_path))
        write_csv(report, csv_path)

        # Relative target so the symlink survives the directory being moved.
        latest = os.path.join(out_dir, "latest.html")
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.symlink(os.path.basename(html_path), latest)
        except OSError:
            pass

        return html_path, csv_path
    except Exception as e:
        print(f"[MISSION MAP] [WARN] could not write report: {e}")
        return None, None


def main():
    ap = argparse.ArgumentParser(description="Render the post-mission survivor map.")
    ap.add_argument("-o", "--out", default="/tmp/nidar_survivor_map.html")
    ap.add_argument("--report", default=REPORT)
    ap.add_argument("--layout", default=LAYOUT)
    a = ap.parse_args()

    report = load(a.report)
    if not report or not report.get("survivors"):
        raise SystemExit(f"No geotag report at {a.report} -- run a mission first.")

    html = render(report, load(a.layout, {}), a.out)
    with open(a.out, "w") as f:
        f.write(html)
    csv_path = os.path.splitext(a.out)[0] + ".csv"
    write_csv(report, csv_path)
    print(f"[MISSION MAP] {len(report['survivors'])} survivors -> {a.out}")
    print(f"[MISSION MAP] geotag table -> {csv_path}")


if __name__ == "__main__":
    main()
