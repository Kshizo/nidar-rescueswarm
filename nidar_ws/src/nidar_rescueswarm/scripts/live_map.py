#!/usr/bin/env python3
"""
NIDAR RescueSwarm - Live Mission Map
====================================
Serves a self-refreshing plan view of the mission WHILE it flies:

    http://localhost:8080

Drone positions, headings and trails update several times a second, and each
survivor appears the moment the registry confirms it -- turning green once its
aid kit is away. It is the same field, obstacles and launchpad the post-run
report draws, in the same visual language, so the live view and the archived
report read as one thing.

Deliberately decoupled from the flight loop: the mission hands this module a
*callback*, and a snapshot is taken only when a browser asks for one. There is
no background timer, no shared mutable buffer, and every handler is wrapped --
so a closed tab, a slow client, or a viewer crash cannot perturb the mission.
The server thread is a daemon, so it never holds up process exit either.

Run standalone (replays whatever the last mission left behind) with:

    python3 live_map.py
"""

import json
import math
import threading
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8080


# --------------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RescueSwarm · Live</title>
<style>
  :root {
    color-scheme: light;
    --ground:#eceff1; --surface:#fbfbfa; --raised:#ffffff;
    --ink:#111820; --muted:#5a6672; --faint:#8b96a1;
    --line:#dde2e6; --field:#f4f6f7; --grid:#e2e7ea;
    --obs:#c8d0d6; --obs-tree:#bfd0c4;
    --good:#0ca30c; --crit:#d03b3b;
    --s1:#2a78d6; --s2:#eb6834;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --ground:#0b1015; --surface:#161c22; --raised:#1c242b;
      --ink:#e8ecef; --muted:#8c99a5; --faint:#69757f;
      --line:#28323a; --field:#121920; --grid:#222c34;
      --obs:#2d3841; --obs-tree:#2a3a33;
      --good:#0ca30c; --crit:#d03b3b;
      --s1:#3987e5; --s2:#d95926;
    }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--ground); color:var(--ink);
         font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
         font-size:15px; line-height:1.5; -webkit-font-smoothing:antialiased; }
  .num { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
         font-variant-numeric:tabular-nums; }
  .wrap { max-width:1320px; margin:0 auto; padding:22px 20px 40px; }

  header { display:flex; flex-wrap:wrap; align-items:center; gap:10px 18px;
           padding-bottom:14px; border-bottom:1px solid var(--line); }
  h1 { font-size:21px; font-weight:640; letter-spacing:-.02em; margin:0; }
  .live { display:inline-flex; align-items:center; gap:7px; font-size:11px;
          font-weight:700; letter-spacing:.12em; text-transform:uppercase;
          color:var(--crit); }
  .live::before { content:""; width:8px; height:8px; border-radius:50%;
                  background:var(--crit); animation:pulse 1.6s ease-in-out infinite; }
  .live.stale { color:var(--faint); }
  .live.stale::before { background:var(--faint); animation:none; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
  .spacer { flex:1 1 auto; }
  .kpi { display:flex; gap:22px; }
  .kpi div { text-align:right; }
  .kpi .k { font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
            color:var(--faint); font-weight:600; }
  .kpi .v { font-size:20px; font-weight:600; letter-spacing:-.02em; }

  .cols { display:grid; grid-template-columns:minmax(0,1fr) 330px; gap:18px;
          align-items:start; margin-top:18px; }
  @media (max-width:1000px) { .cols { grid-template-columns:1fr; } }
  .panel { background:var(--raised); border:1px solid var(--line);
           border-radius:12px; padding:14px 16px 16px; }
  .panel h2 { font-size:12px; letter-spacing:.09em; text-transform:uppercase;
              color:var(--muted); font-weight:650; margin:0 0 10px; }
  svg { display:block; width:100%; height:auto; }

  .field { fill:var(--field); }
  .frame { fill:none; stroke:var(--line); stroke-width:1; }
  .grid  { stroke:var(--grid); stroke-width:1; }
  .tick  { fill:var(--faint); font-size:10px;
           font-family:ui-monospace,Menlo,monospace; }
  .sector { stroke:var(--muted); stroke-width:1; stroke-dasharray:3 5; opacity:.5; }
  .sector-lbl { fill:var(--faint); font-size:9.5px; letter-spacing:.11em; font-weight:600; }
  .obs { fill:var(--obs); }
  .obs-tree { fill:var(--obs-tree); }
  .lpad-deck { fill:var(--obs); stroke:var(--muted); stroke-width:1; opacity:.55; }
  .lpad-mark { stroke:var(--muted); stroke-width:2; opacity:.7; stroke-linecap:round; }
  .lpad-lbl { fill:var(--faint); font-size:8.5px; letter-spacing:.11em; font-weight:600; }

  .trail { fill:none; stroke-width:1.6; opacity:.42; stroke-linecap:round;
           stroke-linejoin:round; }
  .sv-mark { stroke:var(--raised); stroke-width:2; }
  .sv-lbl { fill:var(--muted); font-size:9.5px; font-weight:650;
            font-family:ui-monospace,Menlo,monospace; }
  .sv-new { animation:ping 1.1s ease-out 2; }
  @keyframes ping { 0%{r:6;opacity:1} 100%{r:20;opacity:0} }
  .drone-body { stroke:var(--raised); stroke-width:1.5; }
  .drone-lbl { font-size:10px; font-weight:700;
               font-family:ui-monospace,Menlo,monospace; }
  .det-ring { fill:none; stroke-width:2; opacity:.85; }

  .dstat { display:flex; align-items:baseline; gap:8px; padding:8px 0;
           border-bottom:1px solid var(--line); font-size:13px; }
  .dstat:last-child { border-bottom:none; }
  .dot { width:9px; height:9px; border-radius:50%; flex:none; }
  .dstat .nm { font-weight:650; }
  .dstat .sp { margin-left:auto; color:var(--muted); font-size:12.5px; }
  .badge { font-size:10px; font-weight:700; letter-spacing:.06em; padding:1px 7px;
           border-radius:999px; text-transform:uppercase; }
  .badge.det { color:var(--crit); background:color-mix(in srgb,var(--crit) 14%,transparent); }

  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th { text-align:left; font-size:10px; letter-spacing:.09em; text-transform:uppercase;
       color:var(--faint); font-weight:650; padding:0 8px 6px 0;
       border-bottom:1px solid var(--line); }
  td { padding:6px 8px 6px 0; border-bottom:1px solid var(--line); white-space:nowrap; }
  tr:last-child td { border-bottom:none; }
  .geo { font-family:ui-monospace,Menlo,monospace; font-size:11.5px; }
  .chip { font-size:10px; font-weight:650; padding:1px 7px; border-radius:999px; }
  .chip.ok { color:var(--good); background:color-mix(in srgb,var(--good) 14%,transparent); }
  .chip.wait { color:var(--muted); background:color-mix(in srgb,var(--muted) 12%,transparent); }
  .empty { color:var(--faint); font-size:12.5px; padding:6px 0; }
  .scroll { max-height:340px; overflow-y:auto; }
  footer { margin-top:16px; color:var(--faint); font-size:11.5px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>RescueSwarm</h1>
    <span class="live" id="live">live</span>
    <span class="spacer"></span>
    <div class="kpi">
      <div><div class="k">Elapsed</div><div class="v num" id="elapsed">--:--</div></div>
      <div><div class="k">Geotagged</div><div class="v num" id="nfound">0</div></div>
      <div><div class="k">Delivered</div><div class="v num" id="ndeliv">0</div></div>
    </div>
  </header>

  <div class="cols">
    <section class="panel">
      <h2>Plan view · field frame</h2>
      <svg id="map" viewBox="0 0 708 708" role="img"
           aria-label="Live plan view of the search area"></svg>
    </section>

    <div>
      <section class="panel" style="margin-bottom:18px">
        <h2>Drones</h2>
        <div id="drones"><div class="empty">waiting for telemetry…</div></div>
      </section>
      <section class="panel">
        <h2>Survivors</h2>
        <div class="scroll">
          <table>
            <thead><tr><th>ID</th><th>Latitude, longitude</th><th>Hits</th><th>Kit</th></tr></thead>
            <tbody id="svrows"></tbody>
          </table>
          <div class="empty" id="svempty">none confirmed yet</div>
        </div>
      </section>
    </div>
  </div>
  <footer id="foot">connecting…</footer>
</div>

<script>
const SVGNS = "http://www.w3.org/2000/svg";
const SERIES = ["var(--s1)", "var(--s2)"];
const P = 34, S = 640, VB = S + P * 2;
const POLL_MS = 200, TRAIL_MAX = 400;

let layout = null, lo = -30, hi = 30, span = 60;
let trails = {}, seenIds = new Set(), lastOk = 0;

const sx = e => P + (e - lo) / span * S;
const sy = n => P + (hi - n) / span * S;
const el = (t, a) => { const x = document.createElementNS(SVGNS, t);
  for (const k in a) x.setAttribute(k, a[k]); return x; };

function computeBounds(L) {
  let m = 26;
  const push = (x, y) => { m = Math.max(m, Math.abs(x) + 2, Math.abs(y) + 2); };
  for (const k of ["houses", "trees", "towers"])
    (L[k] || []).forEach(o => push(o.x, o.y));
  if (L.launchpad) push(L.launchpad.x + L.launchpad.size / 2,
                        L.launchpad.y + L.launchpad.size / 2);
  const b = Math.ceil((m + 4) / 5) * 5;
  lo = -b; hi = b; span = hi - lo;
}

// Static scenery is drawn once into its own <g>; only the dynamic layer is
// rebuilt each tick, so the browser is not re-parsing the whole field at 5 Hz.
function drawStatic(g) {
  g.appendChild(el("rect", {x: P, y: P, width: S, height: S, class: "field"}));
  const step = 10, g0 = Math.ceil(lo / step) * step;
  for (let v = g0; v <= hi; v += step) {
    g.appendChild(el("line", {x1: sx(v), y1: P, x2: sx(v), y2: P + S, class: "grid"}));
    g.appendChild(el("line", {x1: P, y1: sy(v), x2: P + S, y2: sy(v), class: "grid"}));
    let t = el("text", {x: sx(v), y: P + S + 15, class: "tick", "text-anchor": "middle"});
    t.textContent = v; g.appendChild(t);
    t = el("text", {x: P - 7, y: sy(v), class: "tick", "text-anchor": "end", dy: "0.32em"});
    t.textContent = v; g.appendChild(t);
  }
  g.appendChild(el("line", {x1: sx(0), y1: P, x2: sx(0), y2: P + S, class: "sector"}));
  let t = el("text", {x: sx(0) - 9, y: P + 14, class: "sector-lbl", "text-anchor": "end"});
  t.textContent = "WEST · DRONE 0"; g.appendChild(t);
  t = el("text", {x: sx(0) + 9, y: P + 14, class: "sector-lbl", "text-anchor": "start"});
  t.textContent = "EAST · DRONE 1"; g.appendChild(t);

  const L = layout || {};
  if (L.launchpad) {
    const side = L.launchpad.size / span * S;
    const cx = sx(L.launchpad.x), cy = sy(L.launchpad.y);
    g.appendChild(el("rect", {x: cx - side / 2, y: cy - side / 2, width: side,
                              height: side, rx: 2, class: "lpad-deck"}));
    g.appendChild(el("line", {x1: cx, y1: cy - side * .18, x2: cx,
                              y2: cy + side * .18, class: "lpad-mark"}));
    g.appendChild(el("line", {x1: cx - side * .13, y1: cy, x2: cx + side * .13,
                              y2: cy, class: "lpad-mark"}));
    const lb = el("text", {x: cx, y: cy + side / 2 + 10, class: "lpad-lbl",
                           "text-anchor": "middle"});
    lb.textContent = "LAUNCHPAD"; g.appendChild(lb);
  }
  (L.houses || []).forEach(h => {
    const w = (h.sx || 6) / span * S, ht = (h.sy || 6) / span * S;
    const r = el("rect", {x: sx(h.x) - w / 2, y: sy(h.y) - ht / 2, width: w,
                          height: ht, class: "obs"});
    const ti = el("title"); ti.textContent = (h.name || "house") + " · " +
      (h.h || 0).toFixed(1) + " m"; r.appendChild(ti); g.appendChild(r);
  });
  (L.trees || []).forEach(t2 => {
    const r = (t2.canopy_r || 3) / span * S;
    g.appendChild(el("circle", {cx: sx(t2.x), cy: sy(t2.y), r: r, class: "obs obs-tree"}));
  });
  (L.towers || []).forEach(w => {
    const s2 = 2 / span * S;
    g.appendChild(el("rect", {x: sx(w.x) - s2 / 2, y: sy(w.y) - s2 / 2,
                              width: s2, height: s2, class: "obs"}));
  });
  g.appendChild(el("rect", {x: P, y: P, width: S, height: S, class: "frame"}));
}

function drawDynamic(g, st) {
  // trails first so drones and survivors sit above them
  (st.drones || []).forEach(d => {
    const pts = (d.trail && d.trail.length > 1) ? d.trail : trails[d.id];
    if (!pts || pts.length < 2) return;
    g.appendChild(el("polyline", {
      points: pts.map(p => sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1)).join(" "),
      class: "trail", stroke: SERIES[d.id % 2]}));
  });

  (st.survivors || []).forEach(s => {
    const col = s.delivered ? "var(--good)"
              : SERIES[(s.seen_by && s.seen_by.length ? s.seen_by[0] : 0) % 2];
    const x = sx(s.east), y = sy(s.north);
    if (!seenIds.has(s.id)) {           // one-shot ping on first appearance
      seenIds.add(s.id);
      ping(x, y, col);
    }
    const r = el("rect", {x: x - 5.5, y: y - 5.5, width: 11, height: 11, rx: 2,
                          fill: col, class: "sv-mark"});
    const ti = el("title");
    ti.textContent = "#" + s.id + " · " + s.hits + " hits · alt " +
      s.alt.toFixed(2) + " m" + (s.lat != null ?
      " · " + s.lat.toFixed(7) + ", " + s.lon.toFixed(7) : "");
    r.appendChild(ti); g.appendChild(r);
    const lb = el("text", {x: x + 9, y: y, class: "sv-lbl", dy: "0.32em"});
    lb.textContent = s.id; g.appendChild(lb);
  });

  (st.drones || []).forEach(d => {
    const col = SERIES[d.id % 2], x = sx(d.east), y = sy(d.north);
    if (d.detecting)
      g.appendChild(el("circle", {cx: x, cy: y, r: 13, class: "det-ring", stroke: col}));
    // heading triangle: yaw is clockwise-from-north, SVG y grows downward
    const a = d.yaw, ca = Math.cos(a), sa = Math.sin(a);
    const pt = (fwd, lat) => {
      const e = fwd * sa + lat * ca, n = fwd * ca - lat * sa;
      return (x + e * 1.0).toFixed(1) + "," + (y - n * 1.0).toFixed(1);
    };
    g.appendChild(el("polygon", {points: [pt(9, 0), pt(-5, 5), pt(-5, -5)].join(" "),
                                 fill: col, class: "drone-body"}));
    const lb = el("text", {x: x + 13, y: y - 10, class: "drone-lbl", fill: col});
    lb.textContent = "D" + d.id; g.appendChild(lb);
  });
}

// The dynamic layer is wiped every tick, so a ping drawn there would be torn
// out mid-animation. Pings go in their own persistent layer and self-remove.
function ping(x, y, col) {
  const layer = document.getElementById("ping");
  if (!layer) return;
  const c = el("circle", {cx: x, cy: y, r: 6, fill: "none", stroke: col,
                          "stroke-width": 2, class: "sv-new"});
  layer.appendChild(c);
  setTimeout(() => c.remove(), 2400);
}

function renderSide(st) {
  const dv = document.getElementById("drones");
  if (!st.drones || !st.drones.length) {
    dv.innerHTML = '<div class="empty">waiting for telemetry…</div>';
  } else {
    dv.innerHTML = st.drones.map(d =>
      '<div class="dstat"><span class="dot" style="background:' + SERIES[d.id % 2] + '"></span>' +
      '<span class="nm">Drone ' + d.id + '</span>' +
      (d.detecting ? '<span class="badge det">survivor</span>' : '') +
      '<span class="sp num">' + d.alt.toFixed(1) + ' m · ' + d.speed.toFixed(1) + ' m/s</span></div>'
    ).join("");
  }

  const rows = st.survivors || [];
  document.getElementById("svempty").style.display = rows.length ? "none" : "block";
  document.getElementById("svrows").innerHTML = rows.map(s =>
    '<tr><td class="num">#' + s.id + '</td>' +
    '<td class="geo">' + (s.lat != null ? s.lat.toFixed(7) + ', ' + s.lon.toFixed(7) : '—') + '</td>' +
    '<td class="num">' + s.hits + '</td>' +
    '<td><span class="chip ' + (s.delivered ? 'ok">delivered' : 'wait">pending') + '</span></td></tr>'
  ).join("");

  document.getElementById("elapsed").textContent = st.elapsed || "--:--";
  document.getElementById("nfound").textContent = rows.length;
  document.getElementById("ndeliv").textContent = rows.filter(s => s.delivered).length;
  document.getElementById("foot").textContent =
    (st.phase || "") + " · updated " + new Date().toLocaleTimeString();
}

function tick(st) {
  (st.drones || []).forEach(d => {
    const t = trails[d.id] || (trails[d.id] = []);
    const last = t[t.length - 1];
    if (!last || Math.hypot(last[0] - d.east, last[1] - d.north) > 0.4) {
      t.push([d.east, d.north]);
      if (t.length > TRAIL_MAX) t.shift();
    }
  });
  const svg = document.getElementById("map");
  let dyn = document.getElementById("dyn");
  if (!dyn) {
    svg.setAttribute("viewBox", "0 0 " + VB + " " + VB);
    const stat = el("g", {id: "stat"}); drawStatic(stat); svg.appendChild(stat);
    dyn = el("g", {id: "dyn"}); svg.appendChild(dyn);
    svg.appendChild(el("g", {id: "ping"}));
  }
  while (dyn.firstChild) dyn.removeChild(dyn.firstChild);
  drawDynamic(dyn, st);
  renderSide(st);
}

async function poll() {
  try {
    if (!layout) {
      layout = await (await fetch("layout.json", {cache: "no-store"})).json();
      computeBounds(layout);
    }
    const st = await (await fetch("state.json", {cache: "no-store"})).json();
    lastOk = Date.now();
    tick(st);
  } catch (e) { /* mission not up yet, or gone -- keep trying */ }
  const stale = Date.now() - lastOk > 3000;
  const lv = document.getElementById("live");
  lv.classList.toggle("stale", stale);
  lv.textContent = stale ? "no signal" : "live";
  setTimeout(poll, POLL_MS);
}
poll();
</script>
</body>
</html>
"""


# ------------------------------------------------------------------- server

class _Handler(BaseHTTPRequestHandler):
    server_version = "NidarLiveMap/1.0"

    def _send(self, body, ctype):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        try:
            if path in ("/", "/index.html"):
                self._send(PAGE, "text/html; charset=utf-8")
            elif path == "/layout.json":
                self._send(json.dumps(self.server.layout), "application/json")
            elif path == "/state.json":
                self._send(json.dumps(self.server.state_fn()), "application/json")
            else:
                self.send_error(404)
        except BrokenPipeError:
            pass          # browser navigated away mid-write; nothing to do
        except Exception:
            try:
                self.send_error(500)
            except Exception:
                pass

    def log_message(self, *args):
        pass              # a 5 Hz poll would otherwise flood the mission console


class LiveMapServer:
    """Serves the live map. start() is non-blocking and never raises: losing the
    viewer must never cost us the flight."""

    def __init__(self, state_fn, layout=None, port=DEFAULT_PORT, host=""):
        self.state_fn = state_fn
        self.layout = layout or {}
        self.port = port
        self.host = host
        self._httpd = None
        self._thread = None

    def start(self):
        try:
            httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        except OSError as e:
            print(f"[LIVE MAP] [WARN] port {self.port} unavailable ({e}); "
                  f"live map disabled")
            return None
        httpd.state_fn = self._safe_state
        httpd.layout = self.layout
        httpd.daemon_threads = True
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever,
                                        kwargs={"poll_interval": 0.2}, daemon=True)
        self._thread.start()
        print(f"[LIVE MAP] http://localhost:{self.port}  (live mission map)")
        return httpd

    def _safe_state(self):
        try:
            return self.state_fn()
        except Exception as e:
            return {"error": str(e), "drones": [], "survivors": []}

    def stop(self):
        if self._httpd:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None


TRAIL_MAXLEN = 600      # ~1.5 km of path at the 2.5 m/s cruise
TRAIL_MIN_STEP = 0.6    # metres of travel before a point is worth keeping


def record_trail(trails, idx, east, north):
    """Append to a drone's flown path, from the telemetry loop.

    Recorded here rather than accumulated in the browser so the full path is
    already there when a tab is opened mid-mission -- an operator joining late
    should see where the swarm has been, not just where it goes next.
    """
    t = trails.get(idx)
    if t is None:
        t = trails[idx] = deque(maxlen=TRAIL_MAXLEN)
    if not t or math.hypot(t[-1][0] - east, t[-1][1] - north) > TRAIL_MIN_STEP:
        t.append((round(east, 1), round(north, 1)))


def build_state(drone_positions, drone_yaws, drone_velocities, survivors,
                spawn_poses, detections=None, started=None, phase="",
                trails=None):
    """Snapshot the swarm in the shared FIELD frame (+X East, +Y North).

    Each drone's PX4 local NED origin is its own spawn pad, so positions are
    offset onto the common frame here -- exactly as the registry does -- or the
    two drones would be drawn 10 m apart from where they actually are.
    """
    drones = []
    for idx, pos in sorted((drone_positions or {}).items()):
        try:
            sp_x, sp_y = spawn_poses[idx]
            vel = (drone_velocities or {}).get(idx) or (0.0, 0.0, 0.0)
            det = (detections or {}).get(idx) or {}
            drones.append({
                "id": idx,
                "east": round(pos[1] + sp_x, 2),
                "north": round(pos[0] + sp_y, 2),
                "alt": round(-pos[2], 2),
                "yaw": round((drone_yaws or {}).get(idx, 0.0), 4),
                "speed": round(math.hypot(vel[0], vel[1]), 2),
                "detecting": bool(det.get("detected")),
                "trail": [list(p) for p in (trails or {}).get(idx, ())],
            })
        except Exception:
            continue

    elapsed = "--:--"
    if started is not None:
        s = max(0, int(datetime.now().timestamp() - started))
        elapsed = f"{s // 60:02d}:{s % 60:02d}"

    return {"elapsed": elapsed, "phase": phase,
            "drones": drones, "survivors": survivors or []}


def main():
    """Standalone mode: serve the last mission's report as a static page. Useful
    for checking the viewer without burning a flight."""
    import argparse
    ap = argparse.ArgumentParser(description="Serve the live mission map.")
    ap.add_argument("-p", "--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--report", default="/tmp/nidar_survivor_geotags.json")
    ap.add_argument("--layout", default="/tmp/world_layout.json")
    a = ap.parse_args()

    def load(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return {}

    srv = LiveMapServer(lambda: {"elapsed": "--:--", "phase": "replay (no mission running)",
                                 "drones": [],
                                 "survivors": load(a.report).get("survivors", [])},
                        layout=load(a.layout), port=a.port)
    if srv.start() is None:
        return
    print("[LIVE MAP] Ctrl+C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.stop()


if __name__ == "__main__":
    main()
