import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from scipy.optimize import minimize_scalar

# ============================================================
# EML SPACE LOGISTICS: EARTH-MARS SIMULATION
# STREAMLIT VERSION (ZERO-FLICKER CLIENT CANVAS)
# ============================================================

st.set_page_config(
    page_title="Earth–Mars Lambert Orbital Logistics",
    page_icon="🚀",
    layout="wide",
)

# ============================================================
# INITIAL CONSTANTS
# ============================================================

LAUNCH_WINDOW_DAYS = 60

R_EARTH = 1.0
R_MARS = 1.524

PERIOD_EARTH = 365.25
PERIOD_MARS = 687.0

W_EARTH = float(2 * np.pi / PERIOD_EARTH)
W_MARS = float(2 * np.pi / PERIOD_MARS)

mu = float((2 * np.pi / PERIOD_EARTH) ** 2)

THETA_E_0 = 0.0
THETA_M_0 = float(np.radians(44.0))

# ============================================================
# SESSION STATE DEFAULTS
# ============================================================

DEFAULTS = {
    "outbound_qty": 50,
    "return_qty": 50,
    "transit_time": 259,
    "start_time": -30.0,
    "total_days": 600,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# ORBITAL MECHANICS (LAMBERT SOLVER)
# ============================================================

def solve_lambert_orbit(t_launch, is_outbound, transit_time):
    if is_outbound:
        theta1 = THETA_E_0 + W_EARTH * t_launch
        r1 = R_EARTH
        theta2 = THETA_M_0 + W_MARS * (t_launch + transit_time)
        r2 = R_MARS
    else:
        theta1 = THETA_M_0 + W_MARS * t_launch
        r1 = R_MARS
        theta2 = THETA_E_0 + W_EARTH * (t_launch + transit_time)
        r2 = R_EARTH

    d_theta = theta2 - theta1
    while d_theta < 0:
        d_theta += 2 * np.pi

    dx = r1 * np.cos(theta1) - r2 * np.cos(theta2)
    dy = r1 * np.sin(theta1) - r2 * np.sin(theta2)
    dr = r2 - r1
    norm2 = dx**2 + dy**2

    if norm2 <= 0:
        return None

    ex_p = dr * dx / norm2
    ey_p = dr * dy / norm2
    k_bound = np.sqrt(max(0, 1.0 - dr**2 / norm2) / norm2)

    def calc_tof(k):
        ex = ex_p - k * dy
        ey = ey_p + k * dx
        e = np.sqrt(ex**2 + ey**2)
        if e >= 0.999:
            return 1e9

        omega = np.arctan2(ey, ex)
        p = r1 * (1 + ex * np.cos(theta1) + ey * np.sin(theta1))
        if p <= 0:
            return 1e9

        a = p / (1 - e**2)
        nu1 = theta1 - omega

        E1 = 2 * np.arctan2(
            np.sqrt(1 - e) * np.sin(nu1 / 2),
            np.sqrt(1 + e) * np.cos(nu1 / 2),
        )
        E2 = 2 * np.arctan2(
            np.sqrt(1 - e) * np.sin((nu1 + d_theta) / 2),
            np.sqrt(1 + e) * np.cos((nu1 + d_theta) / 2),
        )

        while E2 < E1:
            E2 += 2 * np.pi
        while E2 > E1 + 2 * np.pi:
            E2 -= 2 * np.pi

        M1 = E1 - e * np.sin(E1)
        M2 = E2 - e * np.sin(E2)
        return (M2 - M1) * np.sqrt(a**3 / mu)

    try:
        result = minimize_scalar(
            lambda k: (calc_tof(k) - transit_time) ** 2,
            bounds=(-k_bound * 0.99, k_bound * 0.99),
            method="bounded",
        )
    except Exception:
        return None

    if not result.success:
        return None

    k = result.x
    ex = ex_p - k * dy
    ey = ey_p + k * dx
    e = np.sqrt(ex**2 + ey**2)

    if e >= 0.999:
        return None

    omega = np.arctan2(ey, ex)
    p = r1 * (1 + ex * np.cos(theta1) + ey * np.sin(theta1))
    if p <= 0:
        return None

    a = p / (1 - e**2)
    nu1 = theta1 - omega
    E1 = 2 * np.arctan2(
        np.sqrt(1 - e) * np.sin(nu1 / 2),
        np.sqrt(1 + e) * np.cos(nu1 / 2),
    )
    M1 = E1 - e * np.sin(E1)

    return {
        "a": float(a),
        "e": float(e),
        "omega": float(omega),
        "M1": float(M1),
        "n": float(np.sqrt(mu / a**3)),
        "t_launch": float(t_launch),
        "transit": float(transit_time),
    }

@st.cache_data(show_spinner="Calculating Lambert transfer orbits...")
def recalculate_orbits(outbound_qty, return_qty, transit_time):
    outbound_start = -LAUNCH_WINDOW_DAYS / 2
    outbound_end = LAUNCH_WINDOW_DAYS / 2

    t_launch_out = np.linspace(outbound_start, outbound_end, outbound_qty)
    t_launch_ret = np.linspace(outbound_start, outbound_end, return_qty)

    outbound = [solve_lambert_orbit(tl, True, transit_time) for tl in t_launch_out]
    incoming = [solve_lambert_orbit(tl, False, transit_time) for tl in t_launch_ret]

    # Filter out invalid numerical results
    return [o for o in outbound if o is not None], [o for o in incoming if o is not None]

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:
    st.header("Simulation Parameters")

    st.session_state.outbound_qty = st.slider(
        "Earth Cargo Quantity",
        min_value=10,
        max_value=500,
        value=st.session_state.outbound_qty,
        step=10,
        help="Number of Earth → Mars cargo trajectories.",
    )

    st.session_state.return_qty = st.slider(
        "Mars Cargo Quantity",
        min_value=10,
        max_value=500,
        value=st.session_state.return_qty,
        step=10,
        help="Number of Mars → Earth cargo trajectories.",
    )

    st.session_state.transit_time = st.slider(
        "Transit Time [days]",
        min_value=100,
        max_value=500,
        value=st.session_state.transit_time,
        step=5,
    )

    st.session_state.start_time = st.slider(
        "Start Time [days]",
        min_value=-500,
        max_value=0,
        value=int(st.session_state.start_time),
        step=10,
    )

    st.session_state.total_days = st.slider(
        "Total Simulation Days",
        min_value=200,
        max_value=1500,
        value=int(st.session_state.total_days),
        step=50,
    )

outbound_orbits, return_orbits = recalculate_orbits(
    st.session_state.outbound_qty,
    st.session_state.return_qty,
    st.session_state.transit_time,
)

# ============================================================
# CLIENT CANVAS COMPONENT (ZERO-LATENCY 60 FPS RENDERER)
# ============================================================

st.title("Interactive Lambert Orbital Logistics Simulation")
st.caption("Earth–Mars / Mars–Earth electromagnetic launch logistics model")

sim_payload = {
    "outbound": outbound_orbits,
    "returns": return_orbits,
    "startTime": float(st.session_state.start_time),
    "totalDays": float(st.session_state.total_days),
    "rEarth": R_EARTH,
    "rMars": R_MARS,
    "wEarth": W_EARTH,
    "wMars": W_MARS,
    "thetaE0": THETA_E_0,
    "thetaM0": THETA_M_0,
    "transitTime": st.session_state.transit_time,
}

html_code = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    padding: 0;
    background-color: #0a0a1a;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    user-select: none;
    overflow: hidden;
  }}
  #controls {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    background: #111126;
    border-radius: 8px;
    margin-bottom: 8px;
    border: 1px solid #222244;
  }}
  button {{
    background: #24244a;
    color: #fff;
    border: 1px solid #44447a;
    border-radius: 6px;
    padding: 6px 14px;
    cursor: pointer;
    font-weight: 600;
  }}
  button:hover {{
    background: #36366d;
  }}
  input[type="range"] {{
    flex: 1;
    cursor: pointer;
    accent-color: #00ffff;
  }}
  #metrics {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 10px;
  }}
  .metric-card {{
    background: #111126;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid #222244;
  }}
  .metric-title {{
    font-size: 11px;
    color: #8888aa;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .metric-val {{
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 2px;
  }}
  canvas {{
    display: block;
    border-radius: 8px;
    background: #0a0a1a;
    border: 1px solid #222244;
  }}
</style>
</head>
<body>

<div id="controls">
  <button id="playBtn">⏸️ Pause</button>
  <button id="resetBtn">↺ Reset</button>
  <span style="font-size: 13px; color: #8888aa;">Scrubber:</span>
  <input type="range" id="timeSlider" step="0.5">
  <span id="speedLabel" style="font-size: 12px; color: #aaa;">1x</span>
  <button id="speedBtn">⏩ Fast</button>
</div>

<canvas id="simCanvas" width="1100" height="650"></canvas>

<div id="metrics">
  <div class="metric-card">
    <div class="metric-title">Simulation Day</div>
    <div class="metric-val" id="metricDay">-30</div>
  </div>
  <div class="metric-card">
    <div class="metric-title">Earth Cargo In Flight</div>
    <div class="metric-val" id="metricOutbound" style="color: #00ff00;">0</div>
  </div>
  <div class="metric-card">
    <div class="metric-title">Mars Cargo In Flight</div>
    <div class="metric-val" id="metricReturns" style="color: #ffaa00;">0</div>
  </div>
  <div class="metric-card">
    <div class="metric-title">Transit Time</div>
    <div class="metric-val" id="metricTransit">{st.session_state.transit_time} days</div>
  </div>
</div>

<script>
const sim = {json.dumps(sim_payload)};

const canvas = document.getElementById("simCanvas");
const ctx = canvas.getContext("2d");
const playBtn = document.getElementById("playBtn");
const resetBtn = document.getElementById("resetBtn");
const speedBtn = document.getElementById("speedBtn");
const speedLabel = document.getElementById("speedLabel");
const slider = document.getElementById("timeSlider");

const metricDay = document.getElementById("metricDay");
const metricOutbound = document.getElementById("metricOutbound");
const metricReturns = document.getElementById("metricReturns");

let isRunning = true;
let speed = 1.0;
let currentTime = sim.startTime;
const maxTime = sim.startTime + sim.totalDays;

slider.min = sim.startTime;
slider.max = maxTime;
slider.value = currentTime;

playBtn.onclick = () => {{
  isRunning = !isRunning;
  playBtn.innerText = isRunning ? "⏸️ Pause" : "▶️ Play";
}};

resetBtn.onclick = () => {{
  currentTime = sim.startTime;
  slider.value = currentTime;
}};

speedBtn.onclick = () => {{
  if (speed === 1.0) {{ speed = 2.5; speedLabel.innerText = "2.5x"; }}
  else if (speed === 2.5) {{ speed = 5.0; speedLabel.innerText = "5x"; }}
  else {{ speed = 1.0; speedLabel.innerText = "1x"; }}
}};

slider.oninput = () => {{
  currentTime = parseFloat(slider.value);
  isRunning = false;
  playBtn.innerText = "▶️ Play";
}};

function getPayloadPos(t, orbit) {{
  if (t < orbit.t_launch || t > orbit.t_launch + orbit.transit) return null;
  const dt = t - orbit.t_launch;
  const M = orbit.M1 + orbit.n * dt;
  const e = orbit.e;

  let E = M;
  for (let i = 0; i < 5; i++) {{
    const den = 1 - e * Math.cos(E);
    if (Math.abs(den) < 1e-12) break;
    E = E - (E - e * Math.sin(E) - M) / den;
  }}

  const nu = 2 * Math.atan2(
    Math.sqrt(1 + e) * Math.sin(E / 2),
    Math.sqrt(1 - e) * Math.cos(E / 2)
  );
  const r = orbit.a * (1 - e * Math.cos(E));
  const theta = orbit.omega + nu;
  return [r * Math.cos(theta), r * Math.sin(theta)];
}}

function render() {{
  if (isRunning) {{
    currentTime += 0.5 * speed;
    if (currentTime > maxTime) currentTime = sim.startTime;
    slider.value = currentTime;
  }}

  // Resize canvas responsively
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== width || canvas.height !== height) {{
    canvas.width = width;
    canvas.height = height;
  }}

  const cx = width / 2;
  const cy = height / 2;
  const scale = Math.min(width, height) / 4.4;

  ctx.clearRect(0, 0, width, height);

  // Helper coordinate projector
  const toX = (x) => cx + x * scale;
  const toY = (y) => cy - y * scale;

  // Orbit Rings
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);

  ctx.strokeStyle = "rgba(0, 255, 255, 0.25)";
  ctx.beginPath();
  ctx.arc(cx, cy, sim.rEarth * scale, 0, 2 * Math.PI);
  ctx.stroke();

  ctx.strokeStyle = "rgba(255, 68, 68, 0.25)";
  ctx.beginPath();
  ctx.arc(cx, cy, sim.rMars * scale, 0, 2 * Math.PI);
  ctx.stroke();

  ctx.setLineDash([]);

  // Sun
  ctx.fillStyle = "#ffdd00";
  ctx.shadowColor = "#ffaa00";
  ctx.shadowBlur = 15;
  ctx.beginPath();
  ctx.arc(cx, cy, 9, 0, 2 * Math.PI);
  ctx.fill();
  ctx.shadowBlur = 0;

  // Earth
  const thetaE = sim.thetaE0 + sim.wEarth * currentTime;
  const ex = sim.rEarth * Math.cos(thetaE);
  const ey = sim.rEarth * Math.sin(thetaE);
  ctx.fillStyle = "#00ffff";
  ctx.shadowColor = "#00ffff";
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.arc(toX(ex), toY(ey), 6.5, 0, 2 * Math.PI);
  ctx.fill();

  // Mars
  const thetaM = sim.thetaM0 + sim.wMars * currentTime;
  const mx = sim.rMars * Math.cos(thetaM);
  const my = sim.rMars * Math.sin(thetaM);
  ctx.fillStyle = "#ff4444";
  ctx.shadowColor = "#ff4444";
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.arc(toX(mx), toY(my), 5.5, 0, 2 * Math.PI);
  ctx.fill();
  ctx.shadowBlur = 0;

  // Earth -> Mars Payloads
  ctx.fillStyle = "#00ff66";
  let activeOutbound = 0;
  for (let i = 0; i < sim.outbound.length; i++) {{
    const pos = getPayloadPos(currentTime, sim.outbound[i]);
    if (pos) {{
      activeOutbound++;
      ctx.beginPath();
      ctx.arc(toX(pos[0]), toY(pos[1]), 3.2, 0, 2 * Math.PI);
      ctx.fill();
    }}
  }}

  // Mars -> Earth Payloads
  ctx.fillStyle = "#ffaa00";
  let activeReturns = 0;
  for (let i = 0; i < sim.returns.length; i++) {{
    const pos = getPayloadPos(currentTime, sim.returns[i]);
    if (pos) {{
      activeReturns++;
      ctx.beginPath();
      ctx.arc(toX(pos[0]), toY(pos[1]), 3.2, 0, 2 * Math.PI);
      ctx.fill();
    }}
  }}

  metricDay.innerText = Math.round(currentTime);
  metricOutbound.innerText = activeOutbound;
  metricReturns.innerText = activeReturns;

  requestAnimationFrame(render);
}}

requestAnimationFrame(render);
</script>
</body>
</html>
"""

components.html(html_code, height=780, scrolling=False)