import { useEffect, useMemo, useRef, useState } from "react";
import { DATASETS, DATASET_MAP } from "./data/datasets.js";
import { LOSSES, fitMetrics, makeDatasetObjective, simulate } from "./lib/optimize.js";

const FNS = {
  bowl: { name: "Simple bowl: f = x² + y²", f: (x, y) => x * x + y * y, g: (x, y) => [2 * x, 2 * y], opt: [0, 0] },
  narrow: { name: "Narrow valley: f = x² + 10y²", f: (x, y) => x * x + 10 * y * y, g: (x, y) => [2 * x, 20 * y], opt: [0, 0] },
  rosen: {
    name: "Curved valley (Rosenbrock-style)",
    f: (x, y) => (1 - x) ** 2 + 5 * (y - x * x) ** 2,
    g: (x, y) => [-2 * (1 - x) - 20 * x * (y - x * x), 10 * (y - x * x)],
    opt: [1, 1],
  },
  bumpy: {
    name: "Bumpy surface (local minima)",
    f: (x, y) => 0.25 * (x * x + y * y) + 0.6 * Math.sin(3 * x) * Math.sin(3 * y),
    g: (x, y) => [0.5 * x + 1.8 * Math.cos(3 * x) * Math.sin(3 * y), 0.5 * y + 1.8 * Math.sin(3 * x) * Math.cos(3 * y)],
    opt: null,
  },
};
const OPTS = { gd: "Gradient descent", mom: "Momentum (β = 0.9)", adam: "Adam" };

function niceTicks(min, max, count = 5) {
  const span = max - min;
  if (!isFinite(span) || span <= 0) return [min];
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let v = start; v <= max + step * 1e-9; v += step) ticks.push(v);
  return ticks;
}

function fmtTick(v) {
  const a = Math.abs(v);
  if (!isFinite(v)) return "—";
  if (a >= 1000) return Math.round(v).toLocaleString();
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  if (a >= 0.01) return v.toFixed(3);
  return v.toExponential(1);
}

function fmtValue(v) {
  if (!isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1000 || (a > 0 && a < 0.01)) return v.toExponential(2);
  return Number(v.toFixed(3)).toString();
}

const TASK_GROUPS = ["Synthetic surfaces", "Real datasets"];
const TASKS = [
  { g: "Synthetic surfaces", t: "Converge. On the simple bowl with gradient descent, run until the gradient is almost zero.", ok: (r) => r.mode === "synthetic" && r.fn === "bowl" && r.opt === "gd" && r.converged !== null },
  { g: "Synthetic surfaces", t: "Break it. On the narrow valley with gradient descent, find a learning rate that makes the loss blow up.", ok: (r) => r.mode === "synthetic" && r.fn === "narrow" && r.opt === "gd" && r.diverged },
  { g: "Synthetic surfaces", t: "Fix the valley. On the narrow valley, use Momentum (or Adam with a small rate) to converge where plain gradient descent oscillated.", ok: (r) => r.mode === "synthetic" && r.fn === "narrow" && r.opt !== "gd" && r.converged !== null },
  { g: "Synthetic surfaces", t: "Get trapped. On the bumpy surface with gradient descent, click two different start points and run both. Do they end at different losses?", ok: (_, h) => { const b = h.filter((r) => r.mode === "synthetic" && r.fn === "bumpy" && r.opt === "gd" && !r.diverged); return b.some((p) => b.some((q) => Math.abs(p.finalLoss - q.finalLoss) > 0.1)); } },
  { g: "Real datasets", t: "Fit real data. Pick any real dataset and use gradient descent until it converges.", ok: (r) => r.mode === "dataset" && r.opt === "gd" && r.converged !== null },
  { g: "Real datasets", t: "Compare optimizers. On the same real dataset and loss, converge with two different optimizers.", ok: (_, h) => { const runs = h.filter((r) => r.mode === "dataset" && r.converged !== null); return runs.some((a) => runs.some((b) => a.datasetId === b.datasetId && a.lossId === b.lossId && a.opt !== b.opt)); } },
  { g: "Real datasets", t: "Compare datasets. Fit two different datasets with gradient descent and the same loss, and find final standardized slopes that differ by more than 0.2.", ok: (_, h) => { const runs = h.filter((r) => r.mode === "dataset" && r.opt === "gd" && r.converged !== null); return runs.some((a) => runs.some((b) => a.datasetId !== b.datasetId && a.lossId === b.lossId && Math.abs(a.finalW - b.finalW) > 0.2)); } },
  { g: "Real datasets", t: "Compare losses. On the same real dataset, converge with MSE and with a robust loss (Huber or MAE), then compare the fitted slopes.", ok: (_, h) => { const runs = h.filter((r) => r.mode === "dataset" && r.converged !== null); const mse = runs.filter((r) => r.lossId === "mse"); const robust = runs.filter((r) => r.lossId === "huber" || r.lossId === "mae"); return mse.some((a) => robust.some((b) => a.datasetId === b.datasetId)); } },
];

const QUIZ = [
  { q: "In which direction does gradient descent move the parameters?", o: ["Along the gradient", "Opposite to the gradient", "Perpendicular to the gradient", "In a random direction"], a: 1, why: "The gradient points uphill, so we subtract it to go downhill." },
  { q: "What usually happens if the learning rate is too large?", o: ["Convergence is faster and safe", "The loss oscillates or diverges", "The gradient becomes zero", "Nothing changes"], a: 1, why: "Steps overshoot the minimum, and if they overshoot by more than they started, the loss grows." },
  { q: "What is the main drawback of a very small learning rate?", o: ["It may diverge", "It skips the minimum", "Convergence is very slow", "It needs a convex loss"], a: 2, why: "Each step is tiny, so many iterations are needed." },
  { q: "On f = x² + 10y², why does plain gradient descent zig-zag?", o: ["The surface is steep in y and shallow in x, so one rate does not suit both", "The function is not differentiable", "The gradient is zero in x", "Momentum is switched on"], a: 0, why: "A rate safe for the steep direction is slow for the shallow one." },
  { q: "How does momentum help?", o: ["It removes the learning rate", "It accumulates past gradients, damping oscillations and speeding up consistent directions", "It guarantees the global minimum", "It uses second derivatives"], a: 1, why: "The velocity term averages recent gradients, so side-to-side components cancel while the common direction builds up." },
  { q: "On a non-convex surface, the point gradient descent reaches depends mainly on:", o: ["The number of CPU cores", "The starting point (and learning rate)", "Only the loss value", "The plotting resolution"], a: 1, why: "Different starts fall into different basins, so they can reach different local minima." },
  { q: "What does Adam adapt for each parameter?", o: ["The loss function", "The step size, using running averages of gradients and squared gradients", "The number of epochs", "The dataset order"], a: 1, why: "Adam scales each update by the running average of the gradient divided by the root of its running squared average." },
];

const Eq = ({ children }) => <div className="eq">{children}</div>;

function Aim() {
  return (
    <div className="prose">
      <h2>Aim</h2>
      <p>Implement the gradient descent algorithm and study how the learning rate, the shape of the loss surface, and the choice of optimizer affect convergence. Then apply the same optimizers to fit linear models to real datasets.</p>
      <h3>Learning outcomes</h3>
      <p>After this lab you can: (1) write the gradient descent update rule and explain each term; (2) predict whether a learning rate will converge, crawl, or diverge; (3) explain why plain gradient descent struggles in narrow valleys and how momentum and Adam help; (4) explain why the starting point matters on non-convex surfaces; (5) fit a linear model to a real dataset and interpret the standardized slope.</p>
      <p>Course outcomes covered: CO1, CO2.</p>
    </div>
  );
}

function Theory() {
  return (
    <div className="prose">
      <h2>Theory</h2>
      <p>Training a model means finding parameters θ that minimise a loss f(θ). Gradient descent does this by repeatedly stepping against the gradient, which is the direction of steepest increase.</p>
      <Eq>θ(t+1) = θ(t) − η · ∇f(θ(t))</Eq>
      <p>η is the learning rate. Too small and progress is slow. Too large and the steps overshoot, so the loss oscillates or diverges. For f = x² + y² the update multiplies each coordinate by (1 − 2η), so it converges only when 0 &lt; η &lt; 1.</p>
      <h3>Ill-conditioned valleys</h3>
      <p>For f = x² + 10y² the safe learning rate is limited by the steep y direction (η &lt; 0.1), which makes progress along x slow. The path zig-zags across the valley.</p>
      <h3>Momentum</h3>
      <Eq>v(t+1) = β·v(t) + η·∇f(θ(t)){"\n"}θ(t+1) = θ(t) − v(t+1)</Eq>
      <p>The velocity v averages recent gradients. Components that flip sign cancel, and components that agree add up.</p>
      <h3>Adam</h3>
      <p>Adam keeps running averages of the gradient (m) and of its square (s), corrects their start-up bias, and divides: each parameter gets its own effective step size.</p>
      <Eq>θ ← θ − η · m̂ / (√ŝ + ε)</Eq>
      <h3>Fitting a real dataset</h3>
      <p>For a one-feature linear model ŷ = wx + b, the loss over N examples is plotted as a surface over the two parameters. The default is mean squared error:</p>
      <Eq>L(w, b) = (1/N) Σ (w·xᵢ + b − yᵢ)²</Eq>
      <p>Real-data mode standardizes x and y before building this surface, so the learning rate is not dominated by the units of the dataset. In standardized coordinates the MSE-optimal slope is the correlation between x and y, and the best intercept is zero. The scatter preview converts the current line back to the original units.</p>
      <h3>Robust losses</h3>
      <p>MSE squares every residual, so a few large errors dominate the fit. Huber loss is quadratic for small residuals but linear for large ones, and MAE uses absolute residuals. Both reduce the influence of outliers, at the cost of a different-shaped surface; MAE is not smooth at zero, so its path may chatter near the minimum. In this lab the loss selector changes the surface, the gradient, and the reported loss, while R² and RMSE remain diagnostics on the raw data.</p>
      <h3>Non-convex surfaces</h3>
      <p>With several minima, gradient descent ends in the basin it starts in. Different starting points can give different answers.</p>
    </div>
  );
}

function Procedure() {
  const steps = [
    "Open the Simulation tab and choose a data source: a synthetic loss surface or a real dataset.",
    "Synthetic mode: choose a loss surface. Real-data mode: choose a dataset and read its description.",
    "Click on the surface to place the starting point (the white ring). In real-data mode the axes are the standardized slope w and intercept b.",
    "Choose a loss function in real-data mode, then choose an optimizer, a learning rate, and the number of steps.",
    "Press Run. The path is drawn step by step, the loss curve updates, and the data preview shows the current regression line.",
    "Read the status line: it reports convergence, divergence, or the loss reached.",
    "Work through the guided tasks in the side panel. A task is ticked when your run meets it.",
    "Change one setting at a time so you can tell which one caused the change.",
    "When all tasks are done, take the Assessment.",
  ];
  return (
    <div className="prose">
      <h2>Procedure</h2>
      <ol>{steps.map((s) => <li key={s}>{s}</li>)}</ol>
    </div>
  );
}

function Simulation() {
  const [mode, setMode] = useState("synthetic");
  const [fn, setFn] = useState("bowl");
  const [datasetId, setDatasetId] = useState("iris");
  const [lossId, setLossId] = useState("mse");
  const [opt, setOpt] = useState("gd");
  const [lr, setLr] = useState(0.1);
  const [steps, setSteps] = useState(100);
  const [start, setStart] = useState([-1.5, 1.5]);
  const [run, setRun] = useState(null);
  const [history, setHistory] = useState([]);
  const [k, setK] = useState(0);
  const [play, setPlay] = useState(false);
  const surf = useRef(null), curve = useRef(null), data = useRef(null);

  const dataset = DATASET_MAP[datasetId];
  const objective = useMemo(() => {
    if (mode === "dataset") return makeDatasetObjective(dataset, lossId);
    const f = FNS[fn];
    return { ...f, kind: "synthetic", labels: { x: "x", y: "y" } };
  }, [mode, fn, dataset, lossId]);

  const heat = useMemo(() => {
    const N = 120, c = document.createElement("canvas");
    c.width = c.height = N;
    const x = c.getContext("2d"), im = x.createImageData(N, N), { f } = objective;
    const v = []; let lo = Infinity, hi = -Infinity;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const val = f(-2 + (4 * i) / (N - 1), 2 - (4 * j) / (N - 1));
      v.push(val); lo = Math.min(lo, val); hi = Math.max(hi, val);
    }
    v.forEach((val, n) => {
      const t = Math.sqrt((val - lo) / (hi - lo)), b = (t * 14) % 1 < 0.09 ? 0.85 : 1, o = n * 4;
      im.data[o] = (236 - 216 * t) * b; im.data[o + 1] = (243 - 210 * t) * b; im.data[o + 2] = (255 - 194 * t) * b; im.data[o + 3] = 255;
    });
    x.putImageData(im, 0, 0);
    return c;
  }, [objective]);

  useEffect(() => {
    if (!play || !run) return;
    const id = setInterval(() => setK((p) => { if (p >= run.path.length - 1) { setPlay(false); return p; } return p + 1; }), 50);
    return () => clearInterval(id);
  }, [play, run]);

  useEffect(() => {
    const cv = surf.current;
    if (!cv) return;
    const W = cv.width, px = (x, y) => [((x + 2) / 4) * W, ((2 - y) / 4) * W];
    const c = cv.getContext("2d");
    c.drawImage(heat, 0, 0, W, W);
    const o = objective.opt;
    if (o) { const [a, b] = px(...o); c.fillStyle = "#0f8a5f"; c.fillRect(a - 5, b - 5, 10, 10); }
    if (run) {
      const pts = run.path.slice(0, k + 1).map((p) => px(p.x, p.y));
      c.lineJoin = "round";
      [["#14213d", 5], ["#ffb400", 2.5]].forEach(([col, w]) => { c.strokeStyle = col; c.lineWidth = w; c.beginPath(); pts.forEach(([a, b], i) => (i ? c.lineTo(a, b) : c.moveTo(a, b))); c.stroke(); });
      const [a, b] = pts[pts.length - 1]; c.fillStyle = "#ffb400"; c.strokeStyle = "#14213d"; c.lineWidth = 2; c.beginPath(); c.arc(a, b, 7, 0, 7); c.fill(); c.stroke();
    }
    const [sa, sb] = px(...start); c.strokeStyle = "#fff"; c.lineWidth = 3; c.beginPath(); c.arc(sa, sb, 8, 0, 7); c.stroke();
    c.strokeStyle = "#14213d"; c.lineWidth = 1; c.beginPath(); c.arc(sa, sb, 10, 0, 7); c.stroke();
  }, [heat, run, k, start, objective]);

  useEffect(() => {
    if (mode !== "dataset" || !data.current) return;
    const cv = data.current, c = cv.getContext("2d"), W = cv.width, H = cv.height;
    const ml = 64, mr = 18, mt = 26, mb = 46;
    const pw = W - ml - mr, ph = H - mt - mb;
    c.clearRect(0, 0, W, H); c.fillStyle = "#fbfcfe"; c.fillRect(0, 0, W, H);
    c.fillStyle = "#fff"; c.fillRect(ml, mt, pw, ph);
    const pts = dataset.points;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [x, y] of pts) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const padX = (x1 - x0) * 0.04 || 1, padY = (y1 - y0) * 0.08 || 1;
    x0 -= padX; x1 += padX; y0 -= padY; y1 += padY;
    const X = (x) => ml + ((x - x0) / (x1 - x0)) * pw;
    const Y = (y) => H - mb - ((y - y0) / (y1 - y0)) * ph;
    const xt = niceTicks(x0, x1, 5), yt = niceTicks(y0, y1, 5);

    c.strokeStyle = "#e6ebf2"; c.lineWidth = 1; c.beginPath();
    xt.forEach((v) => { const x = X(v); c.moveTo(x, mt); c.lineTo(x, mt + ph); });
    yt.forEach((v) => { const y = Y(v); c.moveTo(ml, y); c.lineTo(ml + pw, y); });
    c.stroke();

    c.strokeStyle = "#c9d3de"; c.lineWidth = 1; c.strokeRect(ml, mt, pw, ph);
    c.fillStyle = "#7a879b"; c.font = "11px sans-serif";
    c.textAlign = "center"; c.textBaseline = "top";
    xt.forEach((v) => c.fillText(fmtTick(v), X(v), H - mb + 8));
    c.textAlign = "right"; c.textBaseline = "middle";
    yt.forEach((v) => c.fillText(fmtTick(v), ml - 8, Y(v)));

    c.fillStyle = "#5b6b82"; c.font = "12px sans-serif"; c.textAlign = "center"; c.textBaseline = "alphabetic";
    c.fillText(dataset.xLabel, ml + pw / 2, H - 7);
    c.save(); c.translate(16, mt + ph / 2); c.rotate(-Math.PI / 2); c.fillText(dataset.yLabel, 0, 0); c.restore();

    c.save(); c.beginPath(); c.rect(ml, mt, pw, ph); c.clip();
    const p = run ? run.path[Math.min(k, run.path.length - 1)] : { x: start[0], y: start[1] };
    const line = objective.rawLine(p.x, p.y);
    const pred = (x) => line.slope * x + line.intercept;

    c.strokeStyle = pts.length > 300 ? "rgba(198,47,59,0.10)" : "rgba(198,47,59,0.18)";
    c.lineWidth = 1; c.beginPath();
    for (const [x, y] of pts) { const px = X(x); c.moveTo(px, Y(y)); c.lineTo(px, Y(pred(x))); }
    c.stroke();

    c.beginPath();
    for (const [x, y] of pts) { const a = X(x), b = Y(y); c.moveTo(a + 2.4, b); c.arc(a, b, 2.4, 0, 7); }
    c.fillStyle = "rgba(43,78,255,0.52)"; c.fill();
    c.strokeStyle = "rgba(255,255,255,0.85)"; c.lineWidth = 0.8; c.stroke();

    c.lineJoin = "round"; c.lineCap = "round";
    c.save();
    c.shadowColor = "rgba(20,33,61,0.22)"; c.shadowBlur = 7; c.shadowOffsetY = 2;
    [["#14213d", 6], ["#ffb400", 2.5]].forEach(([col, w]) => {
      c.strokeStyle = col; c.lineWidth = w; c.beginPath();
      c.moveTo(X(x0), Y(pred(x0))); c.lineTo(X(x1), Y(pred(x1))); c.stroke();
    });
    c.restore();
    c.restore();
  }, [mode, dataset, objective, run, k, start]);

  useEffect(() => {
    const cv = curve.current, c = cv.getContext("2d"), W = cv.width, H = cv.height;
    c.clearRect(0, 0, W, H); c.fillStyle = "#fff"; c.fillRect(0, 0, W, H);
    c.strokeStyle = "#c9d3de"; c.strokeRect(40, 10, W - 50, H - 30);
    c.fillStyle = "#5b6b82"; c.font = "12px sans-serif"; c.fillText("step", W / 2, H - 4); c.fillText("loss", 4, 24);
    if (!run) return;
    const ls = run.path.map((p) => p.l), top = Math.max(ls[0], 1e-6) * 1.1, bot = Math.min(0, ...ls.slice(0, k + 1));
    const X = (i) => 40 + (i / Math.max(run.steps, 1)) * (W - 50), Y = (l) => 10 + (1 - (Math.min(l, top) - bot) / (top - bot)) * (H - 30);
    c.strokeStyle = "#2b4eff"; c.lineWidth = 2; c.beginPath();
    ls.slice(0, k + 1).forEach((l, i) => (i ? c.lineTo(X(i), Y(l)) : c.moveTo(X(i), Y(l)))); c.stroke();
  }, [run, k]);

  const clearRun = () => { setRun(null); setPlay(false); setK(0); };
  const changeMode = (next) => {
    setMode(next);
    setStart(next === "dataset" ? dataset.start : [-1.5, 1.5]);
    clearRun();
  };
  const changeDataset = (id) => {
    setDatasetId(id);
    setStart(DATASET_MAP[id].start);
    clearRun();
  };
  const pick = (e) => {
    const r = e.target.getBoundingClientRect();
    setStart([-2 + (4 * (e.clientX - r.left)) / r.width, 2 - (4 * (e.clientY - r.top)) / r.height]);
    clearRun();
  };
  const reset = () => {
    setOpt("gd");
    setLr(0.1);
    setSteps(100);
    if (mode === "dataset") {
      setDatasetId("iris");
      setLossId("mse");
      setStart(DATASET_MAP.iris.start);
    } else {
      setFn("bowl");
      setStart([-1.5, 1.5]);
    }
    clearRun();
  };
  const go = () => {
    const r = simulate(objective, opt, lr, steps, start, {
      mode,
      fn: mode === "synthetic" ? fn : null,
      datasetId: mode === "dataset" ? datasetId : null,
      lossId: mode === "dataset" ? lossId : null,
      surface: mode === "synthetic" ? fn : datasetId,
    });
    const h = [...history, r];
    setHistory(h); setRun(r); setK(0); setPlay(true);
  };
  const doneTask = (t) => history.some((r) => t.ok(r, history));
  const doneCount = TASKS.filter(doneTask).length;
  const cur = run ? run.path[Math.min(k, run.path.length - 1)] : null;
  const atEnd = run && k >= run.path.length - 1;
  const previewParams = cur || { x: start[0], y: start[1] };
  const previewLine = mode === "dataset" ? objective.rawLine(previewParams.x, previewParams.y) : null;
  const previewMetrics = mode === "dataset" ? fitMetrics(dataset.points, previewLine) : null;
  const previewLoss = mode === "dataset" ? objective.f(previewParams.x, previewParams.y) : null;
  const fitEquation = previewLine
    ? `y = ${fmtValue(previewLine.slope)}x ${previewLine.intercept >= 0 ? "+" : "−"} ${fmtValue(Math.abs(previewLine.intercept))}`
    : "";
  let status = "Place a start point, then press Run.";
  if (run) {
    status = `Step ${k}, loss ${cur.l.toExponential(2)}`;
    if (mode === "dataset") status += `, w = ${cur.x.toFixed(2)}, b = ${cur.y.toFixed(2)}`;
    if (atEnd) status += run.diverged ? ". Diverged: the loss blew up." : run.converged !== null ? `. Converged: the gradient (or subgradient) is small at step ${run.converged}.` : ". Not converged yet. Try more steps or a different rate.";
  }

  return (
    <div>
      <h2>Simulation</h2>
      <div className="sim">
        <div>
          <canvas ref={surf} width={560} height={560} onClick={pick} aria-label="Loss surface. Click to set the start point." />
          <div className="stats" aria-live="polite"><span>{status}</span></div>
          {mode === "dataset" && (
            <>
              <canvas ref={data} width={560} height={280} aria-label="Scatter plot and current regression line" />
              <div className="fit-bar">
                <span className="fit-eq">{fitEquation}</span>
                <span className="fit-stat" title="Loss computed on standardized residuals">{LOSSES[lossId].short} (std.) {fmtValue(previewLoss)}</span>
                <span className="fit-stat">R² {previewMetrics.r2.toFixed(3)}</span>
                <span className="fit-stat">RMSE {fmtValue(previewMetrics.rmse)}</span>
                <span className="fit-stat">n = {previewMetrics.n}</span>
              </div>
              <div className="legend">
                <span><i className="dot" />Data</span>
                <span><i className="line" />Current fit</span>
                <span className="legend-note">Red lines show residuals</span>
              </div>
            </>
          )}
          <canvas ref={curve} width={560} height={200} aria-label="Loss versus step" />
        </div>
        <div>
          <div className="panel">
            <h3>Settings</h3>
            <label className="ctl">Data source
              <select value={mode} onChange={(e) => changeMode(e.target.value)}>
                <option value="synthetic">Synthetic loss surface</option>
                <option value="dataset">Real dataset</option>
              </select></label>
            {mode === "synthetic" ? (
              <label className="ctl">Loss surface
                <select value={fn} onChange={(e) => { setFn(e.target.value); clearRun(); }}>{Object.entries(FNS).map(([key, v]) => <option key={key} value={key}>{v.name}</option>)}</select></label>
            ) : (
              <>
                <label className="ctl">Dataset
                  <select value={datasetId} onChange={(e) => changeDataset(e.target.value)}>{DATASETS.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}</select></label>
                <label className="ctl">Loss function
                  <select value={lossId} onChange={(e) => { setLossId(e.target.value); clearRun(); }}>{Object.entries(LOSSES).map(([key, v]) => <option key={key} value={key}>{v.name}</option>)}</select></label>
                <div className="loss-tag">{LOSSES[lossId].description}</div>
              </>
            )}
            <label className="ctl">Optimizer
              <select value={opt} onChange={(e) => { setOpt(e.target.value); clearRun(); }}>{Object.entries(OPTS).map(([key, v]) => <option key={key} value={key}>{v}</option>)}</select></label>
            <label className="ctl">Learning rate η: {lr}
              <input type="range" min="0.001" max="0.5" step="0.001" value={lr} onChange={(e) => setLr(+e.target.value)} /></label>
            <label className="ctl">Steps: {steps}
              <input type="range" min="10" max="500" step="10" value={steps} onChange={(e) => setSteps(+e.target.value)} /></label>
            <div className="row">
              <button className="btn" onClick={go}>Run</button>
              <button className="btn alt" onClick={reset}>Reset</button>
              {run && !atEnd && <button className="btn alt" onClick={() => setPlay(!play)}>{play ? "Pause" : "Resume"}</button>}
            </div>
            <small>Start: ({start[0].toFixed(2)}, {start[1].toFixed(2)}). {mode === "dataset" ? "Axes are standardized w and b. For real data, Adam often needs η ≥ 0.02 or more steps." : "Adam usually works best with η between 0.01 and 0.1."}</small>
          </div>
          {mode === "dataset" && (
            <div className="panel" style={{ marginTop: 16 }}>
              <h3>{dataset.name}</h3>
              <p className="dataset-desc">{dataset.description}</p>
              <small className="dataset-meta">{dataset.note}. Source: {dataset.source}.</small>
              <small className="dataset-meta">Surface axes and loss use standardized x/y; R², RMSE, and the preview use raw units.</small>
            </div>
          )}
          <div className="panel" style={{ marginTop: 16 }}>
            <h3>Guided tasks ({doneCount}/{TASKS.length})</h3>
            {TASK_GROUPS.map((g) => (
              <div className="task-group" key={g}>
                <h4>{g}</h4>
                <ul className="tasks">{TASKS.filter((t) => t.g === g).map((t) => <li key={t.t}><span className={doneTask(t) ? "done" : ""}>{doneTask(t) ? "✓" : "○"}</span><span>{t.t}</span></li>)}</ul>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Assessment() {
  const [ans, setAns] = useState({});
  const [sub, setSub] = useState(false);
  const score = QUIZ.filter((q, i) => ans[i] === q.a).length;
  return (
    <div className="prose">
      <h2>Assessment</h2>
      {QUIZ.map((q, i) => (
        <div className="q" key={q.q}>
          <strong>{i + 1}. {q.q}</strong>
          {q.o.map((o, j) => (
            <label key={o}><input type="radio" name={`q${i}`} disabled={sub} checked={ans[i] === j} onChange={() => setAns({ ...ans, [i]: j })} /> {o}</label>
          ))}
          {sub && <div className={`res ${ans[i] === q.a ? "ok" : "bad"}`}>{ans[i] === q.a ? "Correct. " : `Incorrect. Answer: ${q.o[q.a]}. `}{q.why}</div>}
        </div>
      ))}
      <div className="row">
        {!sub ? <button className="btn" disabled={Object.keys(ans).length < QUIZ.length} onClick={() => setSub(true)}>Submit answers</button>
          : <><div className="score">{score}/{QUIZ.length}</div><button className="btn alt" onClick={() => { setAns({}); setSub(false); }}>Try again</button></>}
      </div>
    </div>
  );
}

const TABS = [["Aim", Aim], ["Theory", Theory], ["Procedure", Procedure], ["Simulation", Simulation], ["Assessment", Assessment]];

export default function App() {
  const [tab, setTab] = useState(0);
  const Page = TABS[tab][1];
  return (
    <div>
      <header>
        <h1>Gradient Descent Virtual Lab</h1>
        <nav className="tabs">{TABS.map(([n], i) => <button key={n} aria-current={tab === i} onClick={() => setTab(i)}>{n}</button>)}</nav>
      </header>
      <main>
        <div hidden={tab !== 3}><Simulation /></div>
        {tab !== 3 && <Page />}
      </main>
    </div>
  );
}
