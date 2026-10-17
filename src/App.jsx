import { useEffect, useMemo, useRef, useState } from "react";

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

function simulate(fn, opt, lr, steps, start) {
  const { f, g } = FNS[fn];
  let [x, y] = start, vx = 0, vy = 0, mx = 0, my = 0, sx = 0, sy = 0;
  const path = [{ x, y, l: f(x, y) }];
  let converged = null, diverged = false;
  for (let t = 1; t <= steps; t++) {
    const [gx, gy] = g(x, y);
    if (Math.hypot(gx, gy) < 0.01 && converged === null) converged = t - 1;
    let dx, dy;
    if (opt === "gd") { dx = lr * gx; dy = lr * gy; }
    else if (opt === "mom") { vx = 0.9 * vx + lr * gx; vy = 0.9 * vy + lr * gy; dx = vx; dy = vy; }
    else {
      mx = 0.9 * mx + 0.1 * gx; my = 0.9 * my + 0.1 * gy;
      sx = 0.999 * sx + 0.001 * gx * gx; sy = 0.999 * sy + 0.001 * gy * gy;
      const b1 = 1 - 0.9 ** t, b2 = 1 - 0.999 ** t;
      dx = (lr * (mx / b1)) / (Math.sqrt(sx / b2) + 1e-8);
      dy = (lr * (my / b1)) / (Math.sqrt(sy / b2) + 1e-8);
    }
    x -= dx; y -= dy;
    const l = f(x, y);
    path.push({ x, y, l });
    if (!isFinite(l) || Math.abs(x) > 1e3 || Math.abs(y) > 1e3) { diverged = true; break; }
  }
  return { path, converged, diverged, fn, opt, lr, steps, finalLoss: path[path.length - 1].l };
}

const TASKS = [
  { t: "Make it converge. On the simple bowl with gradient descent, run until the gradient is almost zero.", ok: (r) => r.fn === "bowl" && r.opt === "gd" && r.converged !== null },
  { t: "Break it. On the narrow valley with gradient descent, find a learning rate that makes the loss blow up.", ok: (r) => r.fn === "narrow" && r.opt === "gd" && r.diverged },
  { t: "Watch it crawl. On the simple bowl, use a learning rate of 0.01 or less and 100 steps or fewer. Is the loss still above 0.01?", ok: (r) => r.fn === "bowl" && r.opt === "gd" && r.lr <= 0.01 && r.steps <= 100 && r.finalLoss > 0.01 },
  { t: "Fix the valley. On the narrow valley, use Momentum (or Adam with a small rate) to converge where plain gradient descent oscillated.", ok: (r) => r.fn === "narrow" && r.opt !== "gd" && r.converged !== null },
  { t: "Get trapped. On the bumpy surface with gradient descent, click two different start points and run both. Do they end at different losses?", ok: (_, h) => { const b = h.filter((r) => r.fn === "bumpy" && r.opt === "gd" && !r.diverged); return b.some((p) => b.some((q) => Math.abs(p.finalLoss - q.finalLoss) > 0.1)); } },
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
      <p>Implement the gradient descent algorithm and study how the learning rate, the shape of the loss surface, and the choice of optimizer affect convergence.</p>
      <h3>Learning outcomes</h3>
      <p>After this lab you can: (1) write the gradient descent update rule and explain each term; (2) predict whether a learning rate will converge, crawl, or diverge; (3) explain why plain gradient descent struggles in narrow valleys and how momentum and Adam help; (4) explain why the starting point matters on non-convex surfaces.</p>
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
      <h3>Non-convex surfaces</h3>
      <p>With several minima, gradient descent ends in the basin it starts in. Different starting points can give different answers.</p>
    </div>
  );
}

function Procedure() {
  const steps = [
    "Open the Simulation tab and choose a loss surface.",
    "Click on the surface to place the starting point (the white ring).",
    "Choose an optimizer, a learning rate, and the number of steps.",
    "Press Run. The path is drawn step by step, and the loss curve updates beneath it.",
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
  const [fn, setFn] = useState("bowl");
  const [opt, setOpt] = useState("gd");
  const [lr, setLr] = useState(0.1);
  const [steps, setSteps] = useState(100);
  const [start, setStart] = useState([-1.5, 1.5]);
  const [run, setRun] = useState(null);
  const [history, setHistory] = useState([]);
  const [k, setK] = useState(0);
  const [play, setPlay] = useState(false);
  const surf = useRef(null), curve = useRef(null);

  const heat = useMemo(() => {
    const N = 120, c = document.createElement("canvas");
    c.width = c.height = N;
    const x = c.getContext("2d"), im = x.createImageData(N, N), { f } = FNS[fn];
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
  }, [fn]);

  useEffect(() => {
    if (!play || !run) return;
    const id = setInterval(() => setK((p) => { if (p >= run.path.length - 1) { setPlay(false); return p; } return p + 1; }), 50);
    return () => clearInterval(id);
  }, [play, run]);

  useEffect(() => {
    const cv = surf.current, W = cv.width, px = (x, y) => [((x + 2) / 4) * W, ((2 - y) / 4) * W];
    const c = cv.getContext("2d");
    c.drawImage(heat, 0, 0, W, W);
    const o = FNS[fn].opt;
    if (o) { const [a, b] = px(...o); c.fillStyle = "#0f8a5f"; c.fillRect(a - 5, b - 5, 10, 10); }
    if (run) {
      const pts = run.path.slice(0, k + 1).map((p) => px(p.x, p.y));
      c.lineJoin = "round";
      [["#14213d", 5], ["#ffb400", 2.5]].forEach(([col, w]) => { c.strokeStyle = col; c.lineWidth = w; c.beginPath(); pts.forEach(([a, b], i) => (i ? c.lineTo(a, b) : c.moveTo(a, b))); c.stroke(); });
      const [a, b] = pts[pts.length - 1]; c.fillStyle = "#ffb400"; c.strokeStyle = "#14213d"; c.lineWidth = 2; c.beginPath(); c.arc(a, b, 7, 0, 7); c.fill(); c.stroke();
    }
    const [sa, sb] = px(...start); c.strokeStyle = "#fff"; c.lineWidth = 3; c.beginPath(); c.arc(sa, sb, 8, 0, 7); c.stroke();
    c.strokeStyle = "#14213d"; c.lineWidth = 1; c.beginPath(); c.arc(sa, sb, 10, 0, 7); c.stroke();
  }, [heat, run, k, start, fn]);

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

  const pick = (e) => {
    const r = e.target.getBoundingClientRect();
    setStart([-2 + (4 * (e.clientX - r.left)) / r.width, 2 - (4 * (e.clientY - r.top)) / r.height]);
    setRun(null); setPlay(false); setK(0);
  };
  const reset = () => { setRun(null); setPlay(false); setK(0); };
  const go = () => {
    const r = simulate(fn, opt, lr, steps, start), h = [...history, r];
    setHistory(h); setRun(r); setK(0); setPlay(true);
  };
  const doneTask = (t) => history.some((r) => t.ok(r, history));
  const cur = run ? run.path[Math.min(k, run.path.length - 1)] : null;
  const atEnd = run && k >= run.path.length - 1;
  let status = "Place a start point, then press Run.";
  if (run) {
    status = `Step ${k}, loss ${cur.l.toExponential(2)}`;
    if (atEnd) status += run.diverged ? ". Diverged: the loss blew up." : run.converged !== null ? `. Converged: gradient near zero at step ${run.converged}.` : ". Not converged yet. Try more steps or a different rate.";
  }

  return (
    <div>
      <h2>Simulation</h2>
      <div className="sim">
        <div>
          <canvas ref={surf} width={560} height={560} onClick={pick} aria-label="Loss surface. Click to set the start point." />
          <div className="stats" aria-live="polite"><span>{status}</span></div>
          <canvas ref={curve} width={560} height={200} aria-label="Loss versus step" />
        </div>
        <div>
          <div className="panel">
            <h3>Settings</h3>
            <label className="ctl">Loss surface
              <select value={fn} onChange={(e) => { setFn(e.target.value); reset(); }}>{Object.entries(FNS).map(([key, v]) => <option key={key} value={key}>{v.name}</option>)}</select></label>
            <label className="ctl">Optimizer
              <select value={opt} onChange={(e) => { setOpt(e.target.value); reset(); }}>{Object.entries(OPTS).map(([key, v]) => <option key={key} value={key}>{v}</option>)}</select></label>
            <label className="ctl">Learning rate η: {lr}
              <input type="range" min="0.001" max="0.5" step="0.001" value={lr} onChange={(e) => setLr(+e.target.value)} /></label>
            <label className="ctl">Steps: {steps}
              <input type="range" min="10" max="500" step="10" value={steps} onChange={(e) => setSteps(+e.target.value)} /></label>
            <div className="row">
              <button className="btn" onClick={go}>Run</button>
              <button className="btn alt" onClick={reset}>Reset</button>
              {run && !atEnd && <button className="btn alt" onClick={() => setPlay(!play)}>{play ? "Pause" : "Resume"}</button>}
            </div>
            <small>Start: ({start[0].toFixed(2)}, {start[1].toFixed(2)}). Adam usually works best with η between 0.01 and 0.1.</small>
          </div>
          <div className="panel" style={{ marginTop: 16 }}>
            <h3>Guided tasks ({TASKS.filter(doneTask).length}/{TASKS.length})</h3>
            <ul className="tasks">{TASKS.map((t) => <li key={t.t}><span className={doneTask(t) ? "done" : ""}>{doneTask(t) ? "✓" : "○"}</span><span>{t.t}</span></li>)}</ul>
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
