export const LOSSES = {
  mse: {
    name: "Mean squared error (MSE)",
    short: "MSE",
    tol: 0.01,
    description: "Squares every residual, so large errors dominate. Sensitive to outliers.",
    loss: (e) => e * e,
    d: (e) => 2 * e,
  },
  huber: {
    name: "Huber loss (δ = 1)",
    short: "Huber",
    tol: 0.01,
    description: "Quadratic for small residuals and linear for large ones. Smooth and robust to outliers.",
    loss: (e) => { const a = Math.abs(e); return a <= 1 ? 0.5 * e * e : a - 0.5; },
    d: (e) => { const a = Math.abs(e); return a <= 1 ? e : Math.sign(e); },
  },
  mae: {
    name: "Mean absolute error (MAE)",
    short: "MAE",
    tol: 0.07,
    description: "Uses absolute residuals, making it robust to outliers but non-smooth at zero.",
    loss: (e) => Math.abs(e),
    d: (e) => Math.sign(e),
  },
};

function approximateOptimum(z, loss) {
  const starts = [[0, 0], [0.5, 0], [-0.5, 0]];
  let best = [0, 0], bestVal = Infinity;
  for (const [w0, b0] of starts) {
    let w = w0, b = b0, mw = 0, mb = 0, vw = 0, vb = 0;
    for (let t = 1; t <= 300; t++) {
      let gw = 0, gb = 0;
      for (const [x, y] of z) { const d = loss.d(w * x + b - y); gw += d * x; gb += d; }
      gw /= z.length; gb /= z.length;
      mw = 0.9 * mw + 0.1 * gw; mb = 0.9 * mb + 0.1 * gb;
      vw = 0.999 * vw + 0.001 * gw * gw; vb = 0.999 * vb + 0.001 * gb * gb;
      const b1 = 1 - 0.9 ** t, b2 = 1 - 0.999 ** t;
      w -= 0.05 * (mw / b1) / (Math.sqrt(vw / b2) + 1e-8);
      b -= 0.05 * (mb / b1) / (Math.sqrt(vb / b2) + 1e-8);
    }
    let val = 0;
    for (const [x, y] of z) val += loss.loss(w * x + b - y);
    val /= z.length;
    if (val < bestVal) { bestVal = val; best = [w, b]; }
  }
  return best;
}

export function makeDatasetObjective(dataset, lossId = "mse") {
  const loss = LOSSES[lossId] || LOSSES.mse;
  const pts = dataset.points;
  const n = pts.length;
  let sx = 0, sy = 0;
  for (const [x, y] of pts) { sx += x; sy += y; }
  const mx = sx / n, my = sy / n;
  let vx = 0, vy = 0, cxy = 0;
  for (const [x, y] of pts) {
    const dx = x - mx, dy = y - my;
    vx += dx * dx; vy += dy * dy; cxy += dx * dy;
  }
  const sdx = Math.sqrt(vx / n) || 1;
  const sdy = Math.sqrt(vy / n) || 1;
  const z = pts.map(([x, y]) => [(x - mx) / sdx, (y - my) / sdy]);
  const r = cxy / (Math.sqrt(vx * vy) || 1);

  const f = (w, b) => {
    let s = 0;
    for (const [x, y] of z) s += loss.loss(w * x + b - y);
    return s / n;
  };
  const g = (w, b) => {
    let gw = 0, gb = 0;
    for (const [x, y] of z) {
      const d = loss.d(w * x + b - y);
      gw += d * x;
      gb += d;
    }
    return [gw / n, gb / n];
  };
  const rawLine = (w, b) => {
    const slope = (w * sdy) / sdx;
    return { slope, intercept: my + sdy * b - slope * mx };
  };

  return {
    kind: "dataset",
    dataset,
    lossId,
    tol: loss.tol,
    f,
    g,
    opt: lossId === "mse" ? [r, 0] : approximateOptimum(z, loss),
    labels: { x: "w", y: "b" },
    rawLine,
    stats: { n, r, mx, my, sdx, sdy },
  };
}

export function fitMetrics(points, line) {
  const n = points.length;
  let sy = 0;
  for (const [, y] of points) sy += y;
  const meanY = sy / n;
  let sse = 0, sst = 0;
  for (const [x, y] of points) {
    const e = y - (line.slope * x + line.intercept);
    sse += e * e;
    sst += (y - meanY) ** 2;
  }
  return {
    n,
    rmse: Math.sqrt(sse / n),
    r2: sst ? 1 - sse / sst : 1,
  };
}

export function simulate(objective, opt, lr, steps, start, meta = {}) {
  const { f, g } = objective;
  const tol = objective.tol ?? 0.01;
  let [x, y] = start, vx = 0, vy = 0, mx = 0, my = 0, sx = 0, sy = 0;
  const path = [{ x, y, l: f(x, y) }];
  let converged = null, diverged = false;
  for (let t = 1; t <= steps; t++) {
    const [gx, gy] = g(x, y);
    if (Math.hypot(gx, gy) < tol && converged === null) converged = t - 1;
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
  const last = path[path.length - 1];
  return {
    path,
    converged,
    diverged,
    finalLoss: last.l,
    finalW: last.x,
    finalB: last.y,
    ...meta,
    opt,
    lr,
    steps,
  };
}
