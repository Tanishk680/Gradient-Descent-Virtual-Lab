# Gradient Descent Virtual Lab

An interactive React + Vite virtual lab for understanding gradient descent and fitting linear models to real datasets.

## Features

### Synthetic surfaces
- **Simple bowl** (`f = x² + y²`) — convex, easy to converge
- **Narrow valley** (`f = x² + 10y²`) — ill-conditioned, shows zig-zagging
- **Curved valley (Rosenbrock-style)** — curved optimum
- **Bumpy surface** — non-convex, shows local minima dependence on start point

### Optimizers
- **Gradient Descent (GD)**
- **Momentum** (β = 0.9)
- **Adam** (adaptive moments)

### Real datasets (linear regression)
Six bundled datasets fit in standardized `(w, b)` space. The default loss is **mean squared error (MSE)**; **Huber** and **MAE** are also selectable for robust fitting:

| Dataset | X → Y | Notes |
| --- | --- | --- |
| Iris flowers | Petal length (cm) → Petal width (cm) | Strong correlation, 150 flowers |
| Old Faithful geyser | Eruption duration (min) → Waiting time (min) | Two clusters, rough linear trend |
| Auto MPG | Horsepower → Fuel economy (MPG) | More horsepower → lower MPG |
| Diabetes progression | BMI → Disease progression | 442 patients |
| Palmer penguins | Flipper length (mm) → Body mass (g) | Positive trend, species variation |
| Restaurant tips | Total bill ($) → Tip ($) | Noisy linear trend |

In dataset mode you can choose MSE, Huber, or MAE, see the corresponding loss surface over standardized slope/intercept, click to place a start point, run optimizers, and watch an annotated scatter plot update in original units. The scatter includes gridlines, tick labels, residual lines, a current-fit line, and live equation/standardized-loss/R²/RMSE metrics.

MSE is the default. Huber is smooth and robust to outliers. MAE is also robust but non-smooth at zero, so its optimisation path can chatter near the minimum.

### Guided tasks
Tasks are grouped into **Synthetic surfaces** and **Real datasets**:
- Converge on the bowl with GD
- Break the narrow valley with GD
- Fix the valley with Momentum/Adam
- Get trapped on the bumpy surface (two starts)
- Fit a real dataset with GD until converged
- Compare two optimizers on the same dataset and loss
- Compare two datasets by final standardized slopes
- Compare MSE and a robust loss (Huber or MAE) on the same dataset

### Assessment
A 7-question multiple-choice quiz covering direction of GD, learning rate effects, zig-zagging, momentum, Adam, non-convexity, and start points.

## Tech Stack
- React 18, React DOM
- Vite 5 with `@vitejs/plugin-react`
- HTML5 Canvas 2D (no chart libraries)
- Plain CSS

## Getting Started

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # production build
npm run preview  # preview production build
```

## Project Structure
```
src/
├── App.jsx           # All UI, simulation, tasks, quiz
├── main.jsx          # Entry point
├── styles.css        # Styles
├── data/
│   └── datasets.js   # Bundled datasets + metadata
└── lib/
    └── optimize.js   # Optimizer loop + MSE/Huber/MAE dataset objectives
```

## Notes
- Dataset gradients are computed analytically for MSE, Huber, and MAE. Surfaces are sampled on a 120×120 grid for the heatmap.
- MSE uses its closed-form optimum `[r, 0]`; robust losses use a small internal numerical search for the green optimum marker.
- The convergence threshold is loss-specific because MAE's subgradient does not vanish at the minimum.
- Adam often needs a larger learning rate (e.g. ≥ 0.02) and/or more steps on real datasets than on synthetic bowls.
- Tab switching keeps the simulation mounted so runs and task history persist.
- All data is bundled offline — no network calls required.
