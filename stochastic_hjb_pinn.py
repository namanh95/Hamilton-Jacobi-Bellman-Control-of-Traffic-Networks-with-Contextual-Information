"""Two-region stochastic HJB-PINN traffic-control benchmark.

This script implements the numerical model described in the revised manuscript:

* two-region macroscopic-fundamental-diagram dynamics;
* contextual, time-dependent capacity deformation;
* a finite-horizon stochastic HJB equation;
* a PINN approximation of the value function;
* exact first- and second-order automatic differentiation;
* analytic Hamiltonian minimization for the metering policy;
* independent Latin-hypercube training, validation, and test sets;
* Euler-Maruyama closed-loop Monte Carlo evaluation;
* representative paths, ensemble means, and empirical 95% intervals.

No numerical result is hard-coded. Run the script to generate the CSV tables and
figures used in the revised Section 5.2.

Dependencies
------------
    pip install numpy pandas matplotlib scipy torch

Example
-------
    python stochastic_hjb_pinn.py --epochs 20000 --mc-paths 200

For a quick smoke test:
    python stochastic_hjb_pinn.py --epochs 200 --n-interior 2000 --mc-paths 20
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc
from torch import Tensor, nn


DTYPE = torch.float64


@dataclass(frozen=True)
class ModelConfig:
    # Physical domain and time horizon (minutes)
    n1_jam: float = 4000.0
    n2_jam: float = 4000.0
    horizon: float = 60.0

    # Nominal MFD production parameters: P_i(n)=beta_i*n*(1-n/n_jam)
    beta1: float = 0.055
    beta2: float = 0.050

    # Contextual capacity deformation P1(n,x)=P1^0(n)*(1-gamma*x)
    gamma: float = 0.55

    # External demand profile d(t)=d0+d_peak*exp(-(t-t_d)^2/(2*s_d^2))
    demand_base: float = 55.0
    demand_peak: float = 35.0
    demand_center: float = 24.0
    demand_width: float = 8.0

    # Context mean profile and mean-reverting context dynamics
    context_base: float = 0.05
    context_peak: float = 0.85
    context_center: float = 30.0
    context_width: float = 7.0
    context_reversion: float = 0.35

    # Boundary-degenerate state diffusion and contextual diffusion
    sigma1: float = 70.0
    sigma2: float = 70.0
    eta_x: float = 0.16

    # Running cost and terminal cost weights
    w1: float = 1.00
    w2: float = 0.70
    control_weight: float = 0.20
    terminal_weight: float = 1.00

    # Admissible perimeter-metering control
    u_max: float = 1.0


@dataclass(frozen=True)
class TrainConfig:
    hidden_layers: int = 4
    hidden_width: int = 64
    n_interior: int = 20000
    n_terminal: int = 2000
    n_validation: int = 5000
    n_test: int = 10000
    batch_size: int = 1024
    epochs: int = 20000
    learning_rate: float = 1e-3
    terminal_loss_weight: float = 10.0
    patience: int = 1500
    validation_every: int = 100
    seed: int = 42


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.10
    paths: int = 200
    n1_initial: float = 1700.0
    n2_initial: float = 1500.0
    x_initial: float = 0.05
    seed: int = 1234


class ValueNet(nn.Module):
    """Fully connected value-function network J_theta(n1,n2,x,t)."""

    def __init__(self, hidden_layers: int, hidden_width: int) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        in_dim = 4
        for _ in range(hidden_layers):
            layer = nn.Linear(in_dim, hidden_width, dtype=DTYPE)
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            modules.extend([layer, nn.Tanh()])
            in_dim = hidden_width
        out = nn.Linear(in_dim, 1, dtype=DTYPE)
        nn.init.xavier_normal_(out.weight)
        nn.init.zeros_(out.bias)
        modules.append(out)
        self.net = nn.Sequential(*modules)

    def forward(self, normalized_input: Tensor) -> Tensor:
        return self.net(normalized_input)


class TwoRegionHJB:
    """Model equations, SHJB residual, and feedback law."""

    def __init__(self, cfg: ModelConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.lower = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=DTYPE, device=device)
        self.upper = torch.tensor(
            [cfg.n1_jam, cfg.n2_jam, 1.0, cfg.horizon], dtype=DTYPE, device=device
        )
        self.scale = 2.0 / (self.upper - self.lower)

    def normalize(self, physical: Tensor) -> Tensor:
        return 2.0 * (physical - self.lower) / (self.upper - self.lower) - 1.0

    def context_mean(self, t: Tensor) -> Tensor:
        c = self.cfg
        pulse = c.context_peak * torch.exp(-0.5 * ((t - c.context_center) / c.context_width) ** 2)
        return torch.clamp(c.context_base + pulse, 0.0, 1.0)

    def demand(self, t: Tensor) -> Tensor:
        c = self.cfg
        return c.demand_base + c.demand_peak * torch.exp(
            -0.5 * ((t - c.demand_center) / c.demand_width) ** 2
        )

    def production1(self, n1: Tensor, x: Tensor) -> Tensor:
        c = self.cfg
        nominal = c.beta1 * n1 * (1.0 - n1 / c.n1_jam)
        return torch.clamp(nominal * (1.0 - c.gamma * x), min=0.0)

    def production2(self, n2: Tensor) -> Tensor:
        c = self.cfg
        return torch.clamp(c.beta2 * n2 * (1.0 - n2 / c.n2_jam), min=0.0)

    def state_diffusions(self, n1: Tensor, n2: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        c = self.cfg
        shape1 = (n1 / c.n1_jam) * (1.0 - n1 / c.n1_jam)
        shape2 = (n2 / c.n2_jam) * (1.0 - n2 / c.n2_jam)
        modulation = 0.25 + 0.75 * x
        return c.sigma1 * shape1 * modulation, c.sigma2 * shape2 * modulation

    def context_drift(self, x: Tensor, t: Tensor) -> Tensor:
        return self.cfg.context_reversion * (self.context_mean(t) - x)

    def context_diffusion(self, x: Tensor) -> Tensor:
        # Degenerate at x=0 and x=1, preserving the contextual interval.
        return self.cfg.eta_x * torch.sqrt(torch.clamp(x * (1.0 - x), min=0.0))

    def stage_cost(self, n1: Tensor, n2: Tensor, u: Tensor) -> Tensor:
        c = self.cfg
        q1 = n1 / c.n1_jam
        q2 = n2 / c.n2_jam
        barrier = 0.02 * ((q1 / torch.clamp(1.0 - q1, min=1e-3)) ** 2 +
                          (q2 / torch.clamp(1.0 - q2, min=1e-3)) ** 2)
        return c.w1 * q1**2 + c.w2 * q2**2 + c.control_weight * u**2 + barrier

    def terminal_cost(self, n1: Tensor, n2: Tensor) -> Tensor:
        c = self.cfg
        return c.terminal_weight * ((n1 / c.n1_jam) ** 2 + (n2 / c.n2_jam) ** 2)

    def derivatives(self, net: ValueNet, points: Tensor) -> Dict[str, Tensor]:
        points = points.clone().detach().requires_grad_(True)
        normalized = self.normalize(points)
        value = net(normalized)
        grad = torch.autograd.grad(value.sum(), points, create_graph=True)[0]

        second: list[Tensor] = []
        for idx in (0, 1, 2):
            d2 = torch.autograd.grad(grad[:, idx].sum(), points, create_graph=True)[0][:, idx:idx+1]
            second.append(d2)

        return {
            "points": points,
            "value": value,
            "jn1": grad[:, 0:1],
            "jn2": grad[:, 1:2],
            "jx": grad[:, 2:3],
            "jt": grad[:, 3:4],
            "jn1n1": second[0],
            "jn2n2": second[1],
            "jxx": second[2],
        }

    def optimal_control(self, jn1: Tensor, n2: Tensor) -> Tensor:
        """Analytic Hamiltonian minimizer for affine drift and quadratic effort.

        f1 contains +u*P2(n2), while the running cost contains r*u^2.
        Thus u_unc = -J_n1*P2/(2r), projected to [0,u_max].
        """
        p2 = self.production2(n2)
        u_unc = -(jn1 * p2) / (2.0 * self.cfg.control_weight)
        return torch.clamp(u_unc, 0.0, self.cfg.u_max)

    def residual(self, net: ValueNet, points: Tensor) -> Tuple[Tensor, Tensor]:
        d = self.derivatives(net, points)
        n1 = d["points"][:, 0:1]
        n2 = d["points"][:, 1:2]
        x = d["points"][:, 2:3]
        t = d["points"][:, 3:4]

        p1 = self.production1(n1, x)
        p2 = self.production2(n2)
        u = self.optimal_control(d["jn1"], n2)
        f1 = u * p2 - p1
        f2 = self.demand(t) - p2
        bx = self.context_drift(x, t)
        s1, s2 = self.state_diffusions(n1, n2, x)
        sx = self.context_diffusion(x)

        r = (
            d["jt"]
            + self.stage_cost(n1, n2, u)
            + d["jn1"] * f1
            + d["jn2"] * f2
            + d["jx"] * bx
            + 0.5 * s1**2 * d["jn1n1"]
            + 0.5 * s2**2 * d["jn2n2"]
            + 0.5 * sx**2 * d["jxx"]
        )
        return r, u



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lhs_points(n: int, model: TwoRegionHJB, seed: int) -> Tensor:
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    unit = sampler.random(n)
    lower = model.lower.detach().cpu().numpy()
    upper = model.upper.detach().cpu().numpy()
    physical = qmc.scale(unit, lower, upper)
    return torch.tensor(physical, dtype=DTYPE, device=model.device)


def terminal_points(n: int, model: TwoRegionHJB, seed: int) -> Tensor:
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    unit = sampler.random(n)
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([model.cfg.n1_jam, model.cfg.n2_jam, 1.0])
    xyz = qmc.scale(unit, lower, upper)
    t = np.full((n, 1), model.cfg.horizon)
    return torch.tensor(np.hstack([xyz, t]), dtype=DTYPE, device=model.device)


def batch_indices(n: int, batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def evaluate_losses(
    net: ValueNet,
    model: TwoRegionHJB,
    interior: Tensor,
    terminal: Tensor,
    terminal_weight: float,
) -> Dict[str, float]:
    net.eval()
    with torch.enable_grad():
        residual, _ = model.residual(net, interior)
        terminal_value = net(model.normalize(terminal))
        target = model.terminal_cost(terminal[:, 0:1], terminal[:, 1:2])
        residual_mse = torch.mean(residual**2)
        terminal_mse = torch.mean((terminal_value - target) ** 2)
        total = residual_mse + terminal_weight * terminal_mse
    return {
        "total": float(total.detach().cpu()),
        "residual_mse": float(residual_mse.detach().cpu()),
        "terminal_mse": float(terminal_mse.detach().cpu()),
    }


def train_pinn(
    net: ValueNet,
    model: TwoRegionHJB,
    cfg: TrainConfig,
    output_dir: Path,
) -> pd.DataFrame:
    train_i = lhs_points(cfg.n_interior, model, cfg.seed)
    train_t = terminal_points(cfg.n_terminal, model, cfg.seed + 1)
    val_i = lhs_points(cfg.n_validation, model, cfg.seed + 2)
    val_t = terminal_points(max(500, cfg.n_terminal // 2), model, cfg.seed + 3)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(5, cfg.patience // cfg.validation_every // 3), min_lr=1e-5
    )
    rng = np.random.default_rng(cfg.seed)
    best_val = math.inf
    best_epoch = 0
    best_state: Dict[str, Tensor] | None = None
    history: list[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        net.train()
        epoch_total = 0.0
        batches = 0
        terminal_order = rng.permutation(cfg.n_terminal)

        for b, idx in enumerate(batch_indices(cfg.n_interior, cfg.batch_size, rng)):
            tidx_start = (b * cfg.batch_size) % cfg.n_terminal
            tidx = terminal_order[tidx_start:tidx_start + min(cfg.batch_size, cfg.n_terminal)]
            if len(tidx) == 0:
                tidx = terminal_order[: min(cfg.batch_size, cfg.n_terminal)]

            optimizer.zero_grad(set_to_none=True)
            residual, _ = model.residual(net, train_i[idx])
            terminal_value = net(model.normalize(train_t[tidx]))
            target = model.terminal_cost(train_t[tidx, 0:1], train_t[tidx, 1:2])
            loss_r = torch.mean(residual**2)
            loss_t = torch.mean((terminal_value - target) ** 2)
            loss = loss_r + cfg.terminal_loss_weight * loss_t
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered during PINN training.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_total += float(loss.detach().cpu())
            batches += 1

        if epoch == 1 or epoch % cfg.validation_every == 0 or epoch == cfg.epochs:
            val = evaluate_losses(net, model, val_i, val_t, cfg.terminal_loss_weight)
            scheduler.step(val["total"])
            row = {
                "epoch": epoch,
                "train_batch_loss": epoch_total / max(batches, 1),
                "validation_total": val["total"],
                "validation_residual_mse": val["residual_mse"],
                "validation_terminal_mse": val["terminal_mse"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(
                f"epoch={epoch:6d} train={row['train_batch_loss']:.4e} "
                f"val={val['total']:.4e} residual={val['residual_mse']:.4e}"
            )

            if val["total"] < best_val:
                best_val = val["total"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

            if epoch - best_epoch >= cfg.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    net.load_state_dict({k: v.to(model.device) for k, v in best_state.items()})
    torch.save(net.state_dict(), output_dir / "pinn_best.pt")
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    return history_df


def learned_control_numpy(net: ValueNet, model: TwoRegionHJB, state: np.ndarray) -> float:
    point = torch.tensor(state[None, :], dtype=DTYPE, device=model.device, requires_grad=True)
    value = net(model.normalize(point))
    grad = torch.autograd.grad(value.sum(), point, create_graph=False)[0]
    n2 = point[:, 1:2]
    u = model.optimal_control(grad[:, 0:1], n2)
    return float(u.detach().cpu().item())


def profiles_numpy(t: float, cfg: ModelConfig) -> Tuple[float, float]:
    demand = cfg.demand_base + cfg.demand_peak * math.exp(
        -0.5 * ((t - cfg.demand_center) / cfg.demand_width) ** 2
    )
    context_mean = min(
        1.0,
        cfg.context_base + cfg.context_peak * math.exp(
            -0.5 * ((t - cfg.context_center) / cfg.context_width) ** 2
        ),
    )
    return demand, context_mean


def mfd_numpy(n: float, beta: float, jam: float) -> float:
    return max(beta * n * (1.0 - n / jam), 0.0)


def simulate_closed_loop(
    net: ValueNet,
    model: TwoRegionHJB,
    sim: SimulationConfig,
    policy: str,
) -> pd.DataFrame:
    """Simulate one policy over common-random-number Monte Carlo paths."""
    c = model.cfg
    steps = int(round(c.horizon / sim.dt)) + 1
    times = np.linspace(0.0, c.horizon, steps)
    rng = np.random.default_rng(sim.seed)
    eps1 = rng.standard_normal((sim.paths, steps - 1))
    eps2 = rng.standard_normal((sim.paths, steps - 1))
    epsx = rng.standard_normal((sim.paths, steps - 1))

    rows: list[dict[str, float | int | str]] = []
    for path in range(sim.paths):
        n1, n2, x = sim.n1_initial, sim.n2_initial, sim.x_initial
        for k, t in enumerate(times):
            demand, target_x = profiles_numpy(float(t), c)
            state = np.array([n1, n2, x, float(t)], dtype=float)

            if policy == "pinn_hjb":
                u = learned_control_numpy(net, model, state)
            elif policy == "reactive":
                # Restrict transfer as core accumulation approaches its critical range.
                u = float(np.clip(1.15 - 1.40 * (n1 / c.n1_jam), 0.0, c.u_max))
            elif policy == "predict_then_optimize":
                # Deterministic one-step contextual rule with a biased/noisy shock forecast.
                x_hat = float(np.clip(x + 0.08, 0.0, 1.0))
                p1_hat = max(c.beta1 * n1 * (1.0 - n1 / c.n1_jam) * (1.0 - c.gamma * x_hat), 0.0)
                p2 = mfd_numpy(n2, c.beta2, c.n2_jam)
                desired_transfer = max(p1_hat - 0.25 * n1 / c.n1_jam, 0.0)
                u = float(np.clip(desired_transfer / max(p2, 1e-9), 0.0, c.u_max))
            else:
                raise ValueError(f"Unknown policy: {policy}")

            rows.append({
                "policy": policy,
                "path": path,
                "time": float(t),
                "n1": n1,
                "n2": n2,
                "context": x,
                "context_target": target_x,
                "demand": demand,
                "control": u,
            })
            if k == steps - 1:
                continue

            p1 = max(c.beta1 * n1 * (1.0 - n1 / c.n1_jam) * (1.0 - c.gamma * x), 0.0)
            p2 = mfd_numpy(n2, c.beta2, c.n2_jam)
            f1 = u * p2 - p1
            f2 = demand - p2
            sx1 = c.sigma1 * (n1 / c.n1_jam) * (1.0 - n1 / c.n1_jam) * (0.25 + 0.75 * x)
            sx2 = c.sigma2 * (n2 / c.n2_jam) * (1.0 - n2 / c.n2_jam) * (0.25 + 0.75 * x)
            bx = c.context_reversion * (target_x - x)
            sigx = c.eta_x * math.sqrt(max(x * (1.0 - x), 0.0))

            n1 = float(np.clip(n1 + f1 * sim.dt + sx1 * math.sqrt(sim.dt) * eps1[path, k], 0.0, c.n1_jam))
            n2 = float(np.clip(n2 + f2 * sim.dt + sx2 * math.sqrt(sim.dt) * eps2[path, k], 0.0, c.n2_jam))
            x = float(np.clip(x + bx * sim.dt + sigx * math.sqrt(sim.dt) * epsx[path, k], 0.0, 1.0))

    return pd.DataFrame(rows)


def summarize_paths(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    grouped = df.groupby(["policy", "path"], as_index=False)
    rows: list[dict[str, float | str]] = []
    for (policy, path), g in grouped:
        dt = float(g["time"].iloc[1] - g["time"].iloc[0]) if len(g) > 1 else 1.0
        total_acc = g["n1"] + g["n2"]
        rows.append({
            "policy": str(policy),
            "path": int(path),
            "integrated_accumulation": float(np.trapz(total_acc, g["time"])),
            "max_core_accumulation": float(g["n1"].max()),
            "max_total_accumulation": float(total_acc.max()),
            "mean_control": float(g["control"].mean()),
            "core_jam_violation": float((g["n1"] >= 0.95 * cfg.n1_jam).any()),
            "time_step": dt,
        })
    raw = pd.DataFrame(rows)
    summary = raw.groupby("policy").agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    return summary.reset_index()


def plot_training(history: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.semilogy(history["epoch"], history["validation_residual_mse"], label="Validation SHJB residual")
    plt.semilogy(history["epoch"], history["validation_terminal_mse"], label="Terminal-condition error")
    plt.xlabel("Epoch")
    plt.ylabel("Mean-squared error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def plot_monte_carlo(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    for policy, g in df.groupby("policy"):
        stats = g.groupby("time")["n1"].quantile([0.025, 0.5, 0.975]).unstack()
        plt.plot(stats.index, stats[0.5], label=f"{policy} median")
        plt.fill_between(stats.index, stats[0.025], stats[0.975], alpha=0.16)
    plt.xlabel("Time (min)")
    plt.ylabel("Region 1 accumulation (vehicles)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--n-interior", type=int, default=20000)
    parser.add_argument("--n-terminal", type=int, default=2000)
    parser.add_argument("--n-validation", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--mc-paths", type=int, default=200)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        epochs=args.epochs,
        n_interior=args.n_interior,
        n_terminal=args.n_terminal,
        n_validation=args.n_validation,
        n_test=args.n_test,
    )
    sim_cfg = SimulationConfig(paths=args.mc_paths)
    set_seed(train_cfg.seed)

    model = TwoRegionHJB(model_cfg, device)
    net = ValueNet(train_cfg.hidden_layers, train_cfg.hidden_width).to(device)
    history = train_pinn(net, model, train_cfg, args.output)

    # Independent test-domain diagnostics.
    test_i = lhs_points(train_cfg.n_test, model, train_cfg.seed + 100)
    test_t = terminal_points(max(1000, train_cfg.n_terminal), model, train_cfg.seed + 101)
    test_metrics = evaluate_losses(net, model, test_i, test_t, train_cfg.terminal_loss_weight)
    pd.DataFrame([test_metrics]).to_csv(args.output / "test_metrics.csv", index=False)

    all_paths = []
    for policy in ("pinn_hjb", "reactive", "predict_then_optimize"):
        all_paths.append(simulate_closed_loop(net, model, sim_cfg, policy))
    trajectories = pd.concat(all_paths, ignore_index=True)
    trajectories.to_csv(args.output / "monte_carlo_trajectories.csv", index=False)
    summary = summarize_paths(trajectories, model_cfg)
    summary.to_csv(args.output / "policy_summary.csv", index=False)

    plot_training(history, args.output / "figure_training_convergence.png")
    plot_monte_carlo(trajectories, args.output / "figure_stochastic_accumulation.png")

    metadata = {
        "model": asdict(model_cfg),
        "training": asdict(train_cfg),
        "simulation": asdict(sim_cfg),
        "device": str(device),
        "test_metrics": test_metrics,
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nIndependent test metrics:", test_metrics)
    print("\nPolicy summary:\n", summary.to_string(index=False))
    print(f"\nOutputs written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
