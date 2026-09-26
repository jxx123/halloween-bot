"""System identification of the SO101 servo model against real recordings.

Data: a LeRobot v3 dataset (teleop: `observation.state` is read right before `action` is sent,
so action[t] is answered by state[t+1]) or trajectory JSON from `sysid_collect.py`
({"sequences": [{"name", "hz", "trace": [{t, present, cmd}, ..., {t, present, cmd: None}]}]}).

Replay mirrors the real follower: each tick the goal is clamped to present ± MAX_RELATIVE_TARGET,
bounded to the key's range, written to ctrl, and physics advances one control period. Sim time
stays locked to the control clock (at 30 Hz with a 5 ms timestep that is 7, 7, 6 steps per
tick), matching the wall-clock-paced engine rather than a fixed round(1/(hz*dt)) = 7 steps.

Fit: per motor, log of kp, kv, damping, frictionloss, armature, forcerange (36 values), shared by
both arms, jointly by scipy least_squares (2-point Jacobian, x_scale="jac") on per-key normalized
residuals over the first 80% of each trace. Gripper ticks where an object holds the jaws open are
masked (the fit and the RMSE use a props-free model).
Motors whose commands never move by EXCITE_MIN units are left at their initial values.

    python -m halloween_bot.sim.sysid fit --dataset ~/so101_bimanual_test_dataset \
        [--traces traces.json] --out halloween_bot/sim/params.json --report docs/sysid
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .calib import KEYS, MOTORS, Calibration, joint_name, split_key
from .engine import MAX_RELATIVE_TARGET
from .model import PARAMS_FILE, apply_params, build_model

# forcerange is fitted too (the spec's list): on the teleop episode it cuts the training cost 35 -> 11 vs the
# plan's five fields at the same settings, and is what lets the gripper match (holdout RMSE 5.8 -> 0.9, docs/sysid).
FIELDS = ("kp", "kv", "damping", "frictionloss", "armature", "forcerange")
LOG_BOUNDS = (math.log(1e-4), math.log(2e3))
HOLDOUT = 0.2
EXCITE_MIN = 5.0  # units of commanded range for a key to count as excited
SCALE_FLOOR = 2.0  # units; residual normalizer floor so near-still joints don't dominate
CONTACT_GAP = 3.0  # gripper held this far above its command while still -> object in the jaws
DIFF_STEP = 3e-3  # relative finite-difference step in log space (the cost is non-smooth below ~1e-3)
MULTISTART = (1e-3, 3e-3, 1e-2)  # CLI: one fit per step (parallel), keep the lowest training cost ...
POLISH_STEP = 1e-2  # ... then continue it with this step


@dataclass
class Trace:
    name: str
    present: np.ndarray  # [T+1, 12] measured positions, KEYS order; present[t+1] answers cmd[t]
    cmd: np.ndarray  # [T, 12] commanded positions
    hz: float


# ---- loading -------------------------------------------------------------------------
def load_dataset_traces(root) -> list[Trace]:
    import pandas as pd

    root = Path(root).expanduser()
    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data/chunk-*/file-*.parquet under {root}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    info = root / "meta" / "info.json"
    hz = float(json.loads(info.read_text()).get("fps", 30)) if info.exists() else 30.0
    order = [c for c in ("episode_index", "frame_index") if c in df.columns]
    if order:
        df = df.sort_values(order, kind="stable")
    traces = []
    for ep, g in df.groupby("episode_index", sort=True) if "episode_index" in df.columns else [(0, df)]:
        present = np.stack(g["observation.state"].to_numpy()).astype(float)
        action = np.stack(g["action"].to_numpy()).astype(float)
        traces.append(Trace(f"{root.name}/ep{int(ep)}", present, action[:-1], hz))
    return traces


def _vec(v, fallback: np.ndarray) -> np.ndarray:
    if isinstance(v, dict):
        return np.array([float(v.get(k, fallback[i])) for i, k in enumerate(KEYS)])
    return np.asarray(v, float)


def load_trajectory_traces(path) -> list[Trace]:
    doc = json.loads(Path(path).expanduser().read_text())
    seqs = doc["sequences"] if "sequences" in doc else [{"name": Path(path).stem, **doc}]
    traces = []
    for i, seq in enumerate(seqs):
        entries = seq["trace"]
        present = np.array([_vec(e["present"], np.zeros(len(KEYS))) for e in entries])
        cmds, last = [], present[0]
        for e in entries[:-1]:
            last = _vec(e["cmd"], last)  # partial cmd dicts hold the previous command
            cmds.append(last)
        traces.append(Trace(seq.get("name", f"seq{i}"), present, np.array(cmds).reshape(-1, len(KEYS)),
                            float(seq.get("hz", 30.0))))
    return traces


# ---- replay --------------------------------------------------------------------------
def _io(model: mujoco.MjModel, calib: Calibration):
    qadr = np.array([model.joint(joint_name(k)).qposadr[0] for k in KEYS])
    act = np.array([model.actuator(joint_name(k)).id for k in KEYS])
    a = np.array([calib.affine(k)[0] for k in KEYS])
    b = np.array([calib.affine(k)[1] for k in KEYS])
    lo = np.array([0.0 if k.endswith("gripper.pos") else -100.0 for k in KEYS])
    return qadr, act, a, b, lo, np.full(len(KEYS), 100.0)


def replay(model: mujoco.MjModel, calib: Calibration, trace: Trace) -> np.ndarray:
    """Drive the sim with trace.cmd from trace.present[0]; returns predicted present[1:] [T, 12]."""
    qadr, act, a, b, lo, hi = _io(model, calib)
    d = mujoco.MjData(model)
    p0 = np.asarray(trace.present[0], float)
    d.qpos[qadr] = a * p0 + b
    d.ctrl[act] = a * p0 + b
    mujoco.mj_forward(model, d)
    steps_per_tick = 1.0 / (trace.hz * model.opt.timestep)
    out = np.empty((len(trace.cmd), len(KEYS)))
    done = 0
    for t, c in enumerate(trace.cmd):
        present = (d.qpos[qadr] - b) / a
        goal = np.clip(np.clip(c, present - MAX_RELATIVE_TARGET, present + MAX_RELATIVE_TARGET), lo, hi)
        d.ctrl[act] = a * goal + b
        n = round((t + 1) * steps_per_tick) - done
        if n > 0:
            mujoco.mj_step(model, d, nstep=n)
            done += n
        out[t] = (d.qpos[qadr] - b) / a
    return out


# ---- scoring -------------------------------------------------------------------------
def valid_mask(trace: Trace) -> np.ndarray:
    """[T, 12] bool: False where a gripper is held open by an object (the no-props sim can't)."""
    ok = np.ones(trace.cmd.shape, bool)
    nxt, prev = trace.present[1:], trace.present[:-1]
    for i, k in enumerate(KEYS):
        if k.endswith("gripper.pos"):
            ok[:, i] = ~((nxt[:, i] - trace.cmd[:, i] > CONTACT_GAP) & (np.abs(nxt[:, i] - prev[:, i]) < 0.3))
    return ok


def _split(n: int, holdout: float) -> int:
    return int(round(n * (1.0 - holdout)))


def excitation(traces: list[Trace], holdout: float = HOLDOUT) -> dict[str, float]:
    """Commanded range (max - min) per key over the training part of all traces."""
    rng = {k: 0.0 for k in KEYS}
    for tr in traces:
        c = tr.cmd[:_split(len(tr.cmd), holdout)]
        if len(c):
            for i, k in enumerate(KEYS):
                rng[k] = max(rng[k], float(c[:, i].max() - c[:, i].min()))
    return rng


def _window_rmse(traces: list[Trace], preds: list[np.ndarray], holdout: float, train: bool) -> dict[str, float]:
    sq, n = np.zeros(len(KEYS)), np.zeros(len(KEYS))
    for tr, pred in zip(traces, preds):
        s = _split(len(tr.cmd), holdout)
        win = slice(0, s) if train else slice(s, len(tr.cmd))
        m = valid_mask(tr)[win]
        err = np.where(m, pred[win] - tr.present[1:][win], 0.0)
        sq += (err ** 2).sum(0)
        n += m.sum(0)
    return {k: float(math.sqrt(sq[i] / n[i])) if n[i] else float("nan") for i, k in enumerate(KEYS)}


def rmse(model: mujoco.MjModel, calib: Calibration, traces: list[Trace], holdout: float = HOLDOUT,
         preds: list[np.ndarray] | None = None) -> dict[str, float]:
    """Per-key RMSE (normalized units) of the open-loop replay over the last `holdout` of each trace."""
    preds = preds if preds is not None else [replay(model, calib, tr) for tr in traces]
    return _window_rmse(traces, preds, holdout, train=False)


# ---- fitting -------------------------------------------------------------------------
def model_params(model: mujoco.MjModel, motors=MOTORS) -> dict:
    """Current servo parameters of the (shared) right-arm actuators/joints as a params dict."""
    out = {}
    for m in motors:
        a = model.actuator(f"right_{m}")
        dof = model.joint(f"right_{m}").dofadr[0]
        out[m] = {"kp": float(a.gainprm[0]), "kv": float(-a.biasprm[2]), "damping": float(model.dof_damping[dof]),
                  "frictionloss": float(model.dof_frictionloss[dof]), "armature": float(model.dof_armature[dof]),
                  "forcerange": float(a.forcerange[1])}
    return {"motors": out}


def fit(traces: list[Trace], init: dict | None = None, motors=MOTORS, fields=FIELDS,
        calib: Calibration | None = None, holdout: float = HOLDOUT, max_nfev: int | None = None,
        diff_step: float = DIFF_STEP, verbose: int = 0) -> dict:
    """Least-squares fit of per-motor servo params (shared by both arms). Returns a params dict."""
    from scipy.optimize import least_squares

    calib = calib or Calibration.load()
    model = build_model(params={}, calib=calib, props=False)
    apply_params(model, init or {})
    start = model_params(model)["motors"]

    exc = excitation(traces, holdout)
    fitted = [m for m in motors if max(exc[f"{s}_{m}.pos"] for s in ("left", "right")) >= EXCITE_MIN]
    keys = [i for i, k in enumerate(KEYS) if split_key(k)[1] in fitted and exc[k] >= EXCITE_MIN]
    if not fitted:
        raise ValueError(f"no motor is excited by >= {EXCITE_MIN} units of command")

    windows = []
    for tr in traces:
        s = _split(len(tr.cmd), holdout)
        real = tr.present[1:s + 1]
        scale = np.maximum(real.std(0), SCALE_FLOOR) if s else np.ones(len(KEYS))
        windows.append((s, real, valid_mask(tr)[:s], scale))

    def unpack(x):
        return {"motors": {m: {f: float(math.exp(x[i * len(fields) + j])) for j, f in enumerate(fields)}
                           for i, m in enumerate(fitted)}}

    def residuals(x):
        apply_params(model, unpack(x))
        res = []
        for tr, (s, real, ok, scale) in zip(traces, windows):
            if not s:
                continue
            short = Trace(tr.name, tr.present[:s + 1], tr.cmd[:s], tr.hz)
            err = np.where(ok, replay(model, calib, short) - real, 0.0) / scale
            res.append(err[:, keys].ravel())
        return np.concatenate(res)

    x0 = np.clip([math.log(start[m][f]) for m in fitted for f in fields], LOG_BOUNDS[0] + 1e-9, LOG_BOUNDS[1] - 1e-9)
    t0 = time.perf_counter()
    r0 = residuals(x0)
    sol = least_squares(residuals, x0, x_scale="jac", bounds=LOG_BOUNDS, diff_step=diff_step,
                        max_nfev=max_nfev, verbose=verbose)
    motors_out = {m: dict(p) for m, p in (init or {}).get("motors", {}).items()}
    for m, p in unpack(sol.x)["motors"].items():
        motors_out.setdefault(m, {}).update({k: round(v, 6) for k, v in p.items()})
    sv = np.linalg.svd(sol.jac, compute_uv=False)
    return {"motors": motors_out,
            "fit": {"traces": [tr.name for tr in traces], "fitted_motors": fitted, "fields": list(fields),
                    "residual_keys": [KEYS[i] for i in keys], "excitation": {k: round(v, 2) for k, v in exc.items()},
                    "cost_init": float(0.5 * r0 @ r0), "cost": float(sol.cost), "nfev": int(sol.nfev),
                    "diff_step": diff_step,
                    "status": int(sol.status), "message": sol.message,
                    "jac_condition": float(sv.max() / max(sv.min(), 1e-300)),
                    "seconds": round(time.perf_counter() - t0, 1)}}


def _fit_job(kw: dict) -> dict:
    return fit(**kw)


def fit_multistart(traces: list[Trace], diff_steps=MULTISTART, polish: float | None = POLISH_STEP,
                   init: dict | None = None, **kw) -> dict:
    """fit() once per finite-difference step in parallel processes (the least-squares landscape is
    rugged, so the step decides which local minimum it stalls in), keep the lowest TRAINING cost,
    then continue that one with the `polish` step. Holdout data never influences the choice."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    t0 = time.perf_counter()
    jobs = [dict(traces=traces, init=init, diff_step=ds, **kw) for ds in diff_steps]
    if len(jobs) == 1:
        runs = [fit(**jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=mp.get_context("spawn")) as ex:
            runs = list(ex.map(_fit_job, jobs))
    best = min(runs, key=lambda r: r["fit"]["cost"])
    out = fit(traces, init={"motors": best["motors"]}, diff_step=polish, **kw) if polish else best
    out["fit"].update(cost_init=best["fit"]["cost_init"], seconds=round(time.perf_counter() - t0, 1),
                      starts=[{"diff_step": r["fit"]["diff_step"], "cost": r["fit"]["cost"], "nfev": r["fit"]["nfev"]}
                              for r in runs], polish_step=polish)
    return out


# ---- report --------------------------------------------------------------------------
def approach_offsets(traces: list[Trace], preds: list[np.ndarray], still: int = 15,
                     move: float = 2.0) -> dict[str, dict[str, tuple[int, float, float]]]:
    """Hold offsets split by approach direction (friction hysteresis + gravity sag).

    Over ticks whose command moved < 0.5 units in the last `still` ticks, mean present - cmd,
    grouped by whether the command last came from above or below (by >= `move` units).
    Returns {key: {"above"|"below": (n, real_mean, sim_mean)}}.
    """
    acc: dict[str, dict[str, list]] = {k: {} for k in KEYS}
    for tr, pred in zip(traces, preds):
        c, real, ok = tr.cmd, tr.present[1:], valid_mask(tr)
        for i, k in enumerate(KEYS):
            for t in range(still, len(c)):
                if not ok[t, i] or np.ptp(c[t - still:t + 1, i]) >= 0.5:
                    continue
                prior = np.nonzero(np.abs(c[:t - still, i] - c[t, i]) >= move)[0]
                if len(prior):
                    side = "above" if c[prior[-1], i] > c[t, i] else "below"
                    acc[k].setdefault(side, []).append((real[t, i] - c[t, i], pred[t, i] - c[t, i]))
    return {k: {d: (len(v), float(np.mean([r for r, _ in v])), float(np.mean([p for _, p in v])))
                for d, v in sides.items()} for k, sides in acc.items()}


def _plots(report: Path, traces, preds_default, preds_fit, holdout):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"real": "#0b0b0b", "cmd": "#8a8984", "menagerie": "#eb6834", "fitted": "#2a78d6"}
    exc = {k: [j for j, tr in enumerate(traces) if np.ptp(tr.cmd[:, i]) >= EXCITE_MIN] for i, k in enumerate(KEYS)}
    files = []
    for i, k in enumerate(KEYS):
        rows = (exc[k] or [0])[:4]
        fig, axes = plt.subplots(len(rows), 1, figsize=(11, 3.2 * len(rows)), squeeze=False)
        for ax, j in zip(axes[:, 0], rows):
            tr = traces[j]
            t = np.arange(1, len(tr.cmd) + 1) / tr.hz
            s = _split(len(tr.cmd), holdout)
            ax.axvspan(t[s] if s < len(t) else t[-1], t[-1], color="#eeeeee", lw=0)
            ax.plot(t, tr.present[1:, i], color=colors["real"], lw=2.6, label="real")
            ax.plot(t, tr.cmd[:, i], color=colors["cmd"], lw=1, ls="--", label="command")
            ax.plot(t, preds_default[j][:, i], color=colors["menagerie"], lw=1.2, label="sim, Menagerie")
            ax.plot(t, preds_fit[j][:, i], color=colors["fitted"], lw=1.2, label="sim, fitted")
            ax.set_title(f"{k} · {tr.name}  (shaded: holdout)", fontsize=10, loc="left")
            ax.set_xlabel("s", fontsize=8)
            ax.set_ylabel("units", fontsize=8)
            ax.tick_params(labelsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0, 0].legend(fontsize=8, ncol=4, frameon=False, loc="best")
        fig.tight_layout()
        name = f"overlay_{joint_name(k)}.png"
        fig.savefig(report / name, dpi=80)
        plt.close(fig)
        files.append(name)
    return files


def _pct(a: float, b: float) -> str:
    return f"{(b - a) / a:+.0%}" if a > 0 else ""


def write_report(report: Path, traces, params, rm_default, rm_fit, preds_default, preds_fit, default_params,
                 holdout=HOLDOUT, tr_default=None, tr_fit=None):
    report.mkdir(parents=True, exist_ok=True)
    pngs = _plots(report, traces, preds_default, preds_fit, holdout)
    f = params["fit"]
    exc = f["excitation"]
    body = [k for k in KEYS if not k.endswith("gripper.pos")]
    L = ["# SO101 servo system identification", "",
         "Generated by `python -m halloween_bot.sim.sysid fit`. Traces: "
         + ", ".join(f"`{tr.name}` ({len(tr.cmd)} ticks @ {tr.hz:g} Hz)" for tr in traces) + ".", "",
         "## Method", "",
         f"- **Replay** (props-free model): start at the first measured frame; each tick clamp the command to "
         f"sim present ± {MAX_RELATIVE_TARGET:g} (lerobot `max_relative_target`), bound it to the key's range, "
         "write ctrl, step physics for one control period. Sim time is locked to the control clock (7/7/6 steps of "
         "5 ms per 30 Hz tick), like the wall-clock-paced engine. Open loop: the sim is never reset to the real "
         "state.",
         f"- **Target**: `action[t]` is answered by `observation.state[t+1]`. Gripper ticks where the real jaw "
         f"stays > {CONTACT_GAP:g} units above its command while still (an object in the jaws) are masked.",
         f"- **Fit**: log of {', '.join(f['fields'])} per motor, shared by both arms, one joint "
         "`scipy.optimize.least_squares` (2-point Jacobian, `x_scale='jac'`, bounds 1e-4..2e3). Residuals are "
         f"sim − real over the first {1 - holdout:.0%} of each trace, per key divided by that key's std "
         f"(floor {SCALE_FLOOR:g} units). Only keys whose command spans ≥ {EXCITE_MIN:g} units enter the residual; "
         "a motor with no such key keeps its Menagerie values.",
         "- **Multi-start**: the cost is rugged (friction, force saturation), so the finite-difference step decides "
         "where the optimizer stalls. One fit per step runs in parallel; the lowest *training* cost is continued "
         "with a coarser step. Holdout data never influences any choice.",
         f"- **Holdout**: the last {holdout:.0%} of each trace, scored on the same open-loop replay.", "",
         "## Holdout RMSE per key (normalized units)", "",
         "| key | commanded range | Menagerie | fitted | change |", "|---|---:|---:|---:|---:|"]
    for k in KEYS:
        a, b = rm_default[k], rm_fit[k]
        note = "" if exc[k] >= EXCITE_MIN else " (not excited)"
        L.append(f"| `{k}` | {exc[k]:.1f}{note} | {a:.2f} | {b:.2f} | {_pct(a, b)} |")
    for label, keys in (("mean, body joints", body), ("mean, all keys", KEYS)):
        a, b = np.nanmean([rm_default[k] for k in keys]), np.nanmean([rm_fit[k] for k in keys])
        L.append(f"| **{label}** | | **{a:.2f}** | **{b:.2f}** | **{_pct(a, b)}** |")
    if tr_default and tr_fit:
        L += ["", f"Training-window RMSE, mean over all keys: Menagerie {np.nanmean(list(tr_default.values())):.2f}, "
                  f"fitted {np.nanmean(list(tr_fit.values())):.2f}."]
    starts = ", ".join(f"step {r['diff_step']:g} → {r['cost']:.2f} ({r['nfev']} evals)" for r in f.get("starts", []))
    L += ["", "## Fitted parameters (shared by both arms)", "",
          f"Training cost (½Σr²): Menagerie {f['cost_init']:.1f} → fitted {f['cost']:.2f}. "
          + (f"Starts: {starts}; best continued with step {f['polish_step']:g} ({f['nfev']} evals). "
             if starts else f"{f['nfev']} evals. ")
          + f"Wall time {f['seconds']} s. Jacobian condition number at the optimum {f['jac_condition']:.2g}. "
          f"Residual keys: {', '.join(k.removesuffix('.pos') for k in f['residual_keys'])}.", "",
          "| motor | " + " | ".join(FIELDS) + " |", "|---|" + "---:|" * len(FIELDS)]
    for m in MOTORS:
        mine = params["motors"].get(m, {})
        fp = {**default_params["motors"][m], **mine}
        L.append(f"| {m} | " + " | ".join(f"**{fp[x]:.4g}**" if x in mine else f"{fp[x]:.4g}" for x in FIELDS) + " |")
    L += ["", "Bold = fitted; plain = Menagerie `sts3215` default (kp 998.22, kv 2.731, damping 0.60, "
              "frictionloss 0.052, armature 0.028, forcerange 2.94).", "",
          "## Hold offsets by approach direction (present − command, units)", "",
          "Ticks where the command has been still for 0.5 s, split by whether it last arrived from above or "
          "below. Real servos stop short of the target (friction) and sag under gravity; known real behavior: "
          "the elbow reads ~+5 above command extended horizontally, shoulder_lift sags 1–2. Menagerie's stiff "
          "servo holds ~0 either way.", "",
          "| key | from | n | real | Menagerie | fitted |", "|---|---|---:|---:|---:|---:|"]
    off_d, off_f = approach_offsets(traces, preds_default), approach_offsets(traces, preds_fit)
    for k in body:
        for side in ("above", "below"):
            if side in off_f[k]:
                n, real, sf = off_f[k][side]
                L.append(f"| `{k}` | {side} | {n} | {real:+.2f} | {off_d[k][side][2]:+.2f} | {sf:+.2f} |")
    L += ["", "## Overlays", "", "Grey band = holdout. Thick black = real, blue = fitted sim, orange = Menagerie "
              "sim, dashed grey = command.", ""] + [f"![{p}]({p})" for p in pngs]
    (report / "report.md").write_text("\n".join(L) + "\n")


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m halloween_bot.sim.sysid")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit", help="fit servo params to real traces and write params.json + report")
    p.add_argument("--dataset", action="append", default=[], help="LeRobot v3 dataset root (repeatable)")
    p.add_argument("--traces", action="append", default=[], help="sysid_collect JSON (repeatable)")
    p.add_argument("--out", default=str(PARAMS_FILE))
    p.add_argument("--report", default="docs/sysid")
    p.add_argument("--holdout", type=float, default=HOLDOUT)
    p.add_argument("--max-nfev", type=int, default=None)
    p.add_argument("--fields", default=",".join(FIELDS))
    p.add_argument("--diff-steps", default=",".join(f"{d:g}" for d in MULTISTART))
    p.add_argument("--polish", type=float, default=POLISH_STEP, help="0 disables the polish run")
    p.add_argument("-v", "--verbose", type=int, default=1)
    a = ap.parse_args(argv)

    traces = [t for d in a.dataset for t in load_dataset_traces(d)] + \
             [t for f in a.traces for t in load_trajectory_traces(f)]
    if not traces:
        ap.error("give at least one --dataset or --traces")
    calib = Calibration.load()
    m_default = build_model(params={}, calib=calib, props=False)
    default_params = model_params(m_default)
    params = fit_multistart(traces, diff_steps=[float(d) for d in a.diff_steps.split(",")], polish=a.polish or None,
                            calib=calib, holdout=a.holdout, max_nfev=a.max_nfev, fields=tuple(a.fields.split(",")),
                            verbose=a.verbose)
    m_fit = build_model(params=params, calib=calib, props=False)
    preds_d = [replay(m_default, calib, tr) for tr in traces]
    preds_f = [replay(m_fit, calib, tr) for tr in traces]
    rm_d = rmse(m_default, calib, traces, a.holdout, preds_d)
    rm_f = rmse(m_fit, calib, traces, a.holdout, preds_f)
    tr_d = _window_rmse(traces, preds_d, a.holdout, train=True)
    tr_f = _window_rmse(traces, preds_f, a.holdout, train=True)
    params["fit"]["holdout_rmse"] = {"menagerie": rm_d, "fitted": rm_f}
    Path(a.out).write_text(json.dumps(params, indent=2) + "\n")
    write_report(Path(a.report), traces, params, rm_d, rm_f, preds_d, preds_f, default_params, a.holdout, tr_d, tr_f)
    body = [k for k in KEYS if not k.endswith("gripper.pos")]
    print(f"wrote {a.out} and {a.report}/report.md")
    print(f"holdout RMSE body mean: Menagerie {np.nanmean([rm_d[k] for k in body]):.2f} -> "
          f"fitted {np.nanmean([rm_f[k] for k in body]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
