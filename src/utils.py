"""Shared helpers: config loading, RNG seeding, device + directory utilities."""
from __future__ import annotations

import json
import os
import random
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | None = None) -> dict:
    """Load the YAML config, resolving ${PROJECT_ROOT} to the repo root (portable)."""
    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(path) if path is not None else root / "config" / "default.yaml"
    text = cfg_path.read_text().replace("${PROJECT_ROOT}", str(root))
    return yaml.safe_load(text)


def set_seed(seed: int = 42) -> None:
    """Fix Python / NumPy / PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    """Return 'cuda' if a GPU is visible, else 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def vram_report() -> str:
    """Peak VRAM against the ACTUAL device capacity, not a hardcoded budget.

    Five scripts printed "... GB / 20 GB", the local RTX 4000 Ada figure. On the
    cluster's 97 GB card that made a healthy 26.85 GB peak read as 34% over budget.
    A number is only interpretable against the device it was measured on.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "n/a (no CUDA device)"
        peak = torch.cuda.max_memory_allocated() / 1024**3
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        return f"{peak:.2f} GB / {total:.0f} GB ({props.name})"
    except ImportError:
        return "n/a (torch unavailable)"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_dir(spec: str | list[str]) -> Path:
    """First existing directory from `spec` (a path, or a list tried in order).

    Lets one config serve machines that mount the same dataset at different paths.
    A path that works on a login host may not exist on a compute node, so callers
    provide an ordered list and the first existing directory is selected.

    Mirrors GQADataset.image_dirs, which already took a list for the same reason.
    A plain string still works unchanged, so existing callers/configs are unaffected.
    Falls back to the first entry when none exist, so the caller raises a normal
    "no such file" naming a real path rather than something confusing.
    """
    dirs = [spec] if isinstance(spec, (str, Path)) else list(spec)
    for d in dirs:
        if Path(d).is_dir():
            return Path(d)
    return Path(dirs[0])


def canonical_dir_map(cfg: dict) -> dict[str, str]:
    """Map each RESOLVED image dir back to its canonical (first-listed) form.

    The train/val split is a CRC32 of the image path (06c_train_bridge.py), so mounting the
    same dataset at a different path silently produces a different split. The canonical path
    prevents host-specific mounts from changing validation membership.

    Hashing the canonical form fixes both halves of that: it is host-independent, AND it
    reproduces the locked runs' split bit-for-bit, because the canonical form is the
    first-listed candidate — the path every locked run actually used.

    Returns {resolved_dir: canonical_dir} for the training image sources.
    """
    out: dict[str, str] = {}
    for section in ("llava", "vqa"):
        spec = cfg.get(section, {}).get("image_dir")
        if spec is None:
            continue
        dirs = [spec] if isinstance(spec, (str, Path)) else list(spec)
        out[str(resolve_dir(dirs))] = str(dirs[0])
    return out


def split_key(image_path, dir_map: dict[str, str]) -> str:
    """Host-independent key for the deterministic train/val split (see canonical_dir_map)."""
    p = Path(image_path)
    return f"{dir_map.get(str(p.parent), str(p.parent))}/{p.name}"


class StepProfiler:
    """Phase timings for one training step, reported as TWO decompositions that must never be added.

    Measurement only — it never touches weights, data, or RNG. The trainer constructs one
    ONLY when --profile-steps > 0, so the default training path is unchanged.

    THE OVERLAP PROBLEM, and why there are two decompositions.
    CUDA is asynchronous: the CPU queues kernels and runs on. So a CPU span around a GPU
    call measures LAUNCH time, and the GPU keeps working after the span closes. When the
    CPU later blocks on loss.item(), it waits for that same GPU work to drain. Those
    milliseconds therefore appear TWICE — once inside the CUDA-event time for the forward
    and backward, and again inside the CPU wait. Summing them double counts and can drive
    a residual negative. Two separate views are kept instead:

      * WALL TIMELINE (category "cpu" + "wait") — serial, blocking segments of the step.
        These are additive and their sum is compared against step_wall_ms; whatever is
        left over is reported as an honest residual.
      * CUDA STREAM ELAPSED (category "cuda_elapsed") — time between two CUDA events on
        the stream. This is NOT GPU-busy time and NOT utilisation: the stream can sit
        idle inside the span while the CPU tokenises, builds sequences in Python, and
        submits kernels. It is an UPPER BOUND on device activity for that region.
        True kernel-active time and bitsandbytes attribution come from the Chrome trace
        (--profile-trace), not from here. Never summed with the wall timeline.

    step_wall_ms is the authoritative per-step number. Even so, a profiled step is not a
    throughput measurement: end_step() must synchronize to read the CUDA events, which
    serializes optimizer work that would otherwise overlap the next step's feature I/O.
    Take examples/second from an UNPROFILED run.

    Two further choices, each because the naive version gives a wrong answer here:
      * The first `warmup` steps are timed and DISCARDED — allocator growth, kernel
        autotuning and first-touch page faults make them unrepresentative.
      * Results are median/p90, not mean: the trainer writes a 723 MB checkpoint every
        1000 batches, and one such step poisons a mean.

    Off CUDA every span falls back to wall-clock, so the class is testable on CPU.
    """

    def __init__(self, steps: int, warmup: int = 20, device: str = "cuda",
                 out_path: str | Path | None = None, meta: dict | None = None):
        """Record `steps` measured steps after discarding `warmup` warm-up steps."""
        import torch

        self._torch = torch
        self.steps = int(steps)
        self.warmup = int(warmup)
        self.out_path = Path(out_path) if out_path else None
        self.meta = dict(meta or {})
        self._cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        self._records: list[dict] = []
        self._pending: list[tuple] = []
        self._step: dict = {}
        self._category: dict[str, str] = {}   # key -> cpu | wait | gpu | scalar
        self._n_seen = 0
        self._t_step = 0.0

    @property
    def active(self) -> bool:
        """True while more steps still need to be recorded."""
        return self._n_seen < self.warmup + self.steps

    def start_step(self) -> None:
        """Open a new step record."""
        self._step = {}
        self._pending = []
        self._t_step = time.perf_counter()

    def _record(self, name: str, value: float, category: str) -> None:
        """Store one span value for the current step and remember its category."""
        key = f"{name}_ms" if category != "scalar" else name
        self._step[key] = value
        self._category[key] = category

    @contextmanager
    def _wall_span(self, name: str, category: str):
        """Time a blocking span with the wall clock (ms) under the given category."""
        t = time.perf_counter()
        try:
            yield
        finally:
            self._record(name, (time.perf_counter() - t) * 1000.0, category)

    def cpu(self, name: str):
        """Time a serial CPU segment — additive, counts toward the wall timeline."""
        return self._wall_span(name, "cpu")

    def gpu_wait(self, name: str):
        """Time a CPU block waiting for the GPU to drain.

        Additive in the wall timeline (it really is serial wall time), but the SAME
        milliseconds also sit inside the CUDA-event spans for the work being waited on.
        Never add this to a gpu-category span.
        """
        return self._wall_span(name, "wait")

    @contextmanager
    def cuda_elapsed(self, name: str):
        """Elapsed time between two CUDA events on the stream (ms) — an UPPER BOUND.

        Deliberately not called "gpu time": the stream can idle inside the span while the
        CPU tokenises and submits kernels, and that idle time is included. Not additive
        with cpu/wait spans. For kernel-active time use the Chrome trace.
        """
        if not self._cuda:
            # Off CUDA there is no asynchrony, so the fallback is honest wall-clock, but it
            # keeps the category so the two decompositions stay separate in tests.
            with self._wall_span(name, "cuda_elapsed"):
                yield
            return
        start = self._torch.cuda.Event(enable_timing=True)
        end = self._torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append((name, start, end))

    def note(self, name: str, value) -> None:
        """Attach a scalar (e.g. token count) to the step; never summed as a duration."""
        self._record(name, value, "scalar")

    def end_step(self) -> None:
        """Close the step: sync once, drain the CUDA events, keep the record if past warm-up.

        The synchronize() costs effectively nothing on top of the trainer's existing
        per-step loss.item(), which already forces a device sync at the same point.
        """
        if self._pending:
            self._torch.cuda.synchronize()
            for name, start, end in self._pending:
                self._record(name, start.elapsed_time(end), "cuda_elapsed")
            self._pending = []
        self._step["step_wall_ms"] = (time.perf_counter() - self._t_step) * 1000.0
        self._category["step_wall_ms"] = "total"
        self._n_seen += 1
        if self._n_seen > self.warmup:
            self._records.append(self._step)
        elif self._n_seen == self.warmup and self._cuda:
            self._torch.cuda.reset_peak_memory_stats()  # peak reflects steady state only
        self._step = {}

    def summary(self) -> dict:
        """Two decompositions of the step: the additive wall timeline, and GPU busy time.

        The two are reported separately and must never be added — see the class docstring.
        """
        phases = {}
        for k in sorted({k for r in self._records for k in r}):
            vals = [r[k] for r in self._records if k in r]
            if not vals or not isinstance(vals[0], (int, float)) or isinstance(vals[0], bool):
                continue
            ordered = sorted(vals)
            phases[k] = {
                "median": statistics.median(ordered),
                "p90": ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))],
                "n": len(ordered),
                "category": self._category.get(k, "unknown"),
            }

        wall = phases.get("step_wall_ms", {}).get("median", 0.0)
        # Only "cpu" and "wait" spans are serial segments of the step, so only they may be
        # summed. "cuda_elapsed" spans overlap them; "scalar" notes are not durations.
        timeline = {k: v for k, v in phases.items() if v["category"] in ("cpu", "wait")}
        cuda = {k: v for k, v in phases.items() if v["category"] == "cuda_elapsed"}
        timeline_sum = sum(v["median"] for v in timeline.values())
        cuda_sum = sum(v["median"] for v in cuda.values())

        out = {
            "meta": self.meta,
            "n_steps": len(self._records),
            "authoritative": "step_wall_ms (and even that is perturbed by profiling; "
                             "take examples/second from an unprofiled run)",
            "step_wall_ms": phases.get("step_wall_ms"),
            "wall_timeline": {
                "note": "serial cpu + gpu-wait segments; additive; compared to step_wall_ms",
                "phases": timeline,
                "sum_median_ms": timeline_sum,
                "residual_ms": wall - timeline_sum,
            },
            "cuda_stream_elapsed": {
                "note": "Time between CUDA events on the stream. NOT GPU-busy time and "
                        "NOT utilisation — the stream can idle inside a span while the CPU "
                        "tokenises and submits kernels. An UPPER BOUND on device activity. "
                        "OVERLAPS the wall timeline (the gpu-wait span covers the same "
                        "milliseconds); NEVER add the two. For kernel-active time and "
                        "bitsandbytes attribution use the Chrome trace from --profile-trace.",
                "phases": cuda,
                "sum_median_ms": cuda_sum,
                "sum_over_step_wall": (cuda_sum / wall) if wall else None,
                "sum_over_step_wall_is_not_utilisation": True,
            },
            "scalars": {k: v for k, v in phases.items() if v["category"] == "scalar"},
        }
        if self._cuda:
            out["peak_alloc_gb"] = self._torch.cuda.max_memory_allocated() / 1024 ** 3
            out["peak_reserved_gb"] = self._torch.cuda.max_memory_reserved() / 1024 ** 3
        return out

    def save(self) -> Path | None:
        """Write the summary and the raw per-step records to out_path as JSON."""
        if self.out_path is None:
            return None
        ensure_dir(self.out_path.parent)
        payload = self.summary()
        payload["records"] = self._records
        self.out_path.write_text(json.dumps(payload, indent=2, default=str))
        return self.out_path

    def format_table(self) -> str:
        """Text report: the additive wall timeline, then GPU busy time, kept clearly apart."""
        s = self.summary()
        cuda = s["cuda_stream_elapsed"]
        every = list(s["wall_timeline"]["phases"]) + list(cuda["phases"])
        width = max([len(k) for k in every] + [len("UNATTRIBUTED residual")])
        wall = (s["step_wall_ms"] or {}).get("median", 0.0)
        lines = [f"profile over {s['n_steps']} steps (median / p90, ms)",
                 f"  {'step_wall_ms (authoritative)':<{width}}  {wall:9.2f}",
                 "  -- WALL TIMELINE (additive; sums to step_wall_ms) --"]
        for k, v in sorted(s["wall_timeline"]["phases"].items(), key=lambda kv: -kv[1]["median"]):
            lines.append(f"  {k:<{width}}  {v['median']:9.2f}  {v['p90']:9.2f}  [{v['category']}]")
        lines.append(f"  {'UNATTRIBUTED residual':<{width}}  "
                     f"{s['wall_timeline']['residual_ms']:9.2f}")
        lines.append("  -- CUDA STREAM ELAPSED — overlaps the wall timeline; never add --")
        for k, v in sorted(cuda["phases"].items(), key=lambda kv: -kv[1]["median"]):
            lines.append(f"  {k:<{width}}  {v['median']:9.2f}  {v['p90']:9.2f}  [cuda_elapsed]")
        ratio = cuda["sum_over_step_wall"]
        lines.append(f"  (sum/step_wall = {'n/a' if ratio is None else f'{ratio:.0%}'} — an "
                     "UPPER BOUND on device activity, NOT utilisation; the stream can idle "
                     "inside a span. Kernel-active time comes from the Chrome trace.)")
        return "\n".join(lines)
