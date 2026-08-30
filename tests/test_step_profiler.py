"""Unit tests for the throughput profiler and its measurement-only guarantee."""
import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path

import torch

from src.utils import StepProfiler


def load_trainer_module():
    """Load the numerically prefixed trainer as a module."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "06c_train_bridge.py"
    spec = importlib.util.spec_from_file_location("train_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_instrumented_loop(trainer, prof, steps=12, seed=42):
    """Mirror the trainer's instrumented step block on CPU; return losses and final weights.

    Deliberately uses the trainer's real _span helper and the same span names and nesting
    as the training loop, so the test exercises the shipped code path rather than a copy.
    """
    torch.manual_seed(seed)
    bridge = torch.nn.Linear(8, 4)
    opt = torch.optim.AdamW(bridge.parameters(), lr=1e-3)
    batches = [torch.randn(3, 8, generator=torch.Generator().manual_seed(1000 + i))
               for i in range(steps)]

    losses = []
    for x in batches:
        if prof is not None and prof.active:
            prof.start_step()
        with trainer._span(prof, "cpu", "feature_io_h2d"):
            feats = x.clone()
        with trainer._span(prof, "cpu", "forward_wall"), trainer._span(prof, "cuda", "forward"):
            loss = bridge(feats).pow(2).mean()
        with trainer._span(prof, "cpu", "backward_wall"), trainer._span(prof, "cuda", "backward"):
            loss.backward()
        with trainer._span(prof, "wait", "gpu_drain_at_loss_item"):
            losses.append(loss.item())
        with trainer._span(prof, "cpu", "optimizer_wall"), trainer._span(prof, "cuda", "optimizer"):
            opt.step()
            opt.zero_grad()
        if prof is not None and prof.active:
            prof.end_step()
    return losses, [p.detach().clone() for p in bridge.parameters()]


class StepProfilerTests(unittest.TestCase):
    """Warm-up handling, span recording, summary shape and persistence (CPU only)."""

    def test_warmup_steps_are_timed_then_discarded(self):
        prof = StepProfiler(steps=3, warmup=2, device="cpu")
        for _ in range(5):
            prof.start_step()
            with prof.cpu("work"):
                pass
            prof.end_step()
        self.assertEqual(prof.summary()["n_steps"], 3)
        self.assertFalse(prof.active)

    def test_active_stays_true_for_warmup_plus_steps_then_goes_false(self):
        # warmup=1 + steps=2 => active for exactly 3 steps. The loop mirrors the trainer,
        # which only calls start/end_step while active, so recording stops cleanly.
        prof = StepProfiler(steps=2, warmup=1, device="cpu")
        observed = []
        for _ in range(5):
            observed.append(prof.active)
            if prof.active:
                prof.start_step()
                prof.end_step()
        self.assertEqual(observed, [True, True, True, False, False])
        self.assertEqual(prof.summary()["n_steps"], 2)

    def test_cuda_span_falls_back_to_wall_clock_off_cuda_keeping_its_category(self):
        prof = StepProfiler(steps=1, warmup=0, device="cpu")
        prof.start_step()
        with prof.cuda_elapsed("forward"):
            pass
        prof.end_step()
        # Still reported under cuda_stream_elapsed, not the wall timeline, so the two
        # decompositions stay separate on CPU exactly as they do on a GPU.
        self.assertIn("forward_ms", prof.summary()["cuda_stream_elapsed"]["phases"])

    def test_cuda_spans_are_excluded_from_the_additive_wall_timeline(self):
        # The core overlap correction: a gpu span's milliseconds also sit inside the
        # gpu-wait span, so summing both double counts and drives the residual negative.
        prof = StepProfiler(steps=2, warmup=0, device="cpu")
        for _ in range(2):
            prof.start_step()
            with prof.cpu("forward_wall"), prof.cuda_elapsed("forward"):
                time.sleep(0.002)
            with prof.gpu_wait("gpu_drain_at_loss_item"):
                time.sleep(0.002)
            prof.end_step()
        s = prof.summary()
        self.assertEqual(set(s["wall_timeline"]["phases"]),
                         {"forward_wall_ms", "gpu_drain_at_loss_item_ms"})
        self.assertEqual(set(s["cuda_stream_elapsed"]["phases"]), {"forward_ms"})
        self.assertGreaterEqual(s["wall_timeline"]["residual_ms"], 0.0)

    def test_wait_spans_are_additive_but_cuda_spans_are_not(self):
        prof = StepProfiler(steps=1, warmup=0, device="cpu")
        prof.start_step()
        with prof.cpu("a"):
            pass
        with prof.gpu_wait("b"):
            pass
        with prof.cuda_elapsed("c"):
            pass
        prof.end_step()
        s = prof.summary()
        self.assertEqual(s["wall_timeline"]["phases"]["a_ms"]["category"], "cpu")
        self.assertEqual(s["wall_timeline"]["phases"]["b_ms"]["category"], "wait")
        self.assertEqual(s["cuda_stream_elapsed"]["phases"]["c_ms"]["category"], "cuda_elapsed")
        self.assertNotIn("c_ms", s["wall_timeline"]["phases"])

    def test_scalar_notes_are_never_summed_as_durations(self):
        # Regression: a note() value used to land in the residual sum, so a token count of
        # 2321 was subtracted from the step wall time as if it were 2321 ms.
        prof = StepProfiler(steps=1, warmup=0, device="cpu")
        prof.start_step()
        prof.note("tokens", 2321)
        with prof.cpu("work"):
            pass
        prof.end_step()
        s = prof.summary()
        self.assertEqual(s["scalars"]["tokens"]["median"], 2321)
        self.assertNotIn("tokens", s["wall_timeline"]["phases"])
        self.assertLess(s["wall_timeline"]["sum_median_ms"], 100.0)
        self.assertGreaterEqual(s["wall_timeline"]["residual_ms"], 0.0)

    def test_step_wall_is_reported_as_the_authoritative_total(self):
        prof = StepProfiler(steps=1, warmup=0, device="cpu")
        prof.start_step()
        with prof.cpu("work"):
            time.sleep(0.003)
        prof.end_step()
        s = prof.summary()
        self.assertIn("authoritative", s)
        self.assertGreater(s["step_wall_ms"]["median"], 0.0)
        self.assertNotIn("step_wall_ms", s["wall_timeline"]["phases"])

    def test_save_writes_meta_phases_and_raw_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "profile.json"
            prof = StepProfiler(steps=2, warmup=0, device="cpu",
                                out_path=out, meta={"attn_implementation": "sdpa"})
            for _ in range(2):
                prof.start_step()
                with prof.cpu("work"):
                    pass
                prof.end_step()
            self.assertEqual(prof.save(), out)
            payload = json.loads(out.read_text())
            self.assertEqual(payload["meta"]["attn_implementation"], "sdpa")
            self.assertEqual(payload["n_steps"], 2)
            self.assertEqual(len(payload["records"]), 2)
            self.assertIn("work_ms", payload["wall_timeline"]["phases"])

    def test_json_save_cannot_inflate_the_final_measured_step(self):
        """end_step() must finalise the record BEFORE the trainer serialises the JSON.

        The trainer saves the profile the moment recording completes, so that a later
        failure cannot lose it. If that save ran before the step record was closed, its
        serialisation cost would land inside the last step_wall_ms.
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "profile.json"
            prof = StepProfiler(steps=2, warmup=0, device="cpu", out_path=out)
            real_save = prof.save

            def slow_save():
                time.sleep(0.05)          # 50 ms — far above any real step in this test
                return real_save()

            prof.save = slow_save
            for _ in range(2):            # mirror the trainer's save-on-transition exactly
                prof.start_step()
                with prof.cpu("work"):
                    pass
                prof.end_step()
                if not prof.active:
                    prof.save()
            # p90 over the recorded steps is the worst case; the 50 ms save must be absent.
            self.assertLess(prof.summary()["step_wall_ms"]["p90"], 50.0)
            self.assertTrue(out.exists())

    def test_format_table_keeps_the_two_decompositions_visibly_apart(self):
        prof = StepProfiler(steps=1, warmup=0, device="cpu")
        prof.start_step()
        with prof.cpu("work"):
            pass
        with prof.cuda_elapsed("forward"):
            pass
        prof.end_step()
        table = prof.format_table()
        self.assertIn("WALL TIMELINE", table)
        self.assertIn("CUDA STREAM ELAPSED", table)
        self.assertIn("never add", table)
        self.assertIn("UNATTRIBUTED residual", table)
        self.assertIn("authoritative", table)


class ProfilerIsMeasurementOnlyTests(unittest.TestCase):
    """The profiler must not perturb training: same losses, same weights, bit for bit."""

    @classmethod
    def setUpClass(cls):
        cls.trainer = load_trainer_module()

    def test_span_is_a_shared_noop_when_profiling_is_off(self):
        # Identity, not just equality: proves the disabled path allocates nothing per step.
        self.assertIs(self.trainer._span(None, "cpu", "anything"), self.trainer._NO_SPAN)
        self.assertIs(self.trainer._span(None, "gpu", "anything"), self.trainer._NO_SPAN)

    def test_exhausted_profiler_also_returns_the_shared_noop(self):
        prof = StepProfiler(steps=1, warmup=0, device="cpu")
        prof.start_step()
        prof.end_step()
        self.assertFalse(prof.active)
        self.assertIs(self.trainer._span(prof, "cpu", "work"), self.trainer._NO_SPAN)

    def test_losses_and_weights_are_bitwise_identical_with_and_without_profiling(self):
        off_losses, off_weights = run_instrumented_loop(self.trainer, None)
        prof = StepProfiler(steps=8, warmup=2, device="cpu")
        on_losses, on_weights = run_instrumented_loop(self.trainer, prof)

        self.assertEqual(off_losses, on_losses)  # float equality: must be bit for bit
        for a, b in zip(off_weights, on_weights):
            self.assertTrue(torch.equal(a, b))
        # And the profiler really did observe the run it was supposed to observe.
        self.assertEqual(prof.summary()["n_steps"], 8)

    def test_profiling_records_every_named_phase_in_the_right_decomposition(self):
        prof = StepProfiler(steps=4, warmup=0, device="cpu")
        run_instrumented_loop(self.trainer, prof, steps=4)
        s = prof.summary()
        # Every serial segment of the step is covered, so the residual stays meaningful.
        for name in ("feature_io_h2d_ms", "forward_wall_ms", "backward_wall_ms",
                     "gpu_drain_at_loss_item_ms", "optimizer_wall_ms"):
            self.assertIn(name, s["wall_timeline"]["phases"])
        for name in ("forward_ms", "backward_ms", "optimizer_ms"):
            self.assertIn(name, s["cuda_stream_elapsed"]["phases"])
            self.assertNotIn(name, s["wall_timeline"]["phases"])
        self.assertGreaterEqual(s["wall_timeline"]["residual_ms"], 0.0)

    def test_span_kinds_map_to_the_documented_categories(self):
        prof = StepProfiler(steps=1, warmup=0, device="cpu")
        prof.start_step()
        with self.trainer._span(prof, "cpu", "c"):
            pass
        with self.trainer._span(prof, "wait", "w"):
            pass
        with self.trainer._span(prof, "cuda", "g"):
            pass
        prof.end_step()
        s = prof.summary()
        self.assertEqual(s["wall_timeline"]["phases"]["c_ms"]["category"], "cpu")
        self.assertEqual(s["wall_timeline"]["phases"]["w_ms"]["category"], "wait")
        self.assertEqual(s["cuda_stream_elapsed"]["phases"]["g_ms"]["category"], "cuda_elapsed")


if __name__ == "__main__":
    unittest.main()
