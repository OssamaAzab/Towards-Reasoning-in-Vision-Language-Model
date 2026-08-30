"""Regression tests for RNG-state restoration on --continue-from.

Job 2286251 died with `TypeError: RNG state must be a torch.ByteTensor` AFTER its optimizer
state had been restored. The checkpoint is loaded with `torch.load(..., map_location=device)`
and map_location moves every tensor in it to that device, including the saved RNG states, but
torch.set_rng_state and torch.cuda.set_rng_state_all both require CPU byte tensors.

The bug had never fired because every earlier continuation resumed from a legacy checkpoint
with no rng_state, so the branch was never reached. These tests reach it.
"""
from importlib import import_module
from pathlib import Path
import random
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

T = import_module("06c_train_bridge")


class CpuByteCoercionTests(unittest.TestCase):
    """_cpu_byte must return exactly what the RNG setters accept."""

    def test_a_cpu_byte_tensor_is_returned_unchanged_in_value(self):
        src = torch.get_rng_state()
        got = T._cpu_byte(src)
        self.assertEqual(got.device.type, "cpu")
        self.assertEqual(got.dtype, torch.uint8)
        self.assertTrue(torch.equal(got, src))

    def test_a_non_uint8_tensor_is_coerced(self):
        got = T._cpu_byte(torch.tensor([1, 2, 3], dtype=torch.int64))
        self.assertEqual(got.dtype, torch.uint8)
        self.assertEqual(got.device.type, "cpu")

    def test_result_is_contiguous(self):
        src = torch.arange(16, dtype=torch.uint8).reshape(4, 4).t()
        self.assertFalse(src.is_contiguous())
        self.assertTrue(T._cpu_byte(src).is_contiguous())

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA to reproduce the exact failure")
    def test_a_cuda_resident_state_is_brought_back_to_cpu(self):
        """The precise condition that killed 2286251."""
        src = torch.get_rng_state().cuda()
        got = T._cpu_byte(src)
        self.assertEqual(got.device.type, "cpu")
        torch.set_rng_state(got)          # would raise TypeError without the fix


class RoundTripTests(unittest.TestCase):
    """capture -> (simulated map_location move) -> restore must reproduce the stream."""

    def test_round_trip_reproduces_the_random_stream(self):
        random.seed(1234)
        torch.manual_seed(1234)
        state = T.capture_rng_state(include_cuda=False)
        expect_py = [random.random() for _ in range(3)]
        expect_pt = torch.rand(3)

        random.seed(999)
        torch.manual_seed(999)
        T.restore_rng_state(state, include_cuda=False)
        self.assertEqual([random.random() for _ in range(3)], expect_py)
        self.assertTrue(torch.equal(torch.rand(3), expect_pt))

    def test_round_trip_survives_a_dtype_shift_in_the_saved_state(self):
        """Stands in for map_location on machines without CUDA: the state arrives 'wrong'."""
        random.seed(7)
        torch.manual_seed(7)
        state = T.capture_rng_state(include_cuda=False)
        expect = torch.rand(4)
        # A non-uint8 tensor is exactly what set_rng_state rejects.
        state = dict(state, torch_cpu=state["torch_cpu"].to(torch.int64))
        torch.manual_seed(0)
        T.restore_rng_state(state, include_cuda=False)
        self.assertTrue(torch.equal(torch.rand(4), expect))

    def test_restore_without_the_fix_would_fail(self):
        """Pin the failure mode itself, so a revert cannot pass silently."""
        with self.assertRaises(TypeError):
            torch.set_rng_state(torch.get_rng_state().to(torch.int64))


class CudaAbsenceTests(unittest.TestCase):
    """A CUDA state in the checkpoint on a CPU-only box must fail loudly, not silently."""

    @unittest.skipIf(torch.cuda.is_available(), "only meaningful without CUDA")
    def test_cuda_state_without_cuda_aborts(self):
        state = T.capture_rng_state(include_cuda=False)
        state = dict(state, torch_cuda=[torch.get_rng_state()])
        with self.assertRaises(SystemExit) as cm:
            T.restore_rng_state(state, include_cuda=True)
        self.assertIn("CUDA is unavailable", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
