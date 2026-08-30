"""Tests for the sequence-normalised training objective (experiment #1, arm B).

Scope note, because it decides what these tests are allowed to claim:

The causal-shift and -100 masking convention used by `sequence_normalised_loss` is NOT
established here. It was established against HuggingFace's own `.loss` on real GPU data by
`scripts/34_nll_decomposition.py` in job 2288749, which reproduced `.loss` to a worst-case
1.3e-08 across 16 checkpoints (see the `control` block of each artifact in
outputs/analysis/nll_decomp_2288749/). The fake model below deliberately computes its
`.loss` the same way, so comparing the two here would be circular and proves nothing.

What these tests DO establish, none of it circular:
  * the two reductions coincide exactly when every sequence is the same length, and
    diverge when they are not (a mathematical identity, independent of any implementation);
  * long answers lose their extra gradient weight, which is the entire point of the arm;
  * `vlm_loss_batch` actually routes to the new reduction rather than ignoring the flag;
  * an unrecognised loss_norm is refused before any forward pass runs;
  * a legacy checkpoint cannot be resumed under a different objective.
"""
from __future__ import annotations

import importlib.util
import math
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.models.vlm import LOSS_NORMS, sequence_normalised_loss, vlm_loss_batch

ROOT = Path(__file__).resolve().parents[1]
VOCAB, DIM, N_ENC, ENC_DIM = 64, 16, 4, 8


def load_trainer_module():
    """Load the numerically prefixed trainer as a module."""
    path = ROOT / "scripts" / "06c_train_bridge.py"
    spec = importlib.util.spec_from_file_location("train_bridge_lossnorm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def token_mean(logits, labels):
    """HuggingFace's reduction: one mean over every supervised token in the batch."""
    shift_logits, shift_labels = logits[:, :-1, :], labels[:, 1:]
    keep = shift_labels != -100
    return F.cross_entropy(shift_logits[keep].float(), shift_labels[keep], reduction="mean")


def uniform_logits(rows):
    """Logits whose CE against any target is exactly ln(VOCAB), for hand-computable cases."""
    return torch.zeros(rows, VOCAB)


class FakeTokenizer:
    """Whitespace tokenizer, one id per word, deterministic across processes.

    crc32 rather than hash(): str hashing is salted per interpreter (PYTHONHASHSEED), which
    would make every length and id in these tests vary run to run.
    """

    def __init__(self):
        self.eos_token_id = VOCAB - 1

    def __call__(self, text, add_special_tokens=True):
        ids = [(zlib.crc32(w.encode()) % (VOCAB - 2)) + 1 for w in text.split()]
        return SimpleNamespace(input_ids=ids)


class FakeCausalLM(torch.nn.Module):
    """Minimal stand-in for the frozen LLM: real embeddings, real head, HF-shaped return."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(VOCAB, DIM)
        self.head = torch.nn.Linear(DIM, VOCAB, bias=False)
        self.calls = []

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, use_cache=None):
        self.calls.append({"labels_was_none": labels is None})
        logits = self.head(inputs_embeds)
        loss = token_mean(logits, labels) if labels is not None else None
        return SimpleNamespace(logits=logits, loss=loss)


class FakeBridge(torch.nn.Module):
    """Encoder features -> visual tokens, at the embedding width."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.proj = torch.nn.Linear(ENC_DIM, DIM)

    def forward(self, feats):
        return self.proj(feats)


def make_stack():
    """A bridge + fake LLM + feature batch factory."""
    llm = SimpleNamespace(model=FakeCausalLM(), tokenizer=FakeTokenizer())
    return FakeBridge(), llm


def feats_for(n):
    torch.manual_seed(2)
    return torch.randn(n, N_ENC, ENC_DIM)


class ReductionIdentityTests(unittest.TestCase):
    """The two reductions are the same number exactly when the lengths are equal."""

    def _batch(self, supervised_per_seq, seq_len=8):
        """Logits and labels with a chosen count of supervised tokens per sequence."""
        torch.manual_seed(3)
        b = len(supervised_per_seq)
        logits = torch.randn(b, seq_len, VOCAB)
        labels = torch.full((b, seq_len), -100)
        for i, n in enumerate(supervised_per_seq):
            # Supervised span sits at the tail, as it does in the real layout. The shift
            # drops position 0, so ask for n supervised *post-shift* positions.
            if n:
                labels[i, seq_len - n:] = torch.randint(0, VOCAB, (n,))
        return logits, labels

    def test_equal_lengths_make_the_two_reductions_identical(self):
        """A mathematical identity: mean of equal-size means == mean of the pool."""
        logits, labels = self._batch([3, 3, 3])
        self.assertAlmostEqual(float(sequence_normalised_loss(logits, labels)),
                               float(token_mean(logits, labels)), places=5)

    def test_unequal_lengths_make_the_two_reductions_differ(self):
        """The flag must change the number. If this passes trivially it is not measuring.

        This is the test that fails if `loss_norm="sequence"` silently falls through to
        HuggingFace's reduction, so it is the load-bearing one for the whole arm.
        """
        logits, labels = self._batch([1, 6])
        seq = float(sequence_normalised_loss(logits, labels))
        tok = float(token_mean(logits, labels))
        self.assertNotAlmostEqual(seq, tok, places=4,
                                  msg="unequal-length sequences must reduce differently")

    def test_long_answers_lose_their_extra_weight(self):
        """The motivating property, stated as a number rather than as an intention.

        One cheap 1-token sequence and one expensive 6-token sequence. Under the token
        mean the expensive sequence supplies 6 of 7 tokens and drags the scalar most of
        the way to its own value; under the sequence mean the two count equally, so the
        scalar sits at the midpoint.
        """
        cheap, dear = 0.0, 10.0
        seq_len = 8
        logits = torch.full((2, seq_len, VOCAB), 0.0)
        labels = torch.full((2, seq_len), -100)
        # Sequence 0: one supervised token, driven to near-zero CE by a dominant logit.
        labels[0, seq_len - 1] = 5
        logits[0, seq_len - 2, 5] = 30.0
        # Sequence 1: six supervised tokens, all left at uniform logits -> CE = ln(VOCAB).
        labels[1, seq_len - 6:] = 7

        seq = float(sequence_normalised_loss(logits, labels))
        tok = float(token_mean(logits, labels))
        expensive = math.log(VOCAB)
        self.assertAlmostEqual(seq, (cheap + expensive) / 2, places=3)
        self.assertAlmostEqual(tok, (cheap + 6 * expensive) / 7, places=3)
        self.assertLess(seq, tok, "equal weighting must pull the scalar off the long answer")
        self.assertGreater(dear, 0)  # guards the fixture's own premise

    def test_hand_computed_value(self):
        """Two sequences at known CE: one at ln(VOCAB), one at ~0. Mean must be ln(V)/2."""
        seq_len = 6
        logits = torch.zeros(2, seq_len, VOCAB)
        labels = torch.full((2, seq_len), -100)
        labels[0, seq_len - 2:] = 3               # 2 tokens at uniform logits -> ln(VOCAB)
        labels[1, seq_len - 1] = 9                # 1 token driven to ~0 CE
        logits[1, seq_len - 2, 9] = 40.0
        self.assertAlmostEqual(float(sequence_normalised_loss(logits, labels)),
                               math.log(VOCAB) / 2, places=4)


class MaskingEdgeCaseTests(unittest.TestCase):
    """Fully-masked sequences must be skipped, never counted as a free zero."""

    def test_unsupervised_sequence_is_skipped_not_zeroed(self):
        seq_len = 6
        logits = torch.zeros(3, seq_len, VOCAB)
        labels = torch.full((3, seq_len), -100)
        labels[0, seq_len - 1] = 4
        labels[2, seq_len - 1] = 4                # sequence 1 stays entirely unsupervised
        got = float(sequence_normalised_loss(logits, labels))
        self.assertAlmostEqual(got, math.log(VOCAB), places=4)
        # If the empty sequence were averaged in as 0.0 the mean would be 2/3 of that.
        self.assertNotAlmostEqual(got, 2 * math.log(VOCAB) / 3, places=3)

    def test_batch_with_no_supervision_raises(self):
        logits = torch.zeros(2, 5, VOCAB)
        labels = torch.full((2, 5), -100)
        with self.assertRaisesRegex(RuntimeError, "no supervised tokens"):
            sequence_normalised_loss(logits, labels)

    def test_gradient_reaches_the_logits(self):
        """A loss that does not backpropagate would train nothing and report fine."""
        logits = torch.zeros(2, 6, VOCAB, requires_grad=True)
        labels = torch.full((2, 6), -100)
        labels[0, 4:] = 3
        labels[1, 5] = 8
        sequence_normalised_loss(logits, labels).backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


class BatchRoutingTests(unittest.TestCase):
    """`vlm_loss_batch` must honour the flag, not merely accept it."""

    def test_token_norm_returns_the_models_own_loss(self):
        bridge, llm = make_stack()
        got = vlm_loss_batch(bridge, llm, feats_for(2), ["what colour", "how many"],
                             ["red", "two"], "cpu", loss_norm="token")
        self.assertFalse(llm.model.calls[-1]["labels_was_none"],
                         "the token path must hand labels to the model")
        self.assertTrue(torch.isfinite(got))

    def test_equal_length_answers_agree_across_norms(self):
        """Routing check with the identity as the oracle, so nothing is self-referential."""
        answers = ["red", "two"]                  # one token each under FakeTokenizer
        bridge, llm = make_stack()
        tok = vlm_loss_batch(bridge, llm, feats_for(2), ["q one", "q two"], answers, "cpu",
                             loss_norm="token")
        bridge2, llm2 = make_stack()
        seq = vlm_loss_batch(bridge2, llm2, feats_for(2), ["q one", "q two"], answers, "cpu",
                             loss_norm="sequence")
        self.assertAlmostEqual(float(tok), float(seq), places=5)

    def test_unequal_length_answers_diverge_across_norms(self):
        """The end-to-end version of the load-bearing check."""
        answers = ["red", "a long descriptive sentence about the scene"]
        bridge, llm = make_stack()
        tok = vlm_loss_batch(bridge, llm, feats_for(2), ["q one", "q two"], answers, "cpu",
                             loss_norm="token")
        bridge2, llm2 = make_stack()
        seq = vlm_loss_batch(bridge2, llm2, feats_for(2), ["q one", "q two"], answers, "cpu",
                             loss_norm="sequence")
        self.assertNotAlmostEqual(float(tok), float(seq), places=4)

    def test_sequence_norm_withholds_labels_from_the_model(self):
        bridge, llm = make_stack()
        vlm_loss_batch(bridge, llm, feats_for(2), ["q one", "q two"], ["red", "blue sky"],
                       "cpu", loss_norm="sequence")
        self.assertTrue(llm.model.calls[-1]["labels_was_none"],
                        "the sequence path must not ask the model for a reduction it discards")

    def test_default_is_the_legacy_reduction(self):
        """Omitting the argument must reproduce pre-2026-08-03 behaviour exactly."""
        bridge, llm = make_stack()
        default = vlm_loss_batch(bridge, llm, feats_for(2), ["q one", "q two"],
                                 ["red", "a much longer answer"], "cpu")
        bridge2, llm2 = make_stack()
        explicit = vlm_loss_batch(bridge2, llm2, feats_for(2), ["q one", "q two"],
                                  ["red", "a much longer answer"], "cpu", loss_norm="token")
        self.assertEqual(float(default), float(explicit))

    def test_unknown_norm_is_refused_before_any_forward(self):
        bridge, llm = make_stack()
        with self.assertRaisesRegex(ValueError, "loss_norm must be one of"):
            vlm_loss_batch(bridge, llm, feats_for(2), ["q one", "q two"], ["red", "blue"],
                           "cpu", loss_norm="seqence")     # deliberate typo
        self.assertEqual(llm.model.calls, [],
                         "a bad flag must cost nothing, not be caught after a forward pass")

    def test_norm_names_are_the_two_documented_ones(self):
        self.assertEqual(LOSS_NORMS, ("token", "sequence"))


class RealQwen2ConventionTests(unittest.TestCase):
    """Validate the shift/mask convention against the REAL model class, not the fake.

    A randomly-initialised 2-layer Qwen2 is a few MB on CPU and takes about a second, so
    the actual `Qwen2ForCausalLM.forward` — the one the 7B checkpoint runs — computes the
    reference `.loss` here. Nothing in this class is self-referential: HuggingFace produces
    the number being matched.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError as exc:                       # pragma: no cover
            raise unittest.SkipTest(f"transformers unavailable: {exc}")
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
        cls.model = Qwen2ForCausalLM(cfg).eval()
        cls.embeds = torch.randn(3, 12, 32)
        labels = torch.full((3, 12), -100)
        torch.manual_seed(4)
        labels[0, 8:] = torch.randint(0, 128, (4,))      # deliberately unequal spans
        labels[1, 11:] = torch.randint(0, 128, (1,))
        labels[2, 6:] = torch.randint(0, 128, (6,))
        cls.labels = labels
        with torch.no_grad():
            cls.labelled = cls.model(inputs_embeds=cls.embeds, labels=labels, use_cache=False)
            cls.unlabelled = cls.model(inputs_embeds=cls.embeds, use_cache=False)

    def test_withholding_labels_still_yields_the_same_logits(self):
        """The sequence path depends on this: no labels, no HF loss, identical logits."""
        self.assertIsNone(self.unlabelled.loss)
        self.assertEqual(tuple(self.unlabelled.logits.shape), (3, 12, 128))
        self.assertTrue(torch.equal(self.labelled.logits, self.unlabelled.logits))

    def test_our_convention_reproduces_huggingfaces_own_loss(self):
        """If this drifts, `sequence_normalised_loss` is masking or shifting differently."""
        mine = token_mean(self.unlabelled.logits, self.labels)
        self.assertAlmostEqual(float(mine), float(self.labelled.loss), places=6)

    def test_sequence_norm_equals_the_mean_of_per_sequence_means(self):
        shift_logits, shift_labels = self.unlabelled.logits[:, :-1, :], self.labels[:, 1:]
        per = []
        for i in range(shift_labels.size(0)):
            keep = shift_labels[i] != -100
            per.append(float(F.cross_entropy(shift_logits[i][keep].float(),
                                             shift_labels[i][keep], reduction="mean")))
        expected = sum(per) / len(per)
        got = float(sequence_normalised_loss(self.unlabelled.logits, self.labels))
        self.assertAlmostEqual(got, expected, places=6)
        # Unequal spans (4 / 1 / 6), so the two reductions must not coincide.
        self.assertNotAlmostEqual(got, float(self.labelled.loss), places=6)


class FlagWiringTests(unittest.TestCase):
    """The parsed flag must reach the loss, not just exist in the parser.

    No unit test can see argparse-to-call-site wiring by running the trainer (it needs a
    GPU, cached features and a 7B model), so this reads the trainer's AST instead. A
    structural check is weaker than a behavioural one and is used only because the
    behavioural one is out of reach; it still fails if the keyword is dropped, which is
    the specific regression it exists to catch.
    """

    @classmethod
    def setUpClass(cls):
        import ast
        cls.ast = ast
        cls.tree = ast.parse((ROOT / "scripts" / "06c_train_bridge.py").read_text())

    def _calls_to(self, name):
        return [n for n in self.ast.walk(self.tree)
                if isinstance(n, self.ast.Call)
                and isinstance(n.func, self.ast.Name) and n.func.id == name]

    def _loss_norm_kwarg(self, call):
        for kw in call.keywords:
            if kw.arg == "loss_norm":
                return kw.value
        return None

    def test_every_loss_call_forwards_the_parsed_flag(self):
        calls = self._calls_to("vlm_loss_batch") + self._calls_to("eval_loss")
        self.assertGreaterEqual(len(calls), 3, "expected the train, val and eval_loss sites")
        for call in calls:
            value = self._loss_norm_kwarg(call)
            self.assertIsNotNone(
                value, f"vlm_loss_batch/eval_loss call at line {call.lineno} drops loss_norm")
            # Inside eval_loss the forwarded name is the local parameter, not args.*.
            rendered = self.ast.dump(value)
            self.assertTrue("args" in rendered or "loss_norm" in rendered,
                            f"line {call.lineno} forwards an unexpected value: {rendered}")

    def test_parser_offers_both_norms_and_defaults_to_legacy(self):
        """The default must stay 'token' or every rerun silently changes objective."""
        for node in self.ast.walk(self.tree):
            if (isinstance(node, self.ast.Call) and isinstance(node.func, self.ast.Attribute)
                    and node.func.attr == "add_argument" and node.args
                    and getattr(node.args[0], "value", None) == "--loss-norm"):
                kw = {k.arg: k.value for k in node.keywords}
                self.assertEqual(getattr(kw.get("default"), "value", None), "token")
                self.assertEqual(kw["choices"].id, "LOSS_NORMS")
                return
        self.fail("no --loss-norm argument found in the trainer")


class TrajectoryProvenanceTests(unittest.TestCase):
    """A checkpoint may not change objective on resume, and legacy ones are not 'missing'."""

    @classmethod
    def setUpClass(cls):
        cls.trainer = load_trainer_module()

    def _current(self, loss_norm):
        return {"lr": 1e-4, "weight_decay": 0.01, "batch_size": 8, "grad_accum": 4,
                "max_answer_tokens": 64, "warmup_frac": 0.0, "seed": 42,
                "loss_norm": loss_norm}

    def _legacy_ckpt(self):
        """A checkpoint written before loss_norm existed: every other field present."""
        return {"lr": 1e-4, "weight_decay": 0.01, "batch_size": 8, "grad_accum": 4,
                "max_answer_tokens": 64, "warmup_frac": 0.0, "seed": 42}

    def test_loss_norm_is_trajectory_defining(self):
        self.assertIn("loss_norm", self.trainer.TRAJECTORY_FIELDS)

    def test_legacy_checkpoint_is_not_reported_as_missing_provenance(self):
        """Its value is recoverable with certainty, so it must not demand the ack flag."""
        missing = self.trainer.validate_trajectory_provenance(
            self._current("token"), self._legacy_ckpt(), acknowledge_missing=False)
        self.assertNotIn("loss_norm", missing)

    def test_legacy_checkpoint_refuses_a_different_objective(self):
        """The load-bearing provenance check.

        A bare `.get("loss_norm")` would compare None against 'sequence', find no conflict,
        and silently continue a token-normalised run under a different objective. The
        recovered default is what gives this comparison teeth.
        """
        with self.assertRaisesRegex(SystemExit, "loss-norm"):
            self.trainer.validate_trajectory_provenance(
                self._current("sequence"), self._legacy_ckpt(), acknowledge_missing=False)

    def test_sequence_checkpoint_refuses_a_token_resume(self):
        ckpt = {**self._legacy_ckpt(), "loss_norm": "sequence"}
        with self.assertRaisesRegex(SystemExit, "loss-norm"):
            self.trainer.validate_trajectory_provenance(
                self._current("token"), ckpt, acknowledge_missing=False)

    def test_matching_objective_is_accepted(self):
        ckpt = {**self._legacy_ckpt(), "loss_norm": "sequence"}
        self.trainer.validate_trajectory_provenance(
            self._current("sequence"), ckpt, acknowledge_missing=False)

    def test_incomplete_trajectory_dict_is_refused_not_silently_skipped(self):
        """Omitting a field must not quietly disable that field's own conflict check."""
        partial = self._current("token")
        del partial["loss_norm"]
        with self.assertRaisesRegex(SystemExit, "missing loss_norm"):
            self.trainer.validate_trajectory_provenance(
                partial, self._legacy_ckpt(), acknowledge_missing=False)


if __name__ == "__main__":
    unittest.main()
