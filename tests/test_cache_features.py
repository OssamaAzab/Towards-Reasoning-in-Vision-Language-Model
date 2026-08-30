"""Contracts for exact mixed-set feature caching."""
import hashlib
import importlib.util
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "06a_cache_features.py"
LAUNCHER = ROOT / "outputs" / "diagnostics" / "t039_dinov2_cache.sbatch"


def load_cache_script():
    """Load the numerically prefixed cache script as a module."""
    spec = importlib.util.spec_from_file_location("cache_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDataset:
    """Minimal indexed dataset exposing image paths."""

    def __init__(self, prefix: str, size: int):
        self.prefix = prefix
        self.size = size

    def __len__(self):
        return self.size

    def get(self, index: int):
        return types.SimpleNamespace(image_path=Path(f"{self.prefix}_{index}.jpg"))


class CacheFeatureContracts(unittest.TestCase):
    """Require explicit encoder selection and the exact mixed training manifest."""

    @classmethod
    def setUpClass(cls):
        cls.cache_script = load_cache_script()

    def test_mixed_source_matches_training_split(self):
        llava = FakeDataset("llava", 10)
        vqa = FakeDataset("vqa", 20)
        paths = self.cache_script.select_image_paths(
            llava, vqa, source="mixed", limit=10, short_frac=0.7
        )
        self.assertEqual([path.stem for path in paths[:3]], ["llava_0", "llava_1", "llava_2"])
        self.assertEqual([path.stem for path in paths[3:]], [f"vqa_{i}" for i in range(7)])

    def test_manifest_is_deduplicated_by_cache_key(self):
        paths = [Path("a/one.jpg"), Path("b/one.png"), Path("b/two.jpg")]
        unique = self.cache_script.unique_image_paths(paths)
        self.assertEqual(unique, [Path("a/one.jpg"), Path("b/two.jpg")])

    def test_expected_manifest_is_fail_closed(self):
        paths = [Path("one.jpg"), Path("two.jpg")]
        digest = hashlib.sha256(b"one\ntwo\n").hexdigest()
        result = self.cache_script.validate_manifest(paths, 2, digest)
        self.assertEqual(result, {"count": 2, "sha256": digest})
        with self.assertRaisesRegex(SystemExit, "manifest mismatch"):
            self.cache_script.validate_manifest(paths, 3, digest)

    def test_dinov2_launcher_uses_exact_mixed_manifest(self):
        if not LAUNCHER.is_file():
            self.skipTest("historical site launcher is intentionally absent from the clean release")
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=teaching", text)
        self.assertIn("#SBATCH --gpus=1", text)
        self.assertIn("--encoder dinov2", text)
        self.assertIn("--source mixed", text)
        self.assertIn("--limit 500000", text)
        self.assertIn("--short-frac 0.7", text)
        self.assertIn("--expected-keys 143188", text)
        self.assertIn("dc2e4260fded8435ae2f9a4400e9c75edd5b31917c306b9c7e8ef25c6c841c4e", text)
        self.assertIn("PYTHONUNBUFFERED=1", text)


if __name__ == "__main__":
    unittest.main()
