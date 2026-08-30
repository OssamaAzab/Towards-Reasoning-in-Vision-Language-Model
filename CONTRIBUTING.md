# Contributing

1. Open an issue describing the scientific or engineering change.
2. Never edit an existing frozen QID list, exclusion artifact, result row, or
   checkpoint tag in place.
3. Add a new filename/stem for supplementary experiments.
4. Write or update tests before implementation.
5. Run:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
   python -m compileall -q src scripts tests
   bash -n env.sh cluster/train.sbatch cluster/evaluate.sbatch
   sha256sum -c release/RELEASE_MANIFEST.sha256
   ```

6. Parse CSV with a real CSV parser and stop on any raw/source mismatch.
7. Do not commit datasets, checkpoints, caches, credentials, local paths, or
   third-party code without a verified redistribution licence.
8. Use conventional commit messages such as `fix:`, `feat:`, `docs:`, or `test:`.

Scientific interpretation and changes to evidence roles require owner review.
