# Release checklist

- [ ] Confirm ownership and the MIT licence choice.
- [ ] Review `THIRD_PARTY_NOTICES.md`; do not add unlicensed upstream files.
- [ ] Run the full CPU suite and manifest verification.
- [ ] Confirm the repository has one fresh-history commit and no source history.
- [ ] Confirm no dataset images, question/gold corpus, checkpoints, secrets, or
      institutional paths are staged.
- [ ] Inspect every README figure at GitHub rendering size.
- [ ] Verify `artifacts/model_revisions.json` against the intended upstream snapshots.
- [ ] Create a signed `v1.0.0` tag after the first reviewed public commit.
- [ ] Optionally archive the tagged source release on Zenodo for a DOI.
- [ ] Record release commit and archive DOI in `CITATION.cff` without rewriting
      historical scientific artifacts.

Model/checkpoint hosting is intentionally outside this release plan.
