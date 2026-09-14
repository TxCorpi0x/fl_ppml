# Working rules for this repository

This is the codebase behind an accepted MSc thesis on privacy-preserving
federated learning. It is being hardened for open-source release and for
peer-reviewed publication. Security claims in this repo will be read by
reviewers and by security engineers.

## Ground rules

1. **Verify before you fix.** Every defect report is a hypothesis. Confirm it
   against the actual code and, where feasible, a reproducing test. If a
   hypothesis is wrong, say so plainly and show why.
2. **Report before you edit.** When asked to investigate, write findings to
   `audit/<topic>.md` and stop. No source edits until explicitly approved.
3. **Distinguish claim from measurement.** Never describe a security property
   as verified unless a test exercises it. Prover/verifier timing is a cost
   measurement, not a soundness result.
4. **Fail closed.** In any security-relevant path, the correct behaviour on
   error or ambiguity is to abort, not to continue with a degraded guarantee.
5. **No silent scope expansion.** If a fix requires touching code outside the
   named area, stop and ask.
6. **Numbers come from raw results.** `results/**/comparison_report.json` is
   the source of truth. `docs/**` and the thesis text may contain stale or
   contradictory figures; do not cite them as evidence.

## Layout

- `fl/privacy/` — privacy-mode plugins (`base.py`, `registry.py`, `zkp.py`,
  `he_zkp.py`, `he_tenseal.py`, `he_concrete_tfhe.py`, `dp.py`)
- `fl/core/` — internals including `zkp_gnark.py`, `security.py`, and
  `benchmark.py`
- `zkp_gnark_service/` — Go Groth16 service (`main.go`)
- `fl/chain.py`, `fl/chain_contract/FLLedger.sol` — audit ledger
- `fl/compare/` — sweep orchestration and reporting
- `results/<dataset>/<timestamp>/comparison_report.json` — raw benchmark output
