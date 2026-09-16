# Security policy

## Reporting a vulnerability

Report privately through GitHub's "Report a vulnerability" button on the
Security tab of this repository. Please include what you attacked, the
configuration, and a reproduction if you have one.

We aim to acknowledge a report within five working days and to agree a
disclosure timeline with you. Please do not open a public issue first.

## What is and is not a security boundary

These are properties the project claims, and deliberately does not claim.
Reports that these documented limitations exist are not vulnerabilities;
reports that a claimed property fails are.

**Claimed:**

- `zkp`: each client's update is bound to the model it downloaded, and its norm
  is bounded, proved with Groth16 over every coordinate.
- `he_elgamal_zkp` and `he_elgamal_zkp_sampled`: the proof is bound to the
  ciphertext the server aggregates.
- HE modes: the server sees only ciphertexts of client updates.
- `dp`: an (epsilon, delta) differential-privacy guarantee for the published
  model, under the recorded parameters.

**Not claimed:**

- The CKKS and TFHE composites (`he_tenseal_zkp`, `he_concrete_tfhe_zkp` and
  their `_dp` variants) are **confidentiality-only**. Their proof covers a
  client-chosen vector and is not bound to the aggregated ciphertext.
- The Pedersen ZKP backend is an unverified stub; it provides no Byzantine
  protection and is not for production.
- The Groth16 trusted setup is single-party. The setup randomness is not
  recoverable from the published files, but nothing proves it was destroyed.
- An update within the norm bound can still be malicious: the bound limits a
  client's per-round influence, not the direction of its update.
- `zkp_sampled` is a benchmark configuration over plaintext updates.

See `docs/ZKP.md` for the protocols and their limitations in full.
