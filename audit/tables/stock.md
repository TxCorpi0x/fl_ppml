# Regenerated results — stock

Source: unique raw comparison_report.json payloads under results/.
Excluded 1 byte-identical preserved report copy/copies.

Measured elapsed: seconds between the first and last ModelCommit timestamps in the run's chain ledger (excludes setup and training before the round-1 commit). Overhead vs baseline uses this measurement.

Timer estimate: (client_fit + encryption + decryption + proof_generation + DP noise) / clients + proof_verification + server_aggregate, derived from component timers. n is the number of distinct report payloads backing the row.

Test metrics are means over all evaluated rounds, not the final-round model. Proofs/client/round is read from the run's chain ledger; it shows proofs were emitted, not that they verified.

| Mode | Rounds | Clients | n | Seeds | Measured elapsed s | Overhead vs baseline (measured) | Timer estimate s | Proof gen total s | Proof verify total s | Encryption total s | Upload bytes | Proofs/client/round | Test F1 (round mean) | Test accuracy (round mean) | Test AUPRC (round mean) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 20 | 3 | 1 | 42 | 18.4 | 0.0 s (0.0%) | 16.8 | 0.000 | 0.000 | 0.000 | 11656.0 | — | 71.748 | 55.068 | 54.895 |
| dp | 20 | 3 | 1 | 42 | 24.5 | 6.1 s (33.3%) | 22.8 | 0.000 | 0.000 | 0.000 | 11656.0 | — | 72.213 | 52.656 | 57.363 |
| he_concrete_tfhe | 20 | 3 | 1 | 42 | 193.0 | 174.6 s (949.0%) | 175.8 | 0.000 | 0.000 | 213.992 | 18679334.8 | — | 71.745 | 55.556 | 60.603 |
| he_concrete_tfhe_zkp | 20 | 3 | 1 | 42 | 908.9 | 890.5 s (4840.2%) | 930.4 | 2231.422 | 1.113 | 221.001 | 18680877.6 | 7 | 72.652 | 55.556 | 52.863 |
| he_concrete_tfhe_zkp_dp | 20 | 3 | 1 | 42 | 913.8 | 895.4 s (4866.6%) | 936.7 | 2236.830 | 1.155 | 226.093 | 18680875.2 | 7 | 71.848 | 56.065 | 57.857 |
| he_tenseal | 20 | 3 | 1 | 42 | 26.0 | 7.6 s (41.3%) | 22.3 | 0.000 | 0.000 | 2.689 | 673626.7 | — | 72.209 | 55.489 | 53.120 |
| he_tenseal_zkp | 20 | 3 | 1 | 42 | 703.6 | 685.2 s (3724.4%) | 739.0 | 2138.442 | 1.094 | 3.284 | 675208.8 | 7 | 73.425 | 55.671 | 57.480 |
| he_tenseal_zkp_dp | 20 | 3 | 1 | 42 | 718.8 | 700.4 s (3807.0%) | 754.7 | 2164.727 | 1.090 | 3.120 | 675228.1 | 7 | 71.987 | 54.683 | 56.125 |
| zkp | 20 | 3 | 1 | 42 | 677.0 | 658.6 s (3579.7%) | 688.3 | 1991.240 | 2.875 | 0.000 | 13196.0 | 7 | 71.217 | 52.998 | 57.894 |
| zkp_sampled | 20 | 3 | 1 | 42 | 693.7 | 675.3 s (3670.3%) | 704.5 | 2045.953 | 2.983 | 0.000 | 13196.0 | 7 | 73.597 | 52.334 | 61.087 |

**zkp_sampled is not an independent configuration.** The harness maps it to internal mode `zkp` and passes only `--zkp`, so these rows are a second run of full ZKP (identical proofs/client/round). See audit/numbers.md.
