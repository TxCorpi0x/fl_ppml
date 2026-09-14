# Regenerated results — cifar

Source: unique raw comparison_report.json payloads under results/.
Excluded 1 byte-identical preserved report copy/copies.

Measured elapsed: seconds between the first and last ModelCommit timestamps in the run's chain ledger (excludes setup and training before the round-1 commit). Overhead vs baseline uses this measurement.

Timer estimate: (client_fit + encryption + decryption + proof_generation + DP noise) / clients + proof_verification + server_aggregate, derived from component timers. n is the number of distinct report payloads backing the row.

Test metrics are means over all evaluated rounds, not the final-round model. Proofs/client/round is read from the run's chain ledger; it shows proofs were emitted, not that they verified.

| Mode | Rounds | Clients | n | Seeds | Measured elapsed s | Overhead vs baseline (measured) | Timer estimate s | Proof gen total s | Proof verify total s | Encryption total s | Upload bytes | Proofs/client/round | Test F1 (round mean) | Test accuracy (round mean) | Test AUPRC (round mean) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 20 | 3 | 1 | 42 | 520.6 | 0.0 s (0.0%) | 462.5 | 0.000 | 0.000 | 0.000 | 248024.0 | — | 47.511 | 47.237 | 51.454 |
| dp | 20 | 3 | 1 | 42 | 578.0 | 57.4 s (11.0%) | 514.5 | 0.000 | 0.000 | 0.000 | 248024.0 | — | 34.040 | 36.588 | 36.168 |
| he_concrete_tfhe | 20 | 3 | 1 | 42 | 523.3 | 2.8 s (0.5%) | 507.7 | 0.000 | 0.000 | 83.168 | 94743.6 | — | 46.314 | 47.655 | 49.806 |
| he_concrete_tfhe_zkp | 20 | 3 | 1 | 42 | 14598.1 | 14077.5 s (2704.1%) | 15376.8 | 44553.517 | 7.115 | 82.044 | 103175.6 | 39 | 47.442 | 48.591 | 50.938 |
| he_concrete_tfhe_zkp_dp | 20 | 3 | 1 | 42 | 15257.6 | 14737.0 s (2830.8%) | 16047.0 | 46298.683 | 7.175 | 83.630 | 114027.9 | 39 | 33.675 | 34.571 | 35.377 |
| he_tenseal | 20 | 3 | 1 | 42 | 494.0 | -26.6 s (-5.1%) | 477.8 | 0.000 | 0.000 | 0.986 | 230385.6 | — | 44.560 | 46.457 | 47.841 |
| he_tenseal_zkp | 20 | 3 | 1 | 42 | 14846.5 | 14325.9 s (2751.8%) | 15630.3 | 45312.462 | 7.208 | 0.845 | 239010.2 | 39 | 47.103 | 49.847 | 50.841 |
| he_tenseal_zkp_dp | 20 | 3 | 1 | 42 | 15083.5 | 14562.9 s (2797.3%) | 15854.1 | 45847.204 | 7.547 | 0.890 | 239574.5 | 39 | 33.634 | 36.222 | 35.459 |
| zkp | 20 | 3 | 1 | 42 | 15126.7 | 14606.1 s (2805.7%) | 15352.4 | 44422.694 | 44.845 | 0.000 | 256604.0 | 39 | 45.954 | 47.836 | 49.938 |
| zkp_sampled | 20 | 3 | 1 | 42 | 14993.9 | 14473.3 s (2780.1%) | 15216.8 | 44010.398 | 44.703 | 0.000 | 256604.0 | 39 | 45.870 | 47.744 | 49.158 |

**zkp_sampled is not an independent configuration.** The harness maps it to internal mode `zkp` and passes only `--zkp`, so these rows are a second run of full ZKP (identical proofs/client/round). See audit/numbers.md.
