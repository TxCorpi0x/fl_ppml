# Regenerated results — creditcard

Source: unique raw comparison_report.json payloads under results/.
Excluded 1 byte-identical preserved report copy/copies.

Measured elapsed: seconds between the first and last ModelCommit timestamps in the run's chain ledger (excludes setup and training before the round-1 commit). Overhead vs baseline uses this measurement.

Timer estimate: (client_fit + encryption + decryption + proof_generation + DP noise) / clients + proof_verification + server_aggregate, derived from component timers. n is the number of distinct report payloads backing the row.

Test metrics are means over all evaluated rounds, not the final-round model. Proofs/client/round is read from the run's chain ledger; it shows proofs were emitted, not that they verified.

| Mode | Rounds | Clients | n | Seeds | Measured elapsed s | Overhead vs baseline (measured) | Timer estimate s | Proof gen total s | Proof verify total s | Encryption total s | Upload bytes | Proofs/client/round | Test F1 (round mean) | Test accuracy (round mean) | Test AUPRC (round mean) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 20 | 3 | 1 | 42 | 1386.6 | 0.0 s (0.0%) | 1364.2 | 0.000 | 0.000 | 0.000 | 16520.0 | — | 84.203 | 99.946 | 79.675 |
| dp | 20 | 3 | 1 | 42 | 1550.9 | 164.4 s (11.9%) | 1529.0 | 0.000 | 0.000 | 0.000 | 16520.0 | — | 81.115 | 99.930 | 71.972 |
| he_concrete_tfhe | 20 | 3 | 1 | 42 | 1554.5 | 168.0 s (12.1%) | 1579.0 | 0.000 | 0.000 | 257.580 | 26473829.8 | — | 81.883 | 99.936 | 74.403 |
| he_concrete_tfhe_zkp | 20 | 3 | 1 | 42 | 2617.6 | 1231.0 s (88.8%) | 2671.1 | 3072.220 | 1.156 | 283.806 | 26475371.2 | 7 | 86.676 | 99.935 | 81.675 |
| he_concrete_tfhe_zkp_dp | 20 | 3 | 1 | 42 | 2839.1 | 1452.5 s (104.8%) | 2895.6 | 3036.928 | 1.122 | 276.444 | 26475371.7 | 7 | 76.733 | 99.926 | 66.288 |
| he_tenseal | 20 | 3 | 1 | 42 | 1274.6 | -112.0 s (-8.1%) | 1261.7 | 0.000 | 0.000 | 3.084 | 673723.6 | — | 80.029 | 99.939 | 74.728 |
| he_tenseal_zkp | 20 | 3 | 1 | 42 | 2232.4 | 845.8 s (61.0%) | 2285.2 | 2930.102 | 1.100 | 4.166 | 675227.6 | 7 | 81.019 | 99.944 | 76.552 |
| he_tenseal_zkp_dp | 20 | 3 | 1 | 42 | 2436.4 | 1049.8 s (75.7%) | 2495.6 | 2900.781 | 1.116 | 4.050 | 675226.5 | 7 | 82.467 | 99.919 | 73.506 |
| zkp | 20 | 3 | 1 | 42 | 2256.9 | 870.3 s (62.8%) | 2335.9 | 3230.887 | 3.638 | 0.000 | 18060.0 | 7 | 85.954 | 99.941 | 80.341 |
| zkp_sampled | 20 | 3 | 1 | 42 | 2242.1 | 855.5 s (61.7%) | 2191.7 | 2826.827 | 3.612 | 0.000 | 18060.0 | 7 | 85.476 | 99.940 | 80.707 |

**zkp_sampled is not an independent configuration.** The harness maps it to internal mode `zkp` and passes only `--zkp`, so these rows are a second run of full ZKP (identical proofs/client/round). See audit/numbers.md.
