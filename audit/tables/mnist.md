# Regenerated results — mnist

Source: unique raw comparison_report.json payloads under results/.
Excluded 1 byte-identical preserved report copy/copies.

Measured elapsed: seconds between the first and last ModelCommit timestamps in the run's chain ledger (excludes setup and training before the round-1 commit). Overhead vs baseline uses this measurement.

Timer estimate: (client_fit + encryption + decryption + proof_generation + DP noise) / clients + proof_verification + server_aggregate, derived from component timers. n is the number of distinct report payloads backing the row.

Test metrics are means over all evaluated rounds, not the final-round model. Proofs/client/round is read from the run's chain ledger; it shows proofs were emitted, not that they verified.

| Mode | Rounds | Clients | n | Seeds | Measured elapsed s | Overhead vs baseline (measured) | Timer estimate s | Proof gen total s | Proof verify total s | Encryption total s | Upload bytes | Proofs/client/round | Test F1 (round mean) | Test accuracy (round mean) | Test AUPRC (round mean) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 20 | 3 | 1 | 42 | 950.6 | 0.0 s (0.0%) | 908.2 | 0.000 | 0.000 | 0.000 | 177704.0 | — | 97.734 | 97.766 | 99.573 |
| dp | 20 | 3 | 1 | 42 | 1071.2 | 120.6 s (12.7%) | 1026.6 | 0.000 | 0.000 | 0.000 | 177704.0 | — | 93.933 | 93.946 | 98.057 |
| he_concrete_tfhe | 20 | 3 | 1 | 42 | 1016.0 | 65.4 s (6.9%) | 1043.5 | 0.000 | 0.000 | 83.240 | 71038.1 | — | 97.522 | 97.538 | 99.489 |
| he_concrete_tfhe_zkp | 20 | 3 | 1 | 42 | 10741.4 | 9790.8 s (1030.0%) | 11286.7 | 30874.138 | 5.405 | 81.129 | 77790.6 | 31 | 97.395 | 97.424 | 99.435 |
| he_concrete_tfhe_zkp_dp | 20 | 3 | 1 | 42 | 10954.5 | 10004.0 s (1052.4%) | 11506.4 | 31324.555 | 5.421 | 80.025 | 81416.2 | 31 | 94.789 | 95.157 | 98.340 |
| he_tenseal | 20 | 3 | 1 | 42 | 982.8 | 32.2 s (3.4%) | 966.5 | 0.000 | 0.000 | 0.686 | 165189.2 | — | 97.179 | 97.101 | 99.395 |
| he_tenseal_zkp | 20 | 3 | 1 | 42 | 10871.4 | 9920.8 s (1043.7%) | 11419.1 | 31305.805 | 5.596 | 0.570 | 171976.6 | 31 | 97.122 | 97.118 | 99.408 |
| he_tenseal_zkp_dp | 20 | 3 | 1 | 42 | 10741.4 | 9790.8 s (1030.0%) | 11273.9 | 30754.871 | 5.492 | 0.586 | 172307.1 | 31 | 94.260 | 93.919 | 98.225 |
| zkp | 20 | 3 | 1 | 42 | 11274.9 | 10324.3 s (1086.1%) | 11441.3 | 31244.698 | 31.900 | 0.000 | 184524.0 | 31 | 97.620 | 97.557 | 99.557 |
| zkp_sampled | 20 | 3 | 1 | 42 | 10981.3 | 10030.7 s (1055.2%) | 11120.7 | 30407.000 | 31.912 | 0.000 | 184524.0 | 31 | 97.596 | 97.573 | 99.499 |

**zkp_sampled is not an independent configuration.** The harness maps it to internal mode `zkp` and passes only `--zkp`, so these rows are a second run of full ZKP (identical proofs/client/round). See audit/numbers.md.
