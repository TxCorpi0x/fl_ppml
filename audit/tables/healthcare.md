# Regenerated results — healthcare

Source: unique raw comparison_report.json payloads under results/.
Excluded 1 byte-identical preserved report copy/copies.

Measured elapsed: seconds between the first and last ModelCommit timestamps in the run's chain ledger (excludes setup and training before the round-1 commit). Overhead vs baseline uses this measurement.

Timer estimate: (client_fit + encryption + decryption + proof_generation + DP noise) / clients + proof_verification + server_aggregate, derived from component timers. n is the number of distinct report payloads backing the row.

Test metrics are means over all evaluated rounds, not the final-round model. Proofs/client/round is read from the run's chain ledger; it shows proofs were emitted, not that they verified.

| Mode | Rounds | Clients | n | Seeds | Measured elapsed s | Overhead vs baseline (measured) | Timer estimate s | Proof gen total s | Proof verify total s | Encryption total s | Upload bytes | Proofs/client/round | Test F1 (round mean) | Test accuracy (round mean) | Test AUPRC (round mean) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2 | 2 | 2 | 42 | 0.0 ± 0.0 | 0.0 s (0.0%) | 0.1 ± 0.0 | 0.000 | 0.000 | 0.000 | 11656.0 | — | 82.827 ± 2.667 | 75.000 ± 0.687 | 86.452 ± 3.428 |
| baseline | 20 | 3 | 1 | 42 | 7.7 | 0.0 s (0.0%) | 6.7 | 0.000 | 0.000 | 0.000 | 11656.0 | — | 84.515 | 81.907 | 84.224 |
| dp | 20 | 3 | 1 | 42 | 9.1 | 1.4 s (18.4%) | 8.1 | 0.000 | 0.000 | 0.000 | 11656.0 | — | 84.967 | 81.464 | 81.075 |
| he_concrete_tfhe | 20 | 3 | 1 | 42 | 176.8 | 169.0 s (2189.3%) | 159.6 | 0.000 | 0.000 | 208.452 | 18679335.1 | — | 83.923 | 68.345 | 81.177 |
| he_concrete_tfhe_zkp | 20 | 3 | 1 | 42 | 888.5 | 880.8 s (11407.5%) | 910.1 | 2221.960 | 1.115 | 220.920 | 18680876.1 | 7 | 83.908 | 66.410 | 80.896 |
| he_concrete_tfhe_zkp_dp | 20 | 3 | 1 | 42 | 891.2 | 883.4 s (11441.8%) | 914.8 | 2225.738 | 1.132 | 222.220 | 18680876.9 | 7 | 84.367 | 68.514 | 82.831 |
| he_elgamal_zkp | 2 | 2 | 1 | 42 | 84.9 | 84.9 s (222295.5%) | 219.1 | 435.724 | 0.533 | 0.000 | 192240.0 | 26 | 82.041 | 72.431 | 81.440 |
| he_tenseal | 20 | 3 | 1 | 42 | 13.0 | 5.3 s (68.6%) | 9.8 | 0.000 | 0.000 | 2.507 | 673619.2 | — | 86.139 | 80.746 | 81.694 |
| he_tenseal_zkp | 2 | 2 | 2 | 42 | 9.3 ± 0.1 | 9.2 ± 0.1 s (26273.6 ± 2734.4%) | 32.1 ± 15.1 | 63.467 ± 30.426 | 0.055 ± 0.001 | 0.150 ± 0.108 | 1336658.2 ± 934968.4 | 7 | 82.043 ± 10.016 | 57.014 ± 9.428 | 81.321 ± 9.256 |
| he_tenseal_zkp | 20 | 3 | 1 | 42 | 690.6 | 682.9 s (8844.3%) | 724.3 | 2139.288 | 1.089 | 3.108 | 675199.8 | 7 | 87.022 | 81.622 | 85.654 |
| he_tenseal_zkp_dp | 20 | 3 | 1 | 42 | 696.3 | 688.6 s (8918.6%) | 730.7 | 2152.222 | 1.122 | 3.354 | 675221.0 | 7 | 86.137 | 79.841 | 84.327 |
| zkp | 20 | 3 | 1 | 42 | 660.6 | 652.9 s (8455.6%) | 691.4 | 2046.149 | 2.883 | 0.000 | 13196.0 | 7 | 85.510 | 82.773 | 84.586 |
| zkp_sampled | 20 | 3 | 1 | 42 | 680.8 | 673.1 s (8717.8%) | 692.7 | 2049.221 | 2.873 | 0.000 | 13196.0 | 7 | 86.660 | 81.974 | 87.584 |

**zkp_sampled is not an independent configuration.** The harness maps it to internal mode `zkp` and passes only `--zkp`, so these rows are a second run of full ZKP (identical proofs/client/round). See audit/numbers.md.
