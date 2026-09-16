# Contributing

Thanks for your interest. This project implements cryptographic protocols, so
contributions are held to a stricter standard than the average Python library:
a claim about what a mode guarantees has to be backed by a test.

## Contributor Licence Agreement

Before a pull request can be merged you must sign the CLA, which the CLA
assistant will prompt for on your first pull request. It licenses your
contribution to CorpiXo, which holds copyright for the project, so the project
can be relicensed or dual-licensed in future without tracking down every
contributor. You keep the copyright in your own work.

## Development setup

Python 3.12 is required (Concrete-ML needs < 3.13, Flower 1.36 needs > 3.11).

```bash
conda create -n ppflx python=3.12 -y && conda activate ppflx
pip install -r requirements.txt
python -m ppflx.keys generate he_tenseal        # HE keys
python -m ppflx.keys generate dp --output keys/dp/dp_params.json
cd zkp_gnark_service && go build -o gnark_service .   # proof service
```

## Before opening a pull request

1. `pytest tests` passes (the ZKP tests skip without the proof service binary).
2. `go test ./...` passes in `zkp_gnark_service/` if you touched Go code.
3. If you changed a privacy mode, the strategy or the harness, run the modes
   your change affects end to end and say so in the pull request:
   `python compare.py --dataset healthcare --modes <modes> --rounds 2 --num-clients 2`
4. New behaviour in a security-relevant path comes with a test that fails
   without your change.

## What the project expects of a change

- **Fail closed.** On error or ambiguity in a security-relevant path, abort;
  do not continue with a weaker guarantee.
- **Claims match tests.** Do not describe a property as verified unless a test
  exercises it. Prover and verifier timings are cost measurements, not
  soundness results.
- **Numbers come from raw results.** `results/**/comparison_report.json` is the
  source of truth for benchmark figures; documentation follows it, not the
  other way round.
- **No key material or datasets in commits.** Keys are generated locally;
  datasets are downloaded from their own sources.

## Reporting a vulnerability

Do not open a public issue. See [SECURITY.md](SECURITY.md).
