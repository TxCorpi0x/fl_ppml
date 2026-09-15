"""Key-file formats, simulation marking and backend recording.

No gnark service needed.
"""

import json
import pickle
from types import SimpleNamespace as NS

import pytest


# ─── Key and parameter files are never unpickled ─────────────────────────────


class _Boom:
    """Unpickling this would run code; loaders must refuse the file before that."""

    def __reduce__(self):
        return (pytest.fail, ("a key file was unpickled",))


def _legacy_pickle(path):
    with open(path, "wb") as f:
        pickle.dump({"contexte": _Boom()}, f)


def test_tenseal_key_files_round_trip_and_refuse_legacy_pickles(tmp_path):
    from fl.keys import he_tenseal

    secret, public = tmp_path / "secret_context.bin", tmp_path / "public_context.bin"
    he_tenseal.generate(secret_path=str(secret), public_path=str(public))
    assert he_tenseal.load_client(str(secret)).is_private()
    assert not he_tenseal.load_server(str(public)).is_private()
    with pytest.raises(ValueError, match="secret key"):
        he_tenseal.load_server(str(secret))

    legacy = tmp_path / "secret_key.pkl"
    _legacy_pickle(legacy)
    with pytest.raises(ValueError, match="not a TenSEAL key file"):
        he_tenseal.load_client(str(legacy))


def test_dp_params_are_json_and_legacy_pickles_are_refused(tmp_path):
    from fl.keys import dp

    path = tmp_path / "dp_params.json"
    dp.generate(output=str(path), epsilon=1.0, delta=1e-5)
    assert json.loads(path.read_text())["epsilon"] == 1.0
    assert dp.load(str(path)).epsilon == 1.0

    legacy = tmp_path / "dp_params.pkl"
    _legacy_pickle(legacy)
    with pytest.raises(ValueError, match="not a JSON parameter file"):
        dp.load(str(legacy))


def test_pedersen_params_are_json_and_legacy_pickles_are_refused(tmp_path):
    from fl.core.zkp import create_zkp_context, read_zkp_params, write_zkp_params

    path = tmp_path / "zkp_params.json"
    ctx = create_zkp_context(bit_length=256)
    write_zkp_params(str(path), ctx)
    loaded = read_zkp_params(str(path))
    assert (loaded.p, loaded.q, loaded.g, loaded.h) == (ctx.p, ctx.q, ctx.g, ctx.h)

    legacy = tmp_path / "zkp_params.pkl"
    _legacy_pickle(legacy)
    with pytest.raises(ValueError, match="not a JSON ZKP parameter file"):
        read_zkp_params(str(legacy))


# ─── Simulation is opt-in and visible in results ─────────────────────────────


def test_config_defaults_to_real_transport():
    from fl.config import FLConfig

    assert FLConfig().sim_mode is False


def test_benchmark_records_transport_and_zkp_backend():
    from fl.core.benchmark import init_benchmark

    summary = init_benchmark("he_elgamal_zkp", 3, 2, transport="network", zkp_backend="gnark").summary()
    assert summary["transport"] == "network" and summary["zkp_backend"] == "gnark"
    assert init_benchmark("dp", 3, 2, transport="simulated").summary()["zkp_backend"] is None
    with pytest.raises(ValueError):
        init_benchmark("dp", 3, 2, transport="plaintext")


def test_report_marks_simulated_he_results(capsys):
    from fl.compare.report import print_summary

    bm = lambda transport: {"transport": transport, "rounds": 1, "num_clients": 2, "timing": {}, "model_quality": {}}
    print_summary([
        {"mode": "he_tenseal", "success": True, "benchmark": bm("simulated")},
        {"mode": "zkp", "success": True, "benchmark": bm("network")},
    ])
    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines() if line.startswith(("he_tenseal", "zkp"))}
    assert "[SIM]" in lines["he_tenseal"] and "[SIM]" not in lines["zkp"]


def test_simulated_result_never_replaces_a_networked_one(tmp_path):
    from fl.compare.runner import _merge_into_dataset_report

    networked = {"mode": "he_tenseal", "success": True, "benchmark": {"transport": "network", "rounds": 20}}
    (tmp_path / "healthcare").mkdir()
    report = tmp_path / "healthcare" / "comparison_report.json"
    report.write_text(json.dumps([networked]))

    simulated = {"mode": "he_tenseal", "success": True, "benchmark": {"transport": "simulated", "rounds": 2}}
    _merge_into_dataset_report([simulated], str(tmp_path), "healthcare")
    assert json.loads(report.read_text()) == [networked]

    newer = {"mode": "he_tenseal", "success": True, "benchmark": {"transport": "network", "rounds": 5}}
    _merge_into_dataset_report([newer], str(tmp_path), "healthcare")
    assert json.loads(report.read_text()) == [newer]
