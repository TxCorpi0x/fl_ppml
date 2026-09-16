"""Test ppflx.keys package — run with: conda run -n flEnv python3 tests/test_fl_keys.py"""

import sys, os, io, tempfile
from contextlib import redirect_stdout
from pathlib import Path

# Always run relative to the fl_ppml project root
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

results = {}


def check(name, fn):
    try:
        fn()
        results[name] = "PASS"
    except Exception as e:
        results[name] = f"FAIL: {e}"


# 1. Top-level imports + mode list
def t_imports():
    from ppflx.keys import list_modes, generate, load

    assert sorted(list_modes()) == [
        "dp",
        "he_concrete_tfhe",
        "he_tenseal",
        "zkp",
    ]


check("imports", t_imports)


# 2. Per-mode module import
def _mod(m):
    import importlib

    importlib.import_module(m)


for name, mod in [
    ("he_tenseal_mod", "ppflx.keys.he_tenseal"),
    ("dp_mod", "ppflx.keys.dp"),
    ("zkp_mod", "ppflx.keys.zkp"),
    ("concrete_tfhe_mod", "ppflx.keys.concrete_tfhe"),
]:
    check(name, lambda m=mod: _mod(m))


# 3. Prebuilt key bundles
def t_prebuilt():
    from ppflx.keys.concrete_tfhe import list_prebuilt, load_prebuilt, prebuilt_dir

    assert prebuilt_dir.exists(), f"prebuilt dir missing: {prebuilt_dir}"
    bundles = list_prebuilt()
    if not bundles:
        # Fresh clone — bundles are gitignored (private key material).
        # Directory presence is sufficient; skip load test.
        print(
            f"  [skip] no prebuilt bundles in {prebuilt_dir} (run 'python -m ppflx.keys generate he_concrete_tfhe' to create)"
        )
        return
    b = load_prebuilt(bundles[0])
    assert set(b.keys()) == {"secret_keys", "eval_keys", "client_zip", "server_zip"}


check("prebuilt_bundles", t_prebuilt)


# 4. Old concrete_tfhe_keys dir is gone
def t_old_dir_gone():
    from ppflx.keys.concrete_tfhe import prebuilt_dir

    old = prebuilt_dir.parent.parent / "concrete_tfhe_keys"
    assert not old.exists(), f"old dir still exists: {old}"


check("old_dir_removed", t_old_dir_gone)


# 5. CLI list
def t_cli_list():
    from ppflx.keys.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["list"])
    for m in ["he_tenseal", "dp", "zkp", "he_concrete_tfhe"]:
        assert m in buf.getvalue(), f"missing {m} in cli list output"


check("cli_list", t_cli_list)


# 6. CLI prebuilt
def t_cli_prebuilt():
    from ppflx.keys.cli import main
    from ppflx.keys.concrete_tfhe import list_prebuilt

    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["prebuilt"])
    out = buf.getvalue()
    if list_prebuilt():
        # Bundles present — expect listing
        assert "bw14_nc2" in out, f"expected bundle names in output, got: {out!r}"
    else:
        # Fresh clone — expect the "no bundles" message
        assert "No prebuilt bundles found" in out, f"unexpected output: {out!r}"


check("cli_prebuilt", t_cli_prebuilt)


# 7. make_tenseal_context alias
def t_alias():
    from ppflx.core.security import make_tenseal_context, context

    assert make_tenseal_context is context


check("tenseal_alias", t_alias)


# 8. DP generate + load roundtrip
def t_dp_roundtrip():
    from ppflx.keys.dp import generate, load

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp = tf.name
    try:
        p = generate(output=tmp, epsilon=2.5, delta=1e-6, overwrite=True)
        p2 = load(tmp)
        assert abs(p2.epsilon - 2.5) < 1e-9, f"epsilon mismatch: {p2.epsilon}"
        assert abs(p2.delta - 1e-6) < 1e-12, f"delta mismatch: {p2.delta}"
    finally:
        os.unlink(tmp)


check("dp_roundtrip", t_dp_roundtrip)


# 9. Overwrite protection
def t_overwrite_protection():
    from ppflx.keys.dp import generate

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp = tf.name
    try:
        generate(output=tmp, overwrite=True)
        try:
            generate(output=tmp, overwrite=False)
            raise AssertionError("should have raised FileExistsError")
        except FileExistsError:
            pass
    finally:
        os.unlink(tmp)


check("overwrite_protection", t_overwrite_protection)


# 10. load() raises FileNotFoundError on missing file
def t_missing_file():
    from ppflx.keys.dp import load

    try:
        load("/nonexistent/dp_params.json")
        raise AssertionError("should have raised FileNotFoundError")
    except FileNotFoundError:
        pass


check("missing_file_error", t_missing_file)


# 11. Root create_*.py scripts are gone (logic lives in ppflx.keys)
def t_scripts_removed():
    for f in [
        "create_keys.py",
        "create_dp_params.py",
        "create_zkp_params.py",
        "create_concrete_params.py",
    ]:
        assert not Path(f).exists(), f"{f} still exists — should be deleted"


check("scripts_removed", t_scripts_removed)


# 12. fl source tree sanity
def t_tree():
    keys_dir = Path("ppflx/keys")
    expected = {
        "__init__.py",
        "__main__.py",
        "cli.py",
        "he_tenseal.py",
        "dp.py",
        "zkp.py",
        "concrete_tfhe.py",
    }
    actual = {p.name for p in keys_dir.iterdir() if p.suffix == ".py"}
    missing = expected - actual
    assert not missing, f"missing files: {missing}"
    assert (keys_dir / "prebuilt").is_dir(), "prebuilt/ subdir missing"


check("tree_sanity", t_tree)


# 13. Default key paths use the unified keys/<type>/ directory
def t_default_paths():
    import inspect
    from ppflx.keys import he_tenseal, dp, zkp
    from ppflx.config import FLConfig

    cfg = FLConfig()
    assert (
        cfg.he_tenseal_secret_path == "keys/he_tenseal/secret_context.bin"
    ), cfg.he_tenseal_secret_path
    assert (
        cfg.he_tenseal_public_path == "keys/he_tenseal/public_context.bin"
    ), cfg.he_tenseal_public_path
    assert cfg.zkp_params_path == "keys/zkp/zkp_params.json", cfg.zkp_params_path
    assert cfg.dp_params_path == "keys/dp/dp_params.json", cfg.dp_params_path

    sig_ts = inspect.signature(he_tenseal.generate)
    assert sig_ts.parameters["secret_path"].default == "keys/he_tenseal/secret_context.bin"
    assert sig_ts.parameters["public_path"].default == "keys/he_tenseal/public_context.bin"

    sig_dp = inspect.signature(dp.generate)
    assert sig_dp.parameters["output"].default == "keys/dp/dp_params.json"

    sig_zkp = inspect.signature(zkp.generate)
    assert sig_zkp.parameters["output"].default == "keys/zkp/zkp_params.json"

    # Legacy Concrete-ML config module removed; no concrete.generate() available.


check("default_paths", t_default_paths)


passed = sum(1 for v in results.values() if v == "PASS")
total = len(results)
print(f"\n{passed}/{total} passed\n")
for k, v in results.items():
    icon = "V" if v == "PASS" else "X"
    print(f"  [{icon}] {k:<30} {v}")

if passed < total:
    sys.exit(1)
