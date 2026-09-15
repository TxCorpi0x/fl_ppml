"""
fl.keys.cli — Unified CLI for cryptographic key/parameter management.

Usage::

    python -m fl.keys generate he_tenseal [--secret SECRET] [--public PUBLIC] [--overwrite]
    python -m fl.keys generate dp         [--output OUTPUT] [--epsilon E] [--delta D] ...
    python -m fl.keys generate zkp        [--output OUTPUT] [--bit_length N] [--overwrite]
    python -m fl.keys generate he_concrete_tfhe [--bit_width N] [--num_clients N] ...
    python -m fl.keys generate he_elgamal [--overwrite]
    python -m fl.keys list
    python -m fl.keys prebuilt            # list Concrete TFHE prebuilt bundles
"""

from __future__ import annotations

import argparse
import sys


def _parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="python -m fl.keys",
        description="Federated-learning cryptographic key manager",
    )
    sub = root.add_subparsers(dest="command", required=True)

    # ── list ──────────────────────────────────────────────────────────────────
    sub.add_parser("list", help="List supported modes")

    # ── prebuilt ──────────────────────────────────────────────────────────────
    sub.add_parser("prebuilt", help="List Concrete TFHE prebuilt key bundles")

    # ── generate ──────────────────────────────────────────────────────────────
    gen = sub.add_parser("generate", help="Generate key material for a mode")
    gen.add_argument(
        "mode", choices=["he_tenseal", "dp", "zkp", "he_concrete_tfhe", "he_elgamal"]
    )

    # he_tenseal
    gen.add_argument(
        "--secret",
        default="keys/he_tenseal/secret_context.bin",
        help="[he_tenseal] Client secret key path (default: keys/he_tenseal/secret_context.bin)",
    )
    gen.add_argument(
        "--public",
        default="keys/he_tenseal/public_context.bin",
        help="[he_tenseal] Server public key path (default: keys/he_tenseal/public_context.bin)",
    )

    # dp / zkp / concrete — shared output flag
    gen.add_argument(
        "--output",
        default=None,
        help="Output file path (mode-specific default used if omitted)",
    )

    # dp
    gen.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="[dp] Privacy budget ε (default: 1.0)",
    )
    gen.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="[dp] Failure probability δ (default: 1e-5)",
    )
    gen.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="[dp] Gradient clipping bound (default: 1.0)",
    )
    gen.add_argument(
        "--mechanism",
        default="gaussian",
        choices=["gaussian", "laplace"],
        help="[dp] Noise mechanism (default: gaussian)",
    )
    gen.add_argument(
        "--noise_multiplier",
        type=float,
        default=None,
        help="[dp] Override auto-computed noise multiplier",
    )

    # zkp
    gen.add_argument(
        "--bit_length",
        type=int,
        default=2048,
        help="[zkp] Security parameter in bits (default: 2048)",
    )

    # (he_concrete mode removed; use he_concrete_tfhe for Concrete TFHE flows)

    # he_concrete_tfhe
    gen.add_argument(
        "--bit_width",
        type=int,
        default=14,
        help="[he_concrete_tfhe] Fixed-point bit width (default: 14)",
    )
    gen.add_argument(
        "--num_clients",
        type=int,
        default=2,
        help="[he_concrete_tfhe] Number of clients (default: 2)",
    )
    gen.add_argument(
        "--no_fhe",
        action="store_true",
        help="[he_concrete_tfhe] Simulate FHE cost without real crypto",
    )

    # shared
    gen.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing key files"
    )

    return root


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)

    if args.command == "list":
        from fl.keys import list_modes

        print("Supported modes:")
        for m in list_modes():
            print(f"  {m}")
        return

    if args.command == "prebuilt":
        from fl.keys.concrete_tfhe import list_prebuilt, prebuilt_dir

        bundles = list_prebuilt()
        if not bundles:
            print(f"No prebuilt bundles found in {prebuilt_dir}")
        else:
            print(f"Prebuilt Concrete TFHE bundles in {prebuilt_dir}:")
            for b in bundles:
                print(f"  {b}")
        return

    # generate
    mode = args.mode
    overwrite = args.overwrite

    try:
        if mode == "he_tenseal":
            from fl.keys.he_tenseal import generate

            generate(
                secret_path=args.secret, public_path=args.public, overwrite=overwrite
            )

        elif mode == "dp":
            from fl.keys.dp import generate

            generate(
                output=args.output or "dp_params.json",
                epsilon=args.epsilon,
                delta=args.delta,
                max_grad_norm=args.max_grad_norm,
                mechanism=args.mechanism,
                noise_multiplier=args.noise_multiplier,
                overwrite=overwrite,
            )

        elif mode == "zkp":
            from fl.keys.zkp import generate

            generate(
                output=args.output or "zkp_params.json",
                bit_length=args.bit_length,
                overwrite=overwrite,
            )

        elif mode == "he_concrete_tfhe":
            from fl.keys.concrete_tfhe import generate

            generate(
                bit_width=args.bit_width,
                num_clients=args.num_clients,
                enable_fhe=not args.no_fhe,
            )

        elif mode == "he_elgamal":
            from fl.keys.he_elgamal import generate

            generate(overwrite=overwrite)

    except (FileExistsError, FileNotFoundError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}", file=sys.stderr)
        raise
