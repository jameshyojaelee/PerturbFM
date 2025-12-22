"""Command-line interface for PerturbFM."""

from __future__ import annotations

import argparse
import sys


def _add_doctor(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("doctor", help="Basic environment checks.")
    parser.set_defaults(func=_cmd_doctor)


def _add_config(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("config", help="Config utilities (stub).")
    parser.set_defaults(func=_cmd_config)


def _add_data(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("data", help="Dataset utilities.")
    parser.set_defaults(func=_cmd_data)


def _add_splits(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("splits", help="Split utilities.")
    parser.set_defaults(func=_cmd_splits)


def _add_train(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="Training utilities.")
    parser.set_defaults(func=_cmd_train)


def _add_eval(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("eval", help="Evaluation utilities.")
    parser.set_defaults(func=_cmd_eval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="perturbfm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_doctor(subparsers)
    _add_config(subparsers)
    _add_data(subparsers)
    _add_splits(subparsers)
    _add_train(subparsers)
    _add_eval(subparsers)
    return parser


def _cmd_doctor(_args: argparse.Namespace) -> int:
    print("perturbfm: ok")
    return 0


def _cmd_config(_args: argparse.Namespace) -> int:
    print("config: not implemented yet")
    return 0


def _cmd_data(_args: argparse.Namespace) -> int:
    print("data: not implemented yet")
    return 0


def _cmd_splits(_args: argparse.Namespace) -> int:
    print("splits: not implemented yet")
    return 0


def _cmd_train(_args: argparse.Namespace) -> int:
    print("train: not implemented yet")
    return 0


def _cmd_eval(_args: argparse.Namespace) -> int:
    print("eval: not implemented yet")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
