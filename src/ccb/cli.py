from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccb.audits import dataset_shortcut_audit
from ccb.dataset import (
    OfficialEvaluationFirewall,
    build_manifest,
    generate_firewalled_split,
    rehash_manifest,
    write_jsonl,
    write_manifest,
)
from ccb.experiment import load_experiment_config, run_experiment_matrix
from ccb.fit_gate import load_fit_gate_config, run_fit_gate
from ccb.presets import PRIMARY_SPLITS, domain_generator, primary_config
from ccb.encoding import collate_episodes
from ccb.results import write_result
from ccb.training import (
    TrainConfig,
    build_model,
    evaluate_batch,
    jsonl_logger,
    save_checkpoint,
    seed_everything,
    train_fixed_batch,
)
from ccb.structural_presets import build_d1_semantic_structural_splits, build_structural_splits
from ccb.official import OFFICIAL_DATA_HASHES, verify_data_hash, verify_official_records
from ccb.validation import validate_episode


def _generate(args: argparse.Namespace) -> int:
    output = Path(args.output)
    generator = domain_generator(args.domain)
    firewall = OfficialEvaluationFirewall.from_official_records(args.domain)
    splits = {}
    for config in PRIMARY_SPLITS:
        effective = config
        if args.seeds_per_depth is not None:
            effective = type(config)(
                config.name, config.depths, args.seeds_per_depth, config.base_seed
            )
        episodes = generate_firewalled_split(
            generator,
            effective,
            firewall,
        )
        for episode in episodes:
            validate_episode(episode)
        splits[effective.name] = episodes
        write_jsonl(output / f"{effective.name}.jsonl", episodes)

    config = primary_config(args.domain)
    if args.seeds_per_depth is not None:
        config["override_seeds_per_depth"] = args.seeds_per_depth
    manifest = build_manifest(splits, config=config, official_firewall=firewall)
    manifest["shortcut_audits"] = {
        name: dataset_shortcut_audit(episodes) for name, episodes in splits.items()
    }
    manifest = rehash_manifest(manifest)
    write_manifest(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "manifest_hash": manifest["manifest_hash"]}))
    return 0


def _smoke_train(args: argparse.Namespace) -> int:
    if min(args.steps, args.width, args.layers_or_loops, args.examples, args.depth) < 1:
        raise SystemExit("all numeric smoke-train arguments must be positive")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")
    seed_everything(args.seed)
    generator = domain_generator(args.domain)
    episodes = [generator(depth=args.depth, seed=args.seed + index) for index in range(args.examples)]
    batch = collate_episodes(episodes)
    config = TrainConfig(
        model=args.model,
        width=args.width,
        layers_or_loops=args.layers_or_loops,
        learning_rate=args.learning_rate,
        steps=args.steps,
        seed=args.seed,
        ema_decay=args.ema_decay,
    )
    model = build_model(config, batch.codec)
    model.to(args.device)
    batch = batch.to(args.device)
    output = Path(args.output)
    optimizer, history = train_fixed_batch(
        model, batch, config, log_callback=jsonl_logger(output / "train.jsonl")
    )
    metrics = evaluate_batch(model, batch)
    save_checkpoint(
        output / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        codec=batch.codec,
        step=config.steps,
    )
    result = {
        "domain": args.domain,
        "model": args.model,
        "config": config.__dict__,
        "examples": args.examples,
        "depth": args.depth,
        "initial": history[0],
        "final": {**history[-1], **metrics},
    }
    write_result(output / "result.json", result)
    print(json.dumps(result["final"], sort_keys=True))
    return 0


def _generate_structural(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if args.semantic_d1:
        if args.domain != "d1":
            raise SystemExit("--semantic-d1 is only valid with --domain d1")
        splits, structural_audit = build_d1_semantic_structural_splits(
            seeds_per_depth=args.seeds_per_depth
        )
    else:
        splits, structural_audit = build_structural_splits(
            args.domain, seeds_per_depth=args.seeds_per_depth
        )
    for name, episodes in splits.items():
        for episode in episodes:
            validate_episode(episode)
        write_jsonl(output / f"{name}.jsonl", episodes)
    manifest = build_manifest(
        splits,
        config={
            "benchmark": (
                "ccb_d1_semantic_structural_v1"
                if args.semantic_d1
                else "ccb_learn_structural_v1"
            ),
            "domain": args.domain,
            "seeds_per_depth": args.seeds_per_depth,
        },
        official_firewall=(
            OfficialEvaluationFirewall.from_official_records("d1")
            if args.semantic_d1
            else None
        ),
    )
    manifest["structural_audit"] = structural_audit
    manifest["shortcut_audits"] = {
        name: dataset_shortcut_audit(episodes) for name, episodes in splits.items()
    }
    manifest = rehash_manifest(manifest)
    write_manifest(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "manifest_hash": manifest["manifest_hash"]}))
    return 0


def _verify_official(_: argparse.Namespace) -> int:
    reports = []
    for domain in OFFICIAL_DATA_HASHES:
        report = verify_official_records(domain)
        reports.append(
            {
                "domain": domain,
                "data_hash_valid": verify_data_hash(domain),
                "records": report.records,
                "exact_matches": report.exact_matches,
                "compatible": report.compatible,
                "mismatches": list(report.mismatches),
            }
        )
    print(json.dumps(reports, indent=2))
    return 0 if all(item["data_hash_valid"] and item["compatible"] for item in reports) else 1


def _launch(args: argparse.Namespace) -> int:
    config = load_experiment_config(Path(args.config))
    results = run_experiment_matrix(config, dry_run=args.dry_run)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _fit_gate(args: argparse.Namespace) -> int:
    result = run_fit_gate(load_fit_gate_config(Path(args.config)))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ccb", description="CCB-Learn utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate deterministic splits")
    generate.add_argument("--domain", choices=("d1", "d2", "d3"), required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--seeds-per-depth", type=int)
    generate.set_defaults(handler=_generate)
    structural = subparsers.add_parser(
        "generate-structural", help="generate an explicit structural-holdout suite"
    )
    structural.add_argument("--domain", choices=("d1", "d2", "d3"), required=True)
    structural.add_argument("--output", required=True)
    structural.add_argument("--seeds-per-depth", type=int, default=50)
    structural.add_argument(
        "--semantic-d1",
        action="store_true",
        help="use randomized inputs and semantic-transformation exclusion for D1",
    )
    structural.set_defaults(handler=_generate_structural)
    verify = subparsers.add_parser(
        "verify-official", help="verify hashes and exact D1-D3 official-record compatibility"
    )
    verify.set_defaults(handler=_verify_official)
    smoke = subparsers.add_parser("smoke-train", help="run a tiny fixed-batch CPU overfit test")
    smoke.add_argument("--domain", choices=("d1", "d2", "d3"), required=True)
    smoke.add_argument(
        "--model",
        choices=("transformer", "ccb_token_transformer", "trm_upstream_core", "official_trm_ccb"),
        required=True,
    )
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--steps", type=int, default=200)
    smoke.add_argument("--width", type=int, default=32)
    smoke.add_argument("--layers-or-loops", type=int, default=2)
    smoke.add_argument("--learning-rate", type=float, default=1e-3)
    smoke.add_argument("--examples", type=int, default=8)
    smoke.add_argument("--depth", type=int, default=4)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--ema-decay", type=float, default=0.999)
    smoke.set_defaults(handler=_smoke_train)
    launch = subparsers.add_parser(
        "launch", help="run a reproducible model-by-seed experiment matrix"
    )
    launch.add_argument("--config", required=True, help="path to a JSON experiment config")
    launch.add_argument("--dry-run", action="store_true")
    launch.set_defaults(handler=_launch)
    fit_gate = subparsers.add_parser(
        "fit-gate", help="run a fixed-data TRM or Transformer training-path diagnostic"
    )
    fit_gate.add_argument("--config", required=True, help="path to a fit-gate JSON config")
    fit_gate.set_defaults(handler=_fit_gate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "seeds_per_depth", None) is not None and args.seeds_per_depth < 1:
        raise SystemExit("--seeds-per-depth must be positive")
    return args.handler(args)
