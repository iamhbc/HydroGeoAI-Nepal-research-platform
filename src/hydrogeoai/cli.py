"""Command-line interface: `hydrogeoai <command>` (see `hydrogeoai -h`)."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hydrogeoai", description="HydroGeoAI-Nepal research platform")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="dataset pipeline")
    d_sub = d.add_subparsers(dest="action", required=True)
    b = d_sub.add_parser("build", help="raw -> QC -> homogeneity -> events -> features -> versioned dataset")
    b.add_argument("--config", default="configs/data/default.yaml")
    b.add_argument("--events", default="configs/events/taxonomy.yaml")
    b.add_argument("--force", action="store_true")
    d_sub.add_parser("list", help="list datasets in the catalog")
    a = d_sub.add_parser("approve", help="approve a checked dataset")
    a.add_argument("key")

    e = sub.add_parser("experiment", help="run experiments")
    e_sub = e.add_subparsers(dest="action", required=True)
    r = e_sub.add_parser("run")
    r.add_argument("--config", default="configs/experiments/main.yaml")
    r.add_argument("--modes", nargs="*", choices=["temporal", "spatial", "spatiotemporal"])
    e_sub.add_parser("list")
    s = e_sub.add_parser("show")
    s.add_argument("experiment_id")

    m = sub.add_parser("models", help="model registry")
    m_sub = m.add_subparsers(dest="action", required=True)
    m_sub.add_parser("list")
    dep = m_sub.add_parser("deploy")
    dep.add_argument("model_id")
    rb = m_sub.add_parser("rollback")
    rb.add_argument("--name", default="hydrogeoai-nepal-model")

    api = sub.add_parser("api", help="run the FastAPI server")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    api.add_argument("--reload", action="store_true")

    hf = sub.add_parser("hf", help="Hugging Face export / publish (manual only)")
    hf_sub = hf.add_subparsers(dest="action", required=True)
    ex = hf_sub.add_parser("export", help="stage dataset/model/space folders under hf/export (no upload)")
    ex.add_argument("--namespace", default=None)
    hf_sub.add_parser("whoami", help="show the logged-in Hugging Face account")
    pub = hf_sub.add_parser("publish", help="upload staged folders (dry run unless --push; needs `hf auth login`)")
    pub.add_argument("--namespace", default=None, help="defaults to the logged-in HF username")
    pub.add_argument("--push", action="store_true", help="actually upload; otherwise dry run")
    pub.add_argument("--private", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "data":
        if args.action == "build":
            from .data.pipeline import build_dataset
            print(build_dataset(args.config, args.events, force=args.force))
        elif args.action == "list":
            from .data.catalog import DatasetCatalog
            print(json.dumps(DatasetCatalog().list(), indent=2))
        elif args.action == "approve":
            from .data.catalog import DatasetCatalog
            print(json.dumps(DatasetCatalog().set_status(args.key, "approved"), indent=2, default=str))
    elif args.cmd == "experiment":
        from .experiments.registry import Registry
        if args.action == "run":
            from .experiments.runner import run
            print(json.dumps(run(args.config, args.modes), indent=2))
        elif args.action == "list":
            for x in Registry().list_experiments():
                print(f"{x['id']:45s} {x['status']:10s} {x['created']}  dataset={x['dataset_version']}")
        elif args.action == "show":
            print(json.dumps(Registry().get_experiment(args.experiment_id), indent=2, default=str))
    elif args.cmd == "models":
        from .experiments.registry import Registry
        reg = Registry()
        if args.action == "list":
            for x in reg.list_models():
                print(f"{x['id']:45s} {x['status']:10s} dataset={x['dataset_version']}")
        elif args.action == "deploy":
            print(reg.set_model_status(args.model_id, "deployed"))
        elif args.action == "rollback":
            print(reg.rollback_model(args.name))
    elif args.cmd == "api":
        import uvicorn
        uvicorn.run("hydrogeoai.api.main:app", host=args.host, port=args.port, reload=args.reload)
    elif args.cmd == "hf":
        from .hf import publish
        if args.action == "export":
            print(json.dumps(publish.export(args.namespace), indent=2))
        elif args.action == "whoami":
            print(json.dumps(publish.whoami() or {"logged_in": False, "hint": "run `hf auth login`"}, indent=2, default=str))
        else:
            print(json.dumps(publish.publish(args.namespace, push=args.push, private=args.private), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
