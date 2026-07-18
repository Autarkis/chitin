from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="chitin collider build service")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="start the API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8400)
    serve.add_argument("--data-dir", default=None)

    submit = sub.add_parser("submit", help="submit a build job")
    submit.add_argument("file", help="path to input asset")
    submit.add_argument("--server", default="http://127.0.0.1:8400")
    submit.add_argument(
        "--outputs", default="phys,json", help="comma-separated: phys,json,usd"
    )
    submit.add_argument("--concavity", type=float, default=0.05)
    submit.add_argument("--opacity-threshold", type=float, default=0.5)
    submit.add_argument(
        "--poisson-depth",
        type=int,
        default=None,
        help="Poisson depth for point clouds; default auto-selects 4-7 per "
        "cell. 8+ is accepted but unstable (isolated in a subprocess).",
    )
    submit.add_argument("--min-hull-vertices", type=int, default=4)
    submit.add_argument("--max-hulls", type=int, default=256)
    submit.add_argument("--opacity-is-logit", action="store_true")
    submit.add_argument(
        "--coacd-preprocess-mode", default="auto", choices=["auto", "on", "off"]
    )
    submit.add_argument("--coacd-preprocess-resolution", type=int, default=50)
    submit.add_argument("--max-decompose-vertices", type=int, default=200_000)

    status = sub.add_parser("status", help="check job status")
    status.add_argument("job_id")
    status.add_argument("--server", default="http://127.0.0.1:8400")

    download = sub.add_parser("download", help="download job artifacts")
    download.add_argument("job_id")
    download.add_argument("--server", default="http://127.0.0.1:8400")
    download.add_argument("-o", "--output-dir", default=".")

    args = parser.parse_args()

    if args.command == "serve":
        _cmd_serve(args)
    elif args.command == "submit":
        _cmd_submit(args)
    elif args.command == "status":
        _cmd_status(args)
    elif args.command == "download":
        _cmd_download(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_serve(args: argparse.Namespace) -> None:
    if args.data_dir:
        from pathlib import Path

        from chitin_service import app as app_mod
        from chitin_service.store import Store

        app_mod.set_store(Store(Path(args.data_dir)))

    import uvicorn

    uvicorn.run("chitin_service.app:app", host=args.host, port=args.port)


def _cmd_submit(args: argparse.Namespace) -> None:
    import httpx

    path = args.file
    params = {
        "outputs": args.outputs,
        "concavity": args.concavity,
        "opacity_threshold": args.opacity_threshold,
        "min_hull_vertices": args.min_hull_vertices,
        "max_hulls": args.max_hulls,
        "opacity_is_logit": args.opacity_is_logit,
        "coacd_preprocess_mode": args.coacd_preprocess_mode,
        "coacd_preprocess_resolution": args.coacd_preprocess_resolution,
        "max_decompose_vertices": args.max_decompose_vertices,
    }
    # Only override the server's auto depth (4-7 per cell) when asked; the CLI no
    # longer forces the documented-unstable depth 8 on every job.
    if args.poisson_depth is not None:
        params["poisson_depth"] = args.poisson_depth

    with open(path, "rb") as f:
        resp = httpx.post(
            f"{args.server}/v1/jobs",
            files={"file": (path.split("/")[-1], f)},
            params=params,
            timeout=600.0,
        )

    if resp.status_code != 201:
        print(f"error: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    job_id = data["id"]
    status = data["status"]
    cached = data.get("cached_from")

    print(f"job {job_id}: {status}")
    if cached:
        print(f"  cached from: {cached}")

    if data.get("events"):
        last = data["events"][-1]
        if last.get("message"):
            print(f"  {last['message']}")

    if status == "complete":
        print(f"\n  chitin-server download {job_id}")


def _cmd_status(args: argparse.Namespace) -> None:
    import httpx

    resp = httpx.get(f"{args.server}/v1/jobs/{args.job_id}", timeout=10.0)
    if resp.status_code == 404:
        print(f"job {args.job_id} not found", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    print(f"job {data['id']}: {data['status']}")
    print(f"  input: {data['input_uri']}")
    print(f"  outputs: {', '.join(data['outputs'])}")
    if data.get("error"):
        print(f"  error: {data['error']}")
    if data.get("events"):
        print("  events:")
        for ev in data["events"]:
            msg = f"    {ev['status']}"
            if ev.get("message"):
                msg += f" — {ev['message']}"
            print(msg)


def _cmd_download(args: argparse.Namespace) -> None:
    import httpx
    from pathlib import Path

    base = f"{args.server}/v1/jobs/{args.job_id}"
    resp = httpx.get(f"{base}/artifacts", timeout=10.0)
    if resp.status_code != 200:
        print(f"error: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)

    artifacts = resp.json()["artifacts"]
    if not artifacts:
        print("no artifacts available")
        sys.exit(0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in artifacts:
        r = httpx.get(f"{base}/artifacts/{name}", timeout=60.0)
        dest = out_dir / name
        dest.write_bytes(r.content)
        print(f"  {dest}")


if __name__ == "__main__":
    main()
