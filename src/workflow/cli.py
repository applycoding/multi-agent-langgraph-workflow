import argparse
import sys
from dataclasses import replace
from pathlib import Path

import uvicorn
from pydantic import ValidationError

from workflow.api import create_app
from workflow.checkpoint import CheckpointStore
from workflow.config import Provider, Settings, parse_provider, settings_from_env
from workflow.errors import RunNotFound, RunNotWaiting, WorkflowError
from workflow.models import RunView
from workflow.service import WorkflowService, build_service


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except WorkflowError as exc:
        print(str(exc), file=sys.stderr)
        return _exit_code(exc)
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--provider",
        choices=[provider.value for provider in Provider],
        default=None,
        help="Override LLM_PROVIDER. Default: fake.",
    )
    common.add_argument(
        "--checkpoint",
        default=None,
        help="SQLite checkpoint path, or 'memory'. Default: data/checkpoints.sqlite.",
    )
    common.add_argument("--docs-dir", default=None, help="Markdown directory for local search.")

    parser = argparse.ArgumentParser(
        prog="workflow",
        description="Run the multi-agent portfolio demo from the command line.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", parents=[common], help="Start a run and pause for approval.")
    run.add_argument("topic")
    run.add_argument("--audience", default="technical reader")
    run.add_argument("--thread-id", default=None)
    run.add_argument("--max-revisions", type=int, default=None)
    run.add_argument(
        "--yes",
        action="store_true",
        help="Approve the draft in this same command after the graph pauses.",
    )
    run.add_argument("--notes", default="")
    run.set_defaults(handler=run_command)

    status = commands.add_parser("status", parents=[common], help="Show a checkpointed run.")
    status.add_argument("thread_id")
    status.set_defaults(handler=status_command)

    approve = commands.add_parser(
        "approve",
        parents=[common],
        help="Approve or reject the paused draft.",
    )
    approve.add_argument("thread_id")
    approve.add_argument("--reject", action="store_true")
    approve.add_argument("--notes", default="")
    approve.set_defaults(handler=approve_command)

    resume = commands.add_parser(
        "resume",
        parents=[common],
        help="Resume a paused run with an approval payload.",
    )
    resume.add_argument("thread_id")
    resume.add_argument("--approved", choices=["true", "false"], required=True)
    resume.add_argument("--notes", default="")
    resume.set_defaults(handler=resume_command)

    serve = commands.add_parser("serve", parents=[common], help="Start the FastAPI demo server.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=serve_command)
    return parser


def run_command(args: argparse.Namespace) -> int:
    settings = settings_for_args(args)
    max_revisions = settings.max_revisions if args.max_revisions is None else args.max_revisions

    def action(service: WorkflowService) -> RunView:
        view = service.start(
            args.topic,
            audience=args.audience,
            thread_id=args.thread_id,
            max_revisions=max_revisions,
        )
        if args.yes:
            if view.status != "awaiting_approval":
                return view
            return service.approve(view.thread_id, True, args.notes)
        return view

    view = _using_service(settings, action)
    _print(view)
    return 1 if view.status == "failed" else 0


def status_command(args: argparse.Namespace) -> int:
    settings = settings_for_args(args)
    view = _using_service(settings, lambda service: service.get(args.thread_id))
    if view.status == "not_found":
        raise RunNotFound(f"Run {args.thread_id} was not found.")
    _print(view)
    return 0


def approve_command(args: argparse.Namespace) -> int:
    settings = settings_for_args(args)
    view = _using_service(
        settings,
        lambda service: service.approve(args.thread_id, not args.reject, args.notes),
    )
    _print(view)
    return 1 if view.status == "failed" else 0


def resume_command(args: argparse.Namespace) -> int:
    settings = settings_for_args(args)
    approved = args.approved == "true"
    view = _using_service(
        settings,
        lambda service: service.resume(
            args.thread_id,
            {"approved": approved, "notes": args.notes},
        ),
    )
    _print(view)
    return 1 if view.status == "failed" else 0


def serve_command(args: argparse.Namespace) -> int:
    settings = settings_for_args(args)
    store = CheckpointStore(settings.checkpoint_path)
    try:
        service = build_service(settings, store)
        uvicorn.run(create_app(service), host=args.host, port=args.port)
    finally:
        store.close()
    return 0


def settings_for_args(args: argparse.Namespace) -> Settings:
    settings = settings_from_env()
    checkpoint = settings.checkpoint_path
    if args.checkpoint is not None:
        checkpoint = None if args.checkpoint == "memory" else args.checkpoint
    return replace(
        settings,
        provider=settings.provider if args.provider is None else parse_provider(args.provider),
        checkpoint_path=checkpoint,
        docs_dir=settings.docs_dir if args.docs_dir is None else Path(args.docs_dir),
    )


def _using_service(settings, action):
    store = CheckpointStore(settings.checkpoint_path)
    try:
        return action(build_service(settings, store))
    finally:
        store.close()


def _print(view: RunView) -> None:
    print(view.model_dump_json(indent=2))


def _exit_code(exc: WorkflowError) -> int:
    if isinstance(exc, RunNotFound):
        return 2
    if isinstance(exc, RunNotWaiting):
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
