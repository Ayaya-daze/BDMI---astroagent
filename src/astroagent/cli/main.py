from __future__ import annotations

import argparse
from collections.abc import Callable

from astroagent.cli import apply_fit_control_patch, make_review_packet, run_fit_control_loop, run_fit_review


CommandMain = Callable[[list[str] | None], None]


COMMANDS: dict[str, tuple[str, CommandMain]] = {
    "packet": ("Build a review packet from a demo spectrum or CSV.", make_review_packet.main),
    "llm": ("Run a single fit-control or fit-review LLM pass.", run_fit_review.main),
    "apply-patch": ("Apply a fit-control patch and rerun the deterministic fit.", apply_fit_control_patch.main),
    "fit-loop": ("Run the bounded fit-control tool/refit loop.", run_fit_control_loop.main),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="astroagent",
        description="Build and review local quasar absorption-spectrum packets.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS), help="Command to run.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the selected command.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _, command_main = COMMANDS[args.command]
    command_args = list(args.args)
    if command_args and command_args[0] == "--":
        command_args = command_args[1:]
    command_main(command_args)


if __name__ == "__main__":
    main()
