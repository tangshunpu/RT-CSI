"""Command-line interface for the RT-CSI dataset."""

from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "generate": "generate",
    "combine": "combine",
    "inspect": "inspect",
    "disjoint-fewshot": "fewshot",
    "plot-scenes": "plot_scenes",
    "plot-results": "plot_results",
}


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print("RT-CSI dataset tools\n\n"
              "Usage: rt-csi <command> [options]\n\n"
              "Commands:\n"
              "  generate          Ray-trace one scene or all Mix5 scenes\n"
              "  combine           Build the Mix5 NPZ from five scene files\n"
              "  inspect           Show NPZ arrays and metadata\n"
              "  disjoint-fewshot  Create UE-disjoint Shenzhen adaptation sets\n"
              "  plot-scenes       Plot CSI examples from all six scenes\n"
              "  plot-results      Plot Mix5 NMSE results from the manuscript table\n\n"
              "Run 'rt-csi <command> --help' for command options.")
        return
    command = argv[0]
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command {command!r}; run 'rt-csi --help'")
    module = importlib.import_module(f".{COMMANDS[command]}", __package__)
    module.main(argv[1:])
