from __future__ import annotations

import argparse
import os
import subprocess
import sys

from tool_registry import TOOLS, sorted_tools


def _print_tools() -> None:
    print("Debug tools:\n")
    for name, tool in sorted_tools():
        print(f"  {name:16} {tool['desc']}")
    print("\nUsage:")
    print("  python debug/cli.py <tool> [-- args]")
    print("  python debug/cli.py list")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("tool", nargs="?")
    parser.add_argument("tool_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.tool or args.tool in {"list", "-h", "--help"}:
        _print_tools()
        return

    tool = args.tool
    if tool not in TOOLS:
        print(f"Unknown tool: {tool}")
        _print_tools()
        sys.exit(2)

    script = TOOLS[tool]["script"]
    script_path = os.path.join(os.path.dirname(__file__), script)
    cmd = [sys.executable, script_path] + args.tool_args
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
