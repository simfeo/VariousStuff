#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discard uncommitted changes in a git repository.

By default, all operations are performed. If at least one flag is specified,
only the selected operations are performed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SUBPROCESS_TEXT_ENCODING = "utf-8"


def configure_stdio() -> None:
	# Ensure UTF-8 output on Windows consoles.
	if hasattr(sys.stdout, "reconfigure"):
		sys.stdout.reconfigure(encoding="utf-8", errors="replace")
	if hasattr(sys.stderr, "reconfigure"):
		sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run_git(cwd: Path, args: list[str], *, dry_run: bool = False) -> None:
	command = ["git", *args]
	label = " ".join(command)
	if dry_run:
		print(f"[dry-run] {label}")
		return
	print(label)
	result = subprocess.run(command, cwd=cwd)
	if result.returncode != 0:
		raise subprocess.CalledProcessError(result.returncode, command)


def find_repo_root(start_path: Path) -> Path:
	result = subprocess.run(
		["git", "rev-parse", "--show-toplevel"],
		cwd=start_path,
		text=True,
		capture_output=True,
		encoding=SUBPROCESS_TEXT_ENCODING,
		errors="replace",
		check=False,
	)
	if result.returncode != 0:
		stderr = (result.stderr or "").strip()
		raise RuntimeError(f"Failed to find git repository at '{start_path}': {stderr}")
	return Path(result.stdout.strip())


def discard_untracked(repo_root: Path, *, dry_run: bool) -> None:
	clean_args = ["clean", "-fd"]
	if dry_run:
		clean_args.append("-n")
	run_git(repo_root, clean_args, dry_run=False)


def discard_tracked(repo_root: Path, *, dry_run: bool) -> None:
	if dry_run:
		run_git(repo_root, ["status", "--short"], dry_run=False)
		print("[dry-run] git reset --hard HEAD")
		return
	run_git(repo_root, ["reset", "--hard", "HEAD"], dry_run=False)


def discard_submodules(repo_root: Path, *, dry_run: bool) -> None:
	if dry_run:
		run_git(repo_root, ["submodule", "status", "--recursive"], dry_run=False)
		print("[dry-run] git submodule update --init --recursive --force")
		print("[dry-run] git submodule foreach --recursive git reset --hard HEAD")
		print("[dry-run] git submodule foreach --recursive git clean -fd")
		return
	run_git(repo_root, ["submodule", "update", "--init", "--recursive", "--force"], dry_run=False)
	run_git(
		repo_root,
		["submodule", "foreach", "--recursive", "git", "reset", "--hard", "HEAD"],
		dry_run=False,
	)
	clean_args = ["submodule", "foreach", "--recursive", "git", "clean", "-fd"]
	run_git(repo_root, clean_args, dry_run=False)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Discard uncommitted changes in a git repository.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog=(
			"Examples:\n"
			"  python DiscardUncommitted.py\n"
			"  python DiscardUncommitted.py --untracked\n"
			"  python DiscardUncommitted.py --tracked --submodules\n"
			"  python DiscardUncommitted.py --dry-run"
		),
	)
	parser.add_argument(
		"-u",
		"--untracked",
		action="store_true",
		help="Remove untracked files and directories in the root repository.",
	)
	parser.add_argument(
		"-t",
		"--tracked",
		action="store_true",
		help="Discard uncommitted changes in tracked files of the root repository.",
	)
	parser.add_argument(
		"-s",
		"--submodules",
		action="store_true",
		help="Discard changes in submodules (checkout, reset, clean inside submodules).",
	)
	parser.add_argument(
		"-C",
		"--directory",
		type=Path,
		default=Path("."),
		help="Directory used to resolve the git repository root (default: current directory).",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Show commands that would be executed without applying changes.",
	)
	return parser


def main() -> int:
	configure_stdio()
	parser = build_parser()
	args = parser.parse_args()
	selected = args.untracked or args.tracked or args.submodules
	# No flags means all operations; explicit flags select a subset.
	do_untracked = args.untracked or not selected
	do_tracked = args.tracked or not selected
	do_submodules = args.submodules or not selected
	try:
		repo_root = find_repo_root(args.directory.resolve())
	except RuntimeError as error:
		print(error, file=sys.stderr)
		return 1
	print(f"Repository: {repo_root}")
	if args.dry_run:
		print("Dry-run mode: no changes will be applied.")
	try:
		if do_tracked:
			print("=== Discard tracked file changes ===")
			discard_tracked(repo_root, dry_run=args.dry_run)
		if do_submodules:
			print("=== Discard submodule changes ===")
			discard_submodules(repo_root, dry_run=args.dry_run)
		if do_untracked:
			print("=== Remove untracked files ===")
			discard_untracked(repo_root, dry_run=args.dry_run)
	except subprocess.CalledProcessError as error:
		print(f"Command failed ({error.returncode}): {' '.join(error.cmd)}", file=sys.stderr)
		return error.returncode or 1
	print("Done.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
