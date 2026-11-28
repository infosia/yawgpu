#!/usr/bin/env python3
"""Generate the C/C++ headers and proc sources from the pinned Dawn checkout."""

import argparse
from pathlib import Path
import subprocess
import sys


DAWN = Path(__file__).resolve().parents[2] / "third_party/dawn"


def gen_dawn_headers(out: Path) -> None:
    """Run Dawn's API and version generators into out."""
    generator = DAWN / "generator"
    definitions = DAWN / "src/dawn"
    if not (generator / "dawn_json_generator.py").is_file():
        raise FileNotFoundError(f"Missing Dawn submodule at {DAWN}")
    common = ["--root-dir", DAWN, "--template-dir", generator / "templates",
              "--output-dir", Path(out).resolve()]
    for package in ("jinja2", "markupsafe"):
        path = DAWN / "third_party" / package
        if not (path / "__init__.py").is_file():
            raise FileNotFoundError(f"Missing Dawn's vendored {package} at {path}")
        common.extend([f"--{package}-path", path])

    subprocess.run([
        sys.executable, "-B", generator / "dawn_json_generator.py",
        "--dawn-json", definitions / "dawn.json",
        "--wire-json", definitions / "dawn_wire.json",
        "--targets", "headers,cpp_headers,proc", *common,
    ], check=True)
    subprocess.run([
        sys.executable, "-B", generator / "dawn_version_generator.py", "gen",
        "--dawn-dir", DAWN, "--version-json", definitions / "dawn_version.json",
        *common,
    ], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    try:
        gen_dawn_headers(parser.parse_args().out)
    except (OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
