#!/usr/bin/env python3
"""Generate the Dawn table initializer and smoke-test member lists for build.rs."""

import argparse
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
from make_dawn_api_package import DAWN, make_package


YAWGPU_INCLUDE = DAWN.parents[1] / "yawgpu/ffi/webgpu-headers"


def exports(header):
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', header.read_text(), flags=re.S)
    return {name: result.strip() for result, name in
            re.findall(r'^WGPU_EXPORT\s+([^;()]+?)\b(wgpu\w+)\s*\(', text, re.M)}


def members(header):
    layout = re.search(r'typedef struct DawnProcTable\s*\{\s*uint8_t version\[\d+\]\{\};'
                       r'(.*?)\} DawnProcTable;', header.read_text(), re.S)
    if not layout:
        raise ValueError("Unknown DawnProcTable layout")
    fields = layout[1]
    for field in fields.split(";"):
        if field.strip():
            match = re.fullmatch(r'\s*WGPUProc(\w+)\s+(\w+)\s*=\s*nullptr\s*', field)
            if not match:
                raise ValueError(f"Unknown DawnProcTable field: {field}")
            yield match[2], "wgpu" + match[1]


def watch_dawn_revision():
    # A pin bump can change kDawnVersion without changing any template.
    def git_path(name):
        return subprocess.check_output(
            ["git", "-C", DAWN, "rev-parse", "--path-format=absolute", "--git-path", name],
            text=True).strip()

    print("cargo:rerun-if-changed=" + git_path("HEAD"))
    branch = subprocess.run(["git", "-C", DAWN, "symbolic-ref", "-q", "HEAD"],
                            text=True, capture_output=True)
    if branch.returncode == 0:
        for ref in (branch.stdout.strip(), "packed-refs"):
            path = Path(git_path(ref))
            if path.exists():
                print("cargo:rerun-if-changed=" + str(path))


def generate(out):
    watch_dawn_revision()
    package = out / "dawn-api"
    make_package(package)
    dawn = package / "include/dawn"
    functions = exports(dawn / "webgpu.h")
    backed = exports(YAWGPU_INCLUDE / "webgpu.h") | exports(YAWGPU_INCLUDE / "yawgpu.h")
    fields = list(members(dawn / "dawn_proc_table.h"))
    names = {name for _, name in fields}
    if len(names) != len(fields) or names != functions.keys() or not backed.keys() <= names:
        raise ValueError("DawnProcTable and public WebGPU declarations disagree")

    init, checks, aliases = [], [], []
    for member, name in fields:
        checks.append(f"if (!table->{member}) return false;")
        aliases.append(f"#define {name} smoke_dispatch_{name}")
        if name in backed:
            init.append(f"table.{member} = &{name};")
            continue
        result = functions[name]
        fallback = {"WGPUStatus": "return WGPUStatus_Error;", "void": "return;"}.get(result, "return {};")
        init.extend([f"table.{member} = [](auto...) -> {result} {{",
                     "    static std::atomic_flag reported{};",
                     f'    reportUnsupported("{name}", reported);',
                     f"    {fallback}", "};"])
    for filename, lines in [("proc_init.inc", init), ("check_members.inc", checks),
                            ("smoke_dispatch.h", aliases)]:
        (out / filename).write_text("\n".join(lines) + "\n")
    print(f"Generated {len(backed)} yawgpu-backed and {len(fields) - len(backed)} stub members.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    generate(parser.parse_args().out)
