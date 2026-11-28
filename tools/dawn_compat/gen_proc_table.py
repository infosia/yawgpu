#!/usr/bin/env python3
"""Generate the Dawn table initializer and smoke-test member lists for build.rs."""

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
from abi_guard import DEFAULT_INCLUDE, check_abi
from make_dawn_api_package import DAWN, make_package


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


def stub_returns(header):
    """Map recognized Dawn C types to their unsupported-call return statements."""
    source = header.read_text()
    returns = {"void": "return;", "WGPUStatus": "return WGPUStatus_Error;",
               "WGPUBool": "return 0;"}
    for name in ("int", "unsigned", "size_t", "int8_t", "uint8_t", "int16_t", "uint16_t",
                 "int32_t", "uint32_t", "int64_t", "uint64_t"):
        returns[name] = "return 0;"
    for base, alias in re.findall(r"^typedef (\w+) (\w+);", source, re.M):
        if returns.get(base) == "return 0;":
            returns[alias] = "return 0;"
    for handle in re.findall(r"^typedef struct \w+\s*\*\s*(\w+) WGPU_OBJECT_ATTRIBUTE;",
                             source, re.M):
        returns[handle] = "return nullptr;"
    for proc in re.findall(r"^typedef [^;()]+\(\*([A-Za-z_]\w*)\)\(", source, re.M):
        returns[proc] = "return nullptr;"
    for struct in re.findall(r"^} (\w+) WGPU_STRUCTURE_ATTRIBUTE;", source, re.M):
        returns[struct] = "return {};"
    return returns


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


def generate(out, yawgpu_include):
    watch_dawn_revision()
    package = out / "dawn-api"
    make_package(package)
    dawn = package / "include/dawn"
    with tempfile.TemporaryDirectory(dir=out) as work:
        problems = check_abi(yawgpu_include, dawn.parent, Path(work))
    if problems:
        raise ValueError("Incompatible yawgpu ABI:\n" + "\n".join(problems))
    header = dawn / "webgpu.h"
    functions = exports(header)
    returns = stub_returns(header)
    backed = exports(yawgpu_include / "webgpu.h") | exports(yawgpu_include / "yawgpu.h")
    fields = list(members(dawn / "dawn_proc_table.h"))
    names = {name for _, name in fields}
    if len(names) != len(fields) or names != functions.keys() or not backed.keys() <= names:
        raise ValueError("DawnProcTable and public WebGPU declarations disagree")

    init, checks, aliases = [], [], []
    for member, name in fields:
        checks.append(f"if (!table->{member}) return false;")
        aliases.append(f"#define {name} smoke_dispatch_{name}")
        if name in backed:
            target = {"wgpuCreateInstance": "yawgpuDawnCreateInstance",
                      "wgpuGetProcAddress": "getProcAddress"}.get(name, name)
            init.append(f"table.{member} = &{target};")
            continue
        result = functions[name].removeprefix("WGPU_NULLABLE ")
        fallback = "return nullptr;" if result.endswith("*") else returns.get(result)
        if fallback is None:
            raise ValueError(f"Unclassified stub return type {result} for {name}")
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
    parser.add_argument("--yawgpu-include", type=Path, default=DEFAULT_INCLUDE)
    args = parser.parse_args()
    try:
        generate(args.out.resolve(), args.yawgpu_include.resolve())
    except ValueError as error:
        sys.exit(str(error))
