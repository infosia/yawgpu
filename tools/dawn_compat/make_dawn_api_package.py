#!/usr/bin/env python3
"""Assemble the pinned external-Dawn API headers and standalone dispatch sources."""

import argparse
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
from gen_dawn_headers import DAWN, gen_dawn_headers


def standalone_proc(source):
    # Require the version check even if a later template stops emitting it.
    check = "std::ranges::equal(procs_->version, dawn::kDawnVersion)"
    replacements = {
        '#include "src/dawn/common/Compiler.h"': "",
        '#include "src/utils/assert.h"': "#include <cstdlib>",
        '#include "src/utils/log.h"': "#include <iostream>",
        "CreateInstanceThatWarns(const WGPUInstanceDescriptor* desc)":
            "CreateInstanceThatWarns(const WGPUInstanceDescriptor*)",
        f"DAWN_CHECK({check});": f"if (!{check}) std::abort();",
    }
    for original, replacement in replacements.items():
        if source.count(original) != 1:
            raise ValueError(f"Expected exactly one {original!r} in dawn_proc.cpp")
        source = source.replace(original, replacement)
    source, warnings = re.subn(r'dawn::ErrorLog\(\) <<(\s*R"\(.*?\)")\s*;',
                               r'std::cerr <<\1 << std::endl;', source, flags=re.S)
    if warnings != 1:
        raise ValueError("Expected exactly one ErrorLog warning in dawn_proc.cpp")
    source = source.replace('// The sanitizer is disabled for calls to procs.* since those functions may be\n'
                            '// dynamically loaded.\n', '')
    source = source.replace('DAWN_NO_SANITIZE("cfi-icall")\n', '')

    check_dependencies("dawn_proc.cpp", source, {
        '#include "dawn/dawn_proc.h"', '#include "dawn/dawn_version.h"',
        "#include <algorithm>", "#include <iostream>", "#include <cstdlib>"})
    return source


def check_dependencies(filename, source, includes):
    for line in source.splitlines():
        if line.lstrip().startswith("#") and line.strip() not in includes:
            raise ValueError(f"Unknown preprocessor directive in {filename}: {line}")
    # Ignore comments and string literals (including Dawn's multiline warning).
    code = re.sub(r'//[^\n]*|/\*.*?\*/|R"\(.*?\)"|"(?:\\.|[^"\\])*"', '', source, flags=re.S)
    unknown = re.search(r'\b[A-Z][A-Z0-9_]+\b|\bdawn::(?!kDawnVersion\b)\w+', code)
    if unknown:
        raise ValueError(f"Unknown private token in {filename}: {unknown[0]}")


def make_package(out):
    with tempfile.TemporaryDirectory() as tmp:
        generated = Path(tmp)
        gen_dawn_headers(generated)
        src = generated / "src/dawn"
        proc = standalone_proc((src / "dawn_proc.cpp").read_text())
        thread = (src / "dawn_thread_dispatch_proc.cpp").read_text()
        check_dependencies("dawn_thread_dispatch_proc.cpp", thread, {
            '#include "dawn/dawn_thread_dispatch_proc.h"', '#include "dawn/dawn_version.h"',
            "#include <algorithm>", "#include <thread>"})
        for namespace in ("dawn", "webgpu"):
            dest = out / "include" / namespace
            dest.mkdir(parents=True, exist_ok=True)
            for root in (DAWN, generated):
                for header in (root / "include" / namespace).glob("*.h"):
                    if header.name != "webgpu_glfw.h":
                        shutil.copyfile(header, dest / header.name)
        (out / "src").mkdir(parents=True, exist_ok=True)
        (out / "src/dawn_proc.cpp").write_text(proc)
        (out / "src/dawn_thread_dispatch_proc.cpp").write_text(thread)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    make_package(parser.parse_args().out)
