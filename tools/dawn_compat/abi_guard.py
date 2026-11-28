#!/usr/bin/env python3
"""Check yawgpu's public WebGPU C ABI against the pinned Dawn headers."""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

# Import the generator without leaving bytecode in the source tree.
sys.dont_write_bytecode = True
from gen_dawn_headers import gen_dawn_headers


DEFAULT_INCLUDE = Path(__file__).resolve().parents[2] / "yawgpu/ffi/webgpu-headers"
VENDOR_PREFIXES = ("YaWGPU", "YAWGPU_", "yawgpu")


def public_source(preprocessed, paths):
    """Use the preprocessor's line markers to omit declarations from system headers."""
    lines = []
    public = False
    for line in preprocessed.splitlines():
        marker = re.match(r'^# \d+ "([^"]+)"', line)
        if marker:
            public = marker[1] in paths
        elif public:
            lines.append(line)
    if not lines:
        raise ValueError("C preprocessor returned no public header declarations")
    return "\n".join(lines)


def struct_definitions(declarations):
    structs = {}
    for start in re.finditer(r"\b(?:(typedef)\s+)?struct\s*(\w+)?\s*\{", declarations):
        typedef, tag = start.groups()
        match = re.match(r"([^{}]*)\}\s*(\w*)\s*;", declarations[start.end():])
        if not match:
            raise ValueError(f"Cannot read struct definition near {start[0]}")
        body, alias = match.groups()
        name = alias if typedef else tag
        if not name or name.startswith(VENDOR_PREFIXES):
            continue
        members = []
        for field in body.split(";"):
            if not field.strip():
                continue
            for declarator in field.split(","):
                member = re.search(r"\b(\w+)\s*(?:\[[^\]]*\]\s*)*$", declarator)
                if not member:
                    raise ValueError(f"Cannot identify member of {name}: {field}")
                members.append(member[1])
        structs[name] = (alias if typedef else f"struct {tag}", members)
    return structs


class Header:
    def __init__(self, cc, include, headers):
        self.cc = [*cc, "-std=c11", "-I", include, "-x", "c", "-"]
        self.includes = "".join(f"#include <{name}>\n" for name in headers)
        preprocessed = public_source(self.compile("", "-E", "-dD").stdout,
                                     {str(include / name) for name in headers})
        declarations = re.sub(r"^#.*$", "", preprocessed, flags=re.M)

        self.functions = {}
        for declaration in re.split(r"[;{}]", declarations):
            match = re.search(r"\b(wgpu\w+)\s*\(", declaration)
            if match:
                self.functions[match[1]] = declaration.strip() + ";"

        self.aliases = {}
        for declaration in re.findall(r"\btypedef\s+[^;{}]+;", declarations):
            alias = (re.search(r"\(\s*\*\s*(\w+)\s*\)", declaration)
                     or re.search(r"\b(\w+)\s*;$", declaration))
            if alias:
                self.aliases[alias[1]] = declaration

        self.structs = struct_definitions(declarations)
        self.constants = set()
        for body in re.findall(r"\benum\s*(?:\w+\s*)?\{([^{}]*)\}", declarations):
            self.constants.update(name for name in re.findall(r"(?:^|,)\s*(\w+)\b", body)
                                  if not name.startswith(VENDOR_PREFIXES))
        # WebGPU bit flags are static const integers in the C header.
        self.constants.update(re.findall(r"\bstatic\s+const\s+\w+\s+(WGPU\w+)\s*=", declarations))
        for name, value in re.findall(r"^#define (WGPU\w+)[ \t]+([^\n]*)", preprocessed, re.M):
            if value.strip() and not value.startswith("_wgpu_MAKE_INIT_STRUCT("):
                self.constants.add(name)

    def compile(self, source, *args, check=True):
        result = subprocess.run([*self.cc, *args], input=self.includes + source,
                                capture_output=True, text=True)
        if check and result.returncode:
            raise RuntimeError(result.stderr)
        return result


def incompatible_functions(dawn, yawgpu):
    # Check the address before redeclaration so an absent function also fails.
    # Bisect a failing batch to attribute all errors without parsing diagnostics.
    def check(batch):
        checks = "\n".join(f"_Static_assert(sizeof(&{name}), \"{name}\");\n{decl}"
                           for name, decl in batch)
        # Give the redeclarations their original scalar, handle and callback
        # typedefs. Otherwise a change behind the same typedef name is invisible.
        needed = set(re.findall(r"\b\w+\b", checks))
        aliases = []
        for name, declaration in reversed(yawgpu.aliases.items()):
            if name in needed:
                aliases.append(declaration)
                needed.update(re.findall(r"\b\w+\b", declaration))
        source = "\n".join(["void check_signatures(void) {", *reversed(aliases), checks, "}"])
        if dawn.compile(source, "-fsyntax-only", check=False).returncode == 0:
            return []
        if len(batch) == 1:
            return ["function " + batch[0][0]]
        middle = len(batch) // 2
        return check(batch[:middle]) + check(batch[middle:])

    return check(list(yawgpu.functions.items()))


# Keep the full precision of integers and floating-point sentinels (including NaN).
# A non-numeric object macro produces no row.
VALUE_PRINTER = r"""
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
static void signed_value(const char *key, intmax_t value) {
    printf("%s %" PRIdMAX "\n", key, value);
}
static void unsigned_value(const char *key, uintmax_t value) {
    printf("%s %" PRIuMAX "\n", key, value);
}
static void float_value(const char *key, long double value) {
    printf("%s %La\n", key, value);
}
static void skip(const char *key, ...) { (void)key; }
#define VALUE(key, x) _Generic((x), \
    _Bool: unsigned_value, char: signed_value, signed char: signed_value, \
    short: signed_value, int: signed_value, long: signed_value, long long: signed_value, \
    unsigned char: unsigned_value, unsigned short: unsigned_value, unsigned int: unsigned_value, \
    unsigned long: unsigned_value, unsigned long long: unsigned_value, \
    float: float_value, double: float_value, long double: float_value, default: skip)(key, x)
"""


def measurements(header, wanted, executable):
    rows = []
    for name in sorted(wanted.structs.keys() & header.structs.keys()):
        c_type, members = header.structs[name]
        rows.append(f'VALUE("struct {name}", sizeof({c_type}));')
        for member in members:
            rows.append(f'VALUE("member {name}.{member}", offsetof({c_type}, {member}));')
    for name in sorted(wanted.constants & header.constants):
        rows.append(f'VALUE("enum {name}", {name});')
    header.compile(VALUE_PRINTER + "\nint main(void) {\n" + "\n".join(rows) + "\n}\n",
                   "-o", executable)
    output = subprocess.check_output([executable], text=True)
    return dict(line.rsplit(" ", 1) for line in output.splitlines())


def guard(include, work):
    gen_dawn_headers(work)
    cc = shlex.split(os.environ.get("CC", "cc"))
    yawgpu = Header([*cc, "-DYAWGPU_HAS_TILED"], include, ["webgpu.h", "yawgpu.h"])
    dawn = Header(cc, work / "include", ["dawn/webgpu.h"])
    expected = measurements(yawgpu, yawgpu, work / "yawgpu-probe")
    actual = measurements(dawn, yawgpu, work / "dawn-probe")
    problems = incompatible_functions(dawn, yawgpu)
    for key in sorted(expected.keys() | actual.keys()):
        kind, name = key.split()
        # A missing type already identifies all of its missing fields.
        if kind == "member" and name.partition(".")[0] not in dawn.structs:
            continue
        if kind == "enum" and name not in dawn.constants and not name.startswith("WGPU"):
            continue
        if expected.get(key) != actual.get(key):
            problems.append(key)

    counts = [sum(key.startswith(kind + " ") for key in expected)
              for kind in ("struct", "member", "enum")]
    print(f"Checked {len(yawgpu.functions)} functions, {counts[0]} structs, "
          f"{counts[1]} members, {counts[2]} constants.", file=sys.stderr)
    for problem in problems:
        print(problem)
    return int(bool(problems))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yawgpu-include", type=Path, default=DEFAULT_INCLUDE)
    parser.add_argument("--work", type=Path, help="parent directory for temporary probes")
    args = parser.parse_args()
    if args.work:
        args.work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.work) as work:
        return guard(args.yawgpu_include.resolve(), Path(work).resolve())


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
