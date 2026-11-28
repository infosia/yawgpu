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
    # A Windows compiler escapes the backslashes of a marker's path and may mix separators.
    def normalized(path):
        return os.path.normcase(os.path.normpath(path))

    paths = {normalized(path) for path in paths}
    lines = []
    public = False
    for line in preprocessed.splitlines():
        marker = re.match(r'^# \d+ "([^"]+)"', line)
        if marker:
            public = normalized(marker[1].replace("\\\\", "\\")) in paths
        elif public:
            lines.append(line)
    if not lines:
        raise ValueError("C preprocessor returned no public header declarations")
    return "\n".join(lines)


def declarators(field):
    """Split a field declaration at commas outside argument lists and array bounds."""
    start = depth = 0
    for i, char in enumerate(field):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            yield field[start:i]
            start = i + 1
    yield field[start:]


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
        members = {}
        for field in body.split(";"):
            if not field.strip():
                continue
            for declarator in declarators(field):
                member = (re.search(r"\(\s*\*\s*(\w+)\s*(?:\[[^\]]*\]\s*)*\)\s*\(", declarator)
                          or re.search(r"\b(\w+)\s*(?:\[[^\]]*\]\s*)*$", declarator))
                if not member:
                    raise ValueError(f"Cannot identify member of {name}: {field}")
                members[member[1]] = field.strip()
        structs[name] = (f"struct {tag}" if tag else alias, members)
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
        # Keep the tag behind a typedef that also defines its struct or enum.
        typedefs = re.sub(r"\btypedef\s+(struct|enum)\s+(\w+)\s*\{[^{}]*\}\s*(\w+)\s*;",
                          r"typedef \1 \2 \3;", declarations)
        for declaration in re.findall(r"\btypedef\s+[^;{}]+;", typedefs):
            alias = (re.search(r"\(\s*\*\s*(\w+)\s*\)", declaration)
                     or re.search(r"\b(\w+)\s*(?:\[[^\]]*\]\s*)*;$", declaration))
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


def incompatible_declarations(dawn, yawgpu):
    probes = []
    for name, declaration in yawgpu.functions.items():
        # Check the address before redeclaration so an absent function also fails.
        check = f'_Static_assert(sizeof(&{name}), "{name}");\n{declaration}'
        probes.append((f"function {name}", check))
    for name in sorted(yawgpu.structs.keys() & dawn.structs.keys()):
        c_type, dawn_members = dawn.structs[name]
        _, fields = yawgpu.structs[name]
        for member in sorted(fields.keys() & dawn_members.keys()):
            # Replay the C field declaration with its typedefs. Comparing addresses
            # preserves array bounds and qualifiers on the member itself.
            key = f"member {name}.{member}"
            actual = f"__typeof__(&(({c_type}*)0)->{member})"
            expected = f"__typeof__(&reference.{member})"
            check = (f"{{ struct {{ {fields[member]}; }} reference;\n"
                     f'_Static_assert(__builtin_types_compatible_p({actual}, {expected}), "{key}"); }}')
            probes.append((key, check))

    # Bisect a failing batch to attribute all errors without parsing diagnostics.
    def check(batch):
        checks = "\n".join(source for _, source in batch)
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
            return [batch[0][0]]
        middle = len(batch) // 2
        return check(batch[:middle]) + check(batch[middle:])

    return check(probes)


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


def check_abi(include, dawn_include, work):
    """Return ABI mismatches against the supplied generated Dawn headers."""
    # The probes need a GCC-compatible driver. YAWGPU_ABI_GUARD_CC names one where CC is
    # a different compiler, as with MSVC on Windows.
    cc = shlex.split(os.environ.get("YAWGPU_ABI_GUARD_CC") or os.environ.get("CC", "cc"))
    yawgpu = Header([*cc, "-DYAWGPU_HAS_TILED"], include, ["webgpu.h", "yawgpu.h"])
    dawn = Header(cc, dawn_include, ["dawn/webgpu.h"])
    expected = measurements(yawgpu, yawgpu, work / "yawgpu-probe")
    actual = measurements(dawn, yawgpu, work / "dawn-probe")
    problems = set(incompatible_declarations(dawn, yawgpu))
    for key in sorted(expected.keys() | actual.keys()):
        kind, name = key.split()
        # A missing type already identifies all of its missing fields.
        if kind == "member" and name.partition(".")[0] not in dawn.structs:
            continue
        if kind == "enum" and name not in dawn.constants and not name.startswith("WGPU"):
            continue
        if expected.get(key) != actual.get(key):
            problems.add(key)

    counts = [sum(key.startswith(kind + " ") for key in expected)
              for kind in ("struct", "member", "enum")]
    print(f"Checked {len(yawgpu.functions)} functions, {counts[0]} structs, "
          f"{counts[1]} members, {counts[2]} constants.", file=sys.stderr)
    return sorted(problems)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yawgpu-include", type=Path, default=DEFAULT_INCLUDE)
    parser.add_argument("--work", type=Path, help="parent directory for temporary probes")
    args = parser.parse_args()
    if args.work:
        args.work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.work) as work:
        work = Path(work).resolve()
        gen_dawn_headers(work)
        problems = check_abi(args.yawgpu_include.resolve(), work / "include", work)
        for problem in problems:
            print(problem)
        return int(bool(problems))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
