// Tint C shim implementation (Phase 1 smoke). Mirrors the reference driver
// dawn/src/tint/cmd/tint/main.cc: Initialize -> Parse -> ProgramToLoweredIR ->
// msl::writer::Generate, with bindings from tint::GenerateBindings.

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#include "src/tint/api/helpers/generate_bindings.h"
#include "src/tint/api/tint.h"
#include "src/tint/lang/msl/writer/writer.h"
#include "src/tint/lang/wgsl/reader/reader.h"

#include "tint_shim.h"

namespace {

char* dup_string(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out != nullptr) {
        std::memcpy(out, s.c_str(), s.size() + 1);
    }
    return out;
}

template <typename Failure>
void set_error(char** err, const Failure& failure) {
    if (err != nullptr) {
        std::stringstream ss;
        ss << failure;
        *err = dup_string(ss.str());
    }
}

}  // namespace

extern "C" {

void yawgpu_tint_initialize(void) {
    tint::Initialize();
}

char* yawgpu_tint_wgsl_to_msl(const char* wgsl, const char* entry_point, char** err) {
    if (err != nullptr) {
        *err = nullptr;
    }

    tint::Source::File file("shader.wgsl", std::string(wgsl != nullptr ? wgsl : ""));
    tint::Program program = tint::wgsl::reader::Parse(&file);
    if (!program.IsValid()) {
        if (err != nullptr) {
            *err = dup_string(program.Diagnostics().Str());
        }
        return nullptr;
    }

    auto ir = tint::wgsl::reader::ProgramToLoweredIR(program);
    if (ir != tint::Success) {
        set_error(err, ir.Failure());
        return nullptr;
    }

    tint::msl::writer::Options options;
    options.entry_point_name = (entry_point != nullptr) ? entry_point : "";
    // (set_group_to_zero, flatten_bindings) = (true, true) -> flat MSL indices,
    // matching a non-argument-buffer Metal binding model.
    options.bindings = tint::GenerateBindings(ir.Get(), options.entry_point_name, true, true);

    auto result = tint::msl::writer::Generate(ir.Get(), options);
    if (result != tint::Success) {
        set_error(err, result.Failure());
        return nullptr;
    }

    return dup_string(result->msl);
}

void yawgpu_tint_string_free(char* s) {
    std::free(s);
}

}  // extern "C"
