// Tint C shim implementation. Mirrors dawn/src/tint/cmd/tint/main.cc for
// Parse, ProgramToLoweredIR, GenerateBindings, SubstituteOverrides, and writer
// option setup.

#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

#include "src/tint/api/common/substitute_overrides_config.h"
#include "src/tint/api/helpers/generate_bindings.h"
#include "src/tint/api/tint.h"
#include "src/tint/lang/core/constant/value.h"
#include "src/tint/lang/glsl/writer/helpers/generate_bindings.h"
#include "src/tint/lang/glsl/writer/writer.h"
#include "src/tint/lang/msl/writer/writer.h"
#include "src/tint/lang/spirv/writer/writer.h"
#include "src/tint/lang/wgsl/ast/identifier.h"
#include "src/tint/lang/wgsl/ast/module.h"
#include "src/tint/lang/wgsl/ast/override.h"
#include "src/tint/lang/wgsl/inspector/inspector.h"
#include "src/tint/lang/wgsl/reader/reader.h"
#include "src/tint/lang/wgsl/sem/variable.h"

#include "tint_shim.h"

struct YawgpuTintProgram {
    tint::Program program;
    std::vector<tint::inspector::EntryPoint> entry_points;
    std::vector<tint::inspector::Override> overrides;
};

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

void set_error_string(char** err, const std::string& message) {
    if (err != nullptr) {
        *err = dup_string(message);
    }
}

std::string cstr_or_empty(const char* s) {
    return s != nullptr ? std::string(s) : std::string();
}

bool all_remaps_empty(const YawgpuTintBindings* bindings) {
    return bindings == nullptr ||
           (bindings->n_uniform == 0 && bindings->n_storage == 0 && bindings->n_texture == 0 &&
            bindings->n_storage_texture == 0 && bindings->n_sampler == 0);
}

void fill_binding_map(tint::BindingMap& map,
                      const YawgpuTintBindingRemap* remaps,
                      size_t count) {
    if (remaps == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        const auto& r = remaps[i];
        map.emplace(tint::BindingPoint{.group = r.group, .binding = r.binding},
                    tint::BindingPoint{.group = r.dst_group, .binding = r.dst_binding});
    }
}

tint::Bindings make_bindings(const YawgpuTintBindings* bindings) {
    tint::Bindings out;
    if (bindings == nullptr) {
        return out;
    }
    fill_binding_map(out.uniform, bindings->uniform, bindings->n_uniform);
    fill_binding_map(out.storage, bindings->storage, bindings->n_storage);
    fill_binding_map(out.texture, bindings->texture, bindings->n_texture);
    fill_binding_map(out.storage_texture, bindings->storage_texture, bindings->n_storage_texture);
    fill_binding_map(out.sampler, bindings->sampler, bindings->n_sampler);
    return out;
}

tint::diag::Result<tint::SubstituteOverridesConfig> make_override_config(
    const YawgpuTintProgram* program,
    const YawgpuTintOverrideValue* values,
    size_t count) {
    tint::SubstituteOverridesConfig cfg;
    if (count == 0) {
        return cfg;
    }

    tint::inspector::Inspector inspector(program->program);
    auto override_names = inspector.GetNamedOverrideIds();
    cfg.map.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (values == nullptr || values[i].name == nullptr || values[i].name[0] == '\0') {
            return tint::diag::Failure("empty override name");
        }
        std::string name(values[i].name);
        char* end = nullptr;
        unsigned long parsed = std::strtoul(name.c_str(), &end, 10);
        if (end != nullptr && *end == '\0' && parsed <= UINT16_MAX) {
            cfg.map.emplace(tint::OverrideId{static_cast<uint16_t>(parsed)}, values[i].value);
            continue;
        }
        auto found = override_names.find(name);
        if (found == override_names.end()) {
            return tint::diag::Failure("unknown override '" + name + "'");
        }
        cfg.map.emplace(found->second, values[i].value);
    }
    return cfg;
}

tint::Result<tint::core::ir::Module> lower_ir(const YawgpuTintProgram* program) {
    tint::wgsl::reader::IROptions options{
        .dump_ir_when_validating = false,
        .enable_validation_asserts = false,
    };
    return tint::wgsl::reader::ProgramToLoweredIR(program->program, options);
}

void fill_entry_point(const tint::inspector::EntryPoint& ep, YawgpuTintEntryPoint* out) {
    out->name = ep.name.c_str();
    out->stage = static_cast<uint8_t>(ep.stage);
    out->has_workgroup_size = ep.workgroup_size.has_value();
    out->wg_x = ep.workgroup_size ? ep.workgroup_size->x : 0;
    out->wg_y = ep.workgroup_size ? ep.workgroup_size->y : 0;
    out->wg_z = ep.workgroup_size ? ep.workgroup_size->z : 0;
    out->frag_depth_used = ep.frag_depth_used;
    out->sample_mask_used = ep.input_sample_mask_used || ep.output_sample_mask_used;
    out->input_sample_mask_used = ep.input_sample_mask_used;
    out->front_facing_used = ep.front_facing_used;
    out->sample_index_used = ep.sample_index_used;
    out->primitive_index_used = ep.primitive_index_used;
    out->subgroup_invocation_id_used = ep.subgroup_invocation_id_used;
    out->subgroup_size_used = ep.subgroup_size_used;
}

const tint::inspector::EntryPoint* find_entry_point(const YawgpuTintProgram* program,
                                                    const char* ep) {
    if (program == nullptr || ep == nullptr) {
        return nullptr;
    }
    std::string name(ep);
    for (const auto& entry : program->entry_points) {
        if (entry.name == name) {
            return &entry;
        }
    }
    return nullptr;
}

void fill_stage_variable(const tint::inspector::StageVariable& variable,
                         YawgpuTintStageVariable* out) {
    out->has_location = variable.attributes.location.has_value();
    out->location = variable.attributes.location.value_or(0);
    out->component_type = static_cast<uint8_t>(variable.component_type);
    out->composition_type = static_cast<uint8_t>(variable.composition_type);
    out->interpolation_type = static_cast<uint8_t>(variable.interpolation_type);
    out->interpolation_sampling = static_cast<uint8_t>(variable.interpolation_sampling);
}

void fill_resource_binding(const tint::inspector::ResourceBinding& binding,
                           YawgpuTintResourceBinding* out) {
    out->group = binding.bind_group;
    out->binding = binding.binding;
    out->resource_type = static_cast<uint8_t>(binding.resource_type);
    out->dim = static_cast<uint8_t>(binding.dim);
    out->sampled_kind = static_cast<uint8_t>(binding.sampled_kind);
    out->sampler_type = static_cast<uint8_t>(binding.sampler_type);
    out->texel_format = static_cast<uint8_t>(binding.image_format);
    out->size = binding.size;
    out->has_array_size = binding.array_size.has_value();
    out->array_size = binding.array_size.value_or(0);
}

double override_default_value(const tint::sem::GlobalVariable* global,
                              tint::inspector::Override::Type type) {
    if (global == nullptr || global->Initializer() == nullptr ||
        global->Initializer()->ConstantValue() == nullptr) {
        return 0.0;
    }
    const auto* value = global->Initializer()->ConstantValue();
    switch (type) {
        case tint::inspector::Override::Type::kBool:
            return value->ValueAs<bool>() ? 1.0 : 0.0;
        case tint::inspector::Override::Type::kFloat32:
            return static_cast<double>(value->ValueAs<tint::core::f32>());
        case tint::inspector::Override::Type::kUint32:
            return static_cast<double>(value->ValueAs<tint::core::u32>());
        case tint::inspector::Override::Type::kInt32:
            return static_cast<double>(value->ValueAs<tint::core::i32>());
        case tint::inspector::Override::Type::kFloat16:
            return static_cast<double>(static_cast<float>(value->ValueAs<tint::core::f16>()));
    }
    return 0.0;
}

const tint::sem::GlobalVariable* find_override_global(const tint::Program& program,
                                                      const tint::inspector::Override& ov) {
    for (auto* decl : program.AST().Globals<tint::ast::Override>()) {
        auto* global = program.Sem().Get(decl);
        if (global == nullptr) {
            continue;
        }
        if (decl->name->symbol.Name() == ov.name) {
            return global;
        }
    }
    return nullptr;
}

void fill_override(const tint::Program& program,
                   const tint::inspector::Override& ov,
                   YawgpuTintOverride* out) {
    out->name = ov.name.c_str();
    out->id = ov.id.value;
    out->type_class = static_cast<uint8_t>(ov.type);
    out->has_default = ov.is_initialized;
    out->default_value =
        ov.is_initialized ? override_default_value(find_override_global(program, ov), ov.type) : 0.0;
}

}  // namespace

extern "C" {

void yawgpu_tint_initialize(void) {
    // Tint's InternalCompilerError destructor is [[noreturn]]. This Dawn/Tint
    // revision exposes optional per-ICE callbacks, but no global setter, so the
    // shim cannot make ICEs catchable or install a process-wide capture hook.
    tint::Initialize();
}

YawgpuTintProgram* yawgpu_tint_program_create(const char* wgsl,
                                              size_t wgsl_len,
                                              bool shader_f16,
                                              char** err) {
    if (err != nullptr) {
        *err = nullptr;
    }
    try {
        if (wgsl == nullptr) {
            set_error_string(err, "WGSL source pointer is NULL");
            return nullptr;
        }
        tint::Source::File file("shader.wgsl", std::string(wgsl, wgsl_len));
        tint::wgsl::reader::Options options;
        if (shader_f16) {
            options.allowed_features = tint::wgsl::AllowedFeatures::Everything();
        }
        tint::Program parsed = tint::wgsl::reader::Parse(&file, options);
        if (!parsed.IsValid()) {
            set_error_string(err, parsed.Diagnostics().Str());
            return nullptr;
        }

        auto* out = new YawgpuTintProgram();
        out->program = std::move(parsed);
        tint::inspector::Inspector inspector(out->program);
        out->entry_points = inspector.GetEntryPoints();
        out->overrides = inspector.Overrides();
        if (inspector.has_error()) {
            set_error_string(err, inspector.error());
            delete out;
            return nullptr;
        }
        return out;
    } catch (const std::exception& e) {
        set_error_string(err, e.what());
        return nullptr;
    } catch (...) {
        set_error_string(err, "unknown Tint exception");
        return nullptr;
    }
}

void yawgpu_tint_program_destroy(YawgpuTintProgram* program) {
    delete program;
}

size_t yawgpu_tint_entry_point_count(const YawgpuTintProgram* program) {
    return program != nullptr ? program->entry_points.size() : 0;
}

bool yawgpu_tint_entry_point_get(const YawgpuTintProgram* program,
                                 size_t i,
                                 YawgpuTintEntryPoint* out) {
    if (program == nullptr || out == nullptr || i >= program->entry_points.size()) {
        return false;
    }
    fill_entry_point(program->entry_points[i], out);
    return true;
}

size_t yawgpu_tint_entry_point_input_count(const YawgpuTintProgram* program, const char* ep) {
    const auto* entry = find_entry_point(program, ep);
    return entry != nullptr ? entry->input_variables.size() : 0;
}

bool yawgpu_tint_entry_point_input_get(const YawgpuTintProgram* program,
                                       const char* ep,
                                       size_t i,
                                       YawgpuTintStageVariable* out) {
    const auto* entry = find_entry_point(program, ep);
    if (entry == nullptr || out == nullptr || i >= entry->input_variables.size()) {
        return false;
    }
    fill_stage_variable(entry->input_variables[i], out);
    return true;
}

size_t yawgpu_tint_entry_point_output_count(const YawgpuTintProgram* program, const char* ep) {
    const auto* entry = find_entry_point(program, ep);
    return entry != nullptr ? entry->output_variables.size() : 0;
}

bool yawgpu_tint_entry_point_output_get(const YawgpuTintProgram* program,
                                        const char* ep,
                                        size_t i,
                                        YawgpuTintStageVariable* out) {
    const auto* entry = find_entry_point(program, ep);
    if (entry == nullptr || out == nullptr || i >= entry->output_variables.size()) {
        return false;
    }
    fill_stage_variable(entry->output_variables[i], out);
    return true;
}

size_t yawgpu_tint_resource_binding_count(const YawgpuTintProgram* program, const char* ep) {
    if (program == nullptr) {
        return 0;
    }
    tint::inspector::Inspector inspector(program->program);
    return inspector.GetResourceBindings(cstr_or_empty(ep)).size();
}

bool yawgpu_tint_resource_binding_get(const YawgpuTintProgram* program,
                                      const char* ep,
                                      size_t i,
                                      YawgpuTintResourceBinding* out) {
    if (program == nullptr || out == nullptr) {
        return false;
    }
    tint::inspector::Inspector inspector(program->program);
    auto bindings = inspector.GetResourceBindings(cstr_or_empty(ep));
    if (i >= bindings.size()) {
        return false;
    }
    fill_resource_binding(bindings[i], out);
    return true;
}

size_t yawgpu_tint_override_count(const YawgpuTintProgram* program) {
    return program != nullptr ? program->overrides.size() : 0;
}

bool yawgpu_tint_override_get(const YawgpuTintProgram* program, size_t i, YawgpuTintOverride* out) {
    if (program == nullptr || out == nullptr || i >= program->overrides.size()) {
        return false;
    }
    fill_override(program->program, program->overrides[i], out);
    return true;
}

bool yawgpu_tint_generate_msl(const YawgpuTintProgram* program,
                              const char* ep,
                              const YawgpuTintBindings* bindings,
                              const YawgpuTintOverrideValue* ov,
                              size_t n_ov,
                              bool disable_robustness,
                              YawgpuTintMslOutput* out,
                              char** err) {
    if (err != nullptr) {
        *err = nullptr;
    }
    if (out != nullptr) {
        out->msl = nullptr;
        out->needs_storage_buffer_sizes = false;
    }
    try {
        if (program == nullptr || out == nullptr) {
            set_error_string(err, "invalid NULL argument");
            return false;
        }
        std::string entry_point = cstr_or_empty(ep);
        auto ir = lower_ir(program);
        if (ir != tint::Success) {
            set_error(err, ir.Failure());
            return false;
        }
        tint::msl::writer::Options options;
        options.entry_point_name = entry_point;
        options.disable_robustness = disable_robustness;
        options.bindings = all_remaps_empty(bindings)
                               ? tint::GenerateBindings(ir.Get(), entry_point, true, true)
                               : make_bindings(bindings);
        options.immediate_binding_point = tint::BindingPoint{.group = 0u, .binding = 30u};
        auto override_cfg = make_override_config(program, ov, n_ov);
        if (override_cfg != tint::Success) {
            set_error(err, override_cfg.Failure());
            return false;
        }
        options.substitute_overrides_config = override_cfg.Get();

        auto result = tint::msl::writer::Generate(ir.Get(), options);
        if (result != tint::Success) {
            set_error(err, result.Failure());
            return false;
        }
        out->msl = dup_string(result->msl);
        out->needs_storage_buffer_sizes = result->needs_storage_buffer_sizes;
        return out->msl != nullptr;
    } catch (const std::exception& e) {
        set_error_string(err, e.what());
        return false;
    } catch (...) {
        set_error_string(err, "unknown Tint exception");
        return false;
    }
}

bool yawgpu_tint_generate_spirv(const YawgpuTintProgram* program,
                                const char* ep,
                                const YawgpuTintBindings* bindings,
                                const YawgpuTintOverrideValue* ov,
                                size_t n_ov,
                                bool disable_robustness,
                                uint32_t** words_out,
                                size_t* n_words_out,
                                char** err) {
    if (err != nullptr) {
        *err = nullptr;
    }
    if (words_out != nullptr) {
        *words_out = nullptr;
    }
    if (n_words_out != nullptr) {
        *n_words_out = 0;
    }
    try {
        if (program == nullptr || words_out == nullptr || n_words_out == nullptr) {
            set_error_string(err, "invalid NULL argument");
            return false;
        }
        std::string entry_point = cstr_or_empty(ep);
        auto ir = lower_ir(program);
        if (ir != tint::Success) {
            set_error(err, ir.Failure());
            return false;
        }
        tint::spirv::writer::Options options;
        options.entry_point_name = entry_point;
        options.disable_robustness = disable_robustness;
        options.bindings = all_remaps_empty(bindings)
                               ? tint::GenerateBindings(ir.Get(), entry_point, false, false)
                               : make_bindings(bindings);
        auto override_cfg = make_override_config(program, ov, n_ov);
        if (override_cfg != tint::Success) {
            set_error(err, override_cfg.Failure());
            return false;
        }
        options.substitute_overrides_config = override_cfg.Get();

        auto result = tint::spirv::writer::Generate(ir.Get(), options);
        if (result != tint::Success) {
            set_error(err, result.Failure());
            return false;
        }
        size_t byte_len = result->spirv.size() * sizeof(uint32_t);
        auto* words = static_cast<uint32_t*>(std::malloc(byte_len));
        if (words == nullptr && byte_len != 0) {
            set_error_string(err, "failed to allocate SPIR-V output");
            return false;
        }
        if (byte_len != 0) {
            std::memcpy(words, result->spirv.data(), byte_len);
        }
        *words_out = words;
        *n_words_out = result->spirv.size();
        return true;
    } catch (const std::exception& e) {
        set_error_string(err, e.what());
        return false;
    } catch (...) {
        set_error_string(err, "unknown Tint exception");
        return false;
    }
}

bool yawgpu_tint_generate_glsl(const YawgpuTintProgram* program,
                               const char* ep,
                               const YawgpuTintBindings* bindings,
                               const YawgpuTintOverrideValue* ov,
                               size_t n_ov,
                               char** glsl_out,
                               char** err) {
    if (err != nullptr) {
        *err = nullptr;
    }
    if (glsl_out != nullptr) {
        *glsl_out = nullptr;
    }
    try {
        if (program == nullptr || glsl_out == nullptr) {
            set_error_string(err, "invalid NULL argument");
            return false;
        }
        std::string entry_point = cstr_or_empty(ep);
        auto ir = lower_ir(program);
        if (ir != tint::Success) {
            set_error(err, ir.Failure());
            return false;
        }
        tint::glsl::writer::Options options;
        options.entry_point_name = entry_point;
        options.version = tint::glsl::writer::Version();
        if (all_remaps_empty(bindings)) {
            auto generated = tint::glsl::writer::GenerateBindings(ir.Get(), entry_point);
            options.bindings = std::move(generated.bindings);
            options.texture_builtins_from_uniform =
                std::move(generated.texture_builtins_from_uniform);
        } else {
            options.bindings = make_bindings(bindings);
        }
        auto override_cfg = make_override_config(program, ov, n_ov);
        if (override_cfg != tint::Success) {
            set_error(err, override_cfg.Failure());
            return false;
        }
        options.substitute_overrides_config = override_cfg.Get();

        auto result = tint::glsl::writer::Generate(ir.Get(), options);
        if (result != tint::Success) {
            set_error(err, result.Failure());
            return false;
        }
        *glsl_out = dup_string(result->glsl);
        return *glsl_out != nullptr;
    } catch (const std::exception& e) {
        set_error_string(err, e.what());
        return false;
    } catch (...) {
        set_error_string(err, "unknown Tint exception");
        return false;
    }
}

void yawgpu_tint_string_free(char* s) {
    std::free(s);
}

void yawgpu_tint_u32_free(uint32_t* words) {
    std::free(words);
}

}  // extern "C"
