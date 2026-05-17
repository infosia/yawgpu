pub(crate) struct ValidatedWgslModule {
    pub module: naga::Module,
    pub info: naga::valid::ModuleInfo,
}

pub(crate) fn parse_and_validate_wgsl(src: &str) -> Result<ValidatedWgslModule, String> {
    let module = naga::front::wgsl::parse_str(src).map_err(|error| error.to_string())?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    );
    let info = validator
        .validate(&module)
        .map_err(|error| error.to_string())?;
    Ok(ValidatedWgslModule { module, info })
}

#[cfg(test)]
mod tests {
    use super::parse_and_validate_wgsl;

    #[test]
    fn parses_and_validates_trivial_wgsl() {
        let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
        assert!(parse_and_validate_wgsl(source).is_ok());
    }

    #[test]
    fn rejects_invalid_wgsl() {
        assert!(parse_and_validate_wgsl("not wgsl @@@").is_err());
    }
}
