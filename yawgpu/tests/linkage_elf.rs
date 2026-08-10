#![cfg(target_os = "linux")]

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const DT_RPATH: i64 = 15;
const DT_RUNPATH: i64 = 29;

#[test]
fn cdylib_carries_origin_runpath() {
    let cdylib = built_cdylib();
    let elf = fs::read(&cdylib)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", cdylib.display()));
    let paths = dynamic_search_paths(&elf)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", cdylib.display()));

    assert!(
        paths
            .iter()
            .flat_map(|path| path.split(':'))
            .any(|component| component == "$ORIGIN"),
        "{} has no $ORIGIN component in DT_RUNPATH or DT_RPATH (found {paths:?})",
        cdylib.display()
    );
}

#[test]
fn consumer_loads_cdylib_without_ld_library_path() {
    let cdylib = built_cdylib();
    let profile_dir = cdylib
        .parent()
        .expect("built cdylib path should have a parent");
    let temp_dir = fresh_temp_dir();
    let copied_cdylib = temp_dir.join("libyawgpu.so");
    fs::copy(&cdylib, &copied_cdylib).unwrap_or_else(|error| {
        panic!(
            "failed to copy {} to {}: {error}",
            cdylib.display(),
            copied_cdylib.display()
        )
    });

    let shim = profile_dir.join("libtint_shim.so");
    if shim.is_file() {
        let copied_shim = temp_dir.join("libtint_shim.so");
        fs::copy(&shim, &copied_shim).unwrap_or_else(|error| {
            panic!(
                "failed to copy {} to {}: {error}",
                shim.display(),
                copied_shim.display()
            )
        });
    }

    let output = Command::new("python3")
        .args([
            "-c",
            "import ctypes, sys; ctypes.CDLL(sys.argv[1]); print('YAWGPU_LOADED')",
        ])
        .arg(&copied_cdylib)
        .env_remove("LD_LIBRARY_PATH")
        .output()
        .unwrap_or_else(|error| {
            if error.kind() == io::ErrorKind::NotFound {
                panic!("python3 was not found; python3 is a repo build prerequisite");
            }
            panic!("failed to spawn python3: {error}");
        });

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "loading {} failed with {}\nstderr:\n{stderr}",
        copied_cdylib.display(),
        output.status
    );
    assert!(
        stdout.contains("YAWGPU_LOADED"),
        "loading {} succeeded but stdout did not contain YAWGPU_LOADED\nstdout:\n{stdout}\nstderr:\n{stderr}",
        copied_cdylib.display()
    );
}

fn built_cdylib() -> PathBuf {
    let test_exe = std::env::current_exe()
        .unwrap_or_else(|error| panic!("failed to locate the current test executable: {error}"));
    let deps_dir = test_exe.parent().unwrap_or_else(|| {
        panic!(
            "test executable has no deps directory: {}",
            test_exe.display()
        )
    });
    let profile_dir = deps_dir.parent().unwrap_or_else(|| {
        panic!(
            "test executable has no profile directory: {}",
            test_exe.display()
        )
    });
    let deps_cdylib = deps_dir.join("libyawgpu.so");
    if deps_cdylib.is_file() {
        return deps_cdylib;
    }

    // The profile-root file is an uplifted hardlink that test-only builds do
    // not refresh, so use it only when the current build wrote no deps copy.
    let profile_cdylib = profile_dir.join("libyawgpu.so");
    assert!(
        profile_cdylib.is_file(),
        "built yawgpu cdylib is absent at {} and fallback {}",
        deps_cdylib.display(),
        profile_cdylib.display()
    );
    profile_cdylib
}

fn fresh_temp_dir() -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after the Unix epoch")
        .as_nanos();
    let path = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("linkage-elf-{}-{unique}", std::process::id()));
    fs::create_dir(&path).unwrap_or_else(|error| {
        panic!(
            "failed to create temporary directory {}: {error}",
            path.display()
        )
    });
    path
}

#[derive(Clone, Copy)]
enum ByteOrder {
    Little,
    Big,
}

impl ByteOrder {
    fn u16(self, bytes: &[u8], offset: usize) -> Result<u16, String> {
        let raw: [u8; 2] = slice_at(bytes, offset, 2)?
            .try_into()
            .expect("length checked");
        Ok(match self {
            Self::Little => u16::from_le_bytes(raw),
            Self::Big => u16::from_be_bytes(raw),
        })
    }

    fn u32(self, bytes: &[u8], offset: usize) -> Result<u32, String> {
        let raw: [u8; 4] = slice_at(bytes, offset, 4)?
            .try_into()
            .expect("length checked");
        Ok(match self {
            Self::Little => u32::from_le_bytes(raw),
            Self::Big => u32::from_be_bytes(raw),
        })
    }

    fn u64(self, bytes: &[u8], offset: usize) -> Result<u64, String> {
        let raw: [u8; 8] = slice_at(bytes, offset, 8)?
            .try_into()
            .expect("length checked");
        Ok(match self {
            Self::Little => u64::from_le_bytes(raw),
            Self::Big => u64::from_be_bytes(raw),
        })
    }

    fn i64(self, bytes: &[u8], offset: usize) -> Result<i64, String> {
        let raw: [u8; 8] = slice_at(bytes, offset, 8)?
            .try_into()
            .expect("length checked");
        Ok(match self {
            Self::Little => i64::from_le_bytes(raw),
            Self::Big => i64::from_be_bytes(raw),
        })
    }
}

#[derive(Clone, Copy)]
struct SectionHeader {
    name: usize,
    offset: usize,
    size: usize,
    link: usize,
    entry_size: usize,
}

fn dynamic_search_paths(elf: &[u8]) -> Result<Vec<String>, String> {
    if elf.get(..4) != Some(b"\x7fELF") {
        return Err("not an ELF file".into());
    }
    if elf.get(4) != Some(&2) {
        return Err("expected an ELF64 file".into());
    }
    let order = match elf.get(5) {
        Some(1) => ByteOrder::Little,
        Some(2) => ByteOrder::Big,
        _ => return Err("ELF file has an unsupported byte order".into()),
    };

    let section_offset = usize_from_u64(order.u64(elf, 40)?, "section header offset")?;
    let section_entry_size = usize::from(order.u16(elf, 58)?);
    let section_count = usize::from(order.u16(elf, 60)?);
    let section_names_index = usize::from(order.u16(elf, 62)?);
    if section_entry_size < 64 {
        return Err(format!(
            "ELF64 section header size is {section_entry_size}, expected at least 64"
        ));
    }

    let mut sections = Vec::with_capacity(section_count);
    for index in 0..section_count {
        let header_offset =
            checked_table_offset(section_offset, section_entry_size, index, "section header")?;
        slice_at(elf, header_offset, section_entry_size)?;
        sections.push(SectionHeader {
            name: usize_from_u32(order.u32(elf, header_offset)?, "section name offset")?,
            offset: usize_from_u64(order.u64(elf, header_offset + 24)?, "section file offset")?,
            size: usize_from_u64(order.u64(elf, header_offset + 32)?, "section size")?,
            link: usize_from_u32(order.u32(elf, header_offset + 40)?, "section link")?,
            entry_size: usize_from_u64(order.u64(elf, header_offset + 56)?, "section entry size")?,
        });
    }

    let section_names_header = sections
        .get(section_names_index)
        .ok_or_else(|| format!("section name table index {section_names_index} is out of range"))?;
    let section_names = slice_at(elf, section_names_header.offset, section_names_header.size)?;
    let dynamic = sections
        .iter()
        .find(|section| c_string(section_names, section.name).is_ok_and(|name| name == b".dynamic"))
        .ok_or_else(|| "ELF file has no .dynamic section".to_owned())?;
    if dynamic.entry_size < 16 {
        return Err(format!(
            ".dynamic entry size is {}, expected at least 16",
            dynamic.entry_size
        ));
    }
    let strings_header = sections.get(dynamic.link).ok_or_else(|| {
        format!(
            ".dynamic string table index {} is out of range",
            dynamic.link
        )
    })?;
    let strings = slice_at(elf, strings_header.offset, strings_header.size)?;
    let dynamic_bytes = slice_at(elf, dynamic.offset, dynamic.size)?;

    let mut runpaths = Vec::new();
    let mut rpaths = Vec::new();
    for entry_offset in (0..dynamic_bytes.len()).step_by(dynamic.entry_size) {
        if dynamic_bytes.len() - entry_offset < 16 {
            break;
        }
        let tag = order.i64(dynamic_bytes, entry_offset)?;
        if tag == 0 {
            break;
        }
        if tag == DT_RUNPATH || tag == DT_RPATH {
            let string_offset = usize_from_u64(
                order.u64(dynamic_bytes, entry_offset + 8)?,
                "dynamic string offset",
            )?;
            let path = String::from_utf8_lossy(c_string(strings, string_offset)?).into_owned();
            if tag == DT_RUNPATH {
                runpaths.push(path);
            } else {
                rpaths.push(path);
            }
        }
    }

    if runpaths.is_empty() {
        Ok(rpaths)
    } else {
        Ok(runpaths)
    }
}

fn checked_table_offset(
    table_offset: usize,
    entry_size: usize,
    index: usize,
    description: &str,
) -> Result<usize, String> {
    let relative = entry_size
        .checked_mul(index)
        .ok_or_else(|| format!("{description} offset overflow"))?;
    table_offset
        .checked_add(relative)
        .ok_or_else(|| format!("{description} offset overflow"))
}

fn slice_at(bytes: &[u8], offset: usize, length: usize) -> Result<&[u8], String> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| "file offset overflow".to_owned())?;
    bytes
        .get(offset..end)
        .ok_or_else(|| format!("file range {offset}..{end} is out of bounds"))
}

fn c_string(bytes: &[u8], offset: usize) -> Result<&[u8], String> {
    let tail = bytes
        .get(offset..)
        .ok_or_else(|| format!("string offset {offset} is out of bounds"))?;
    let length = tail
        .iter()
        .position(|byte| *byte == 0)
        .ok_or_else(|| format!("string at offset {offset} is not NUL-terminated"))?;
    Ok(&tail[..length])
}

fn usize_from_u32(value: u32, description: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{description} does not fit in usize"))
}

fn usize_from_u64(value: u64, description: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{description} does not fit in usize"))
}
