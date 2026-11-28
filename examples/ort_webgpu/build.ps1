# Builds the ONNX Runtime host against yawgpu's Dawn compatibility layer (Windows / Vulkan).
#
#   $env:ORT = <onnxruntime checkout>; $env:ORT_BUILD = <its build dir>\Release\Release
#   examples\ort_webgpu\build.ps1
#
# Prerequisites: see README.md next to this file. Needs Visual Studio 2022 (C++ workload);
# the script enters its developer environment itself.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
if (-not $env:ORT) { throw 'set ORT to the ONNX Runtime checkout' }
if (-not $env:ORT_BUILD) { throw 'set ORT_BUILD to the directory holding onnxruntime.dll' }
$target = if ($env:YAWGPU_TARGET) { $env:YAWGPU_TARGET } else { "$root\target" }
$outDir = if ($env:OUT_DIR) { $env:OUT_DIR } else { "$root\target\ort_webgpu" }
New-Item -ItemType Directory -Force $outDir | Out-Null

# The Tint shim's import library lives in the newest yawgpu-tint build directory.
$tintLib = Get-ChildItem "$target\release\build" -Recurse -Filter tint_shim.lib |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $tintLib) { throw 'tint_shim.lib not found: build yawgpu-dawn in release first' }

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -property installationPath
$vcvars = "$vs\VC\Auxiliary\Build\vcvars64.bat"

$cl = @(
    'cl', '/nologo', '/std:c++20', '/O1', '/EHsc', '/MD', '/utf-8',
    "`"$root\examples\ort_webgpu\main.cc`"",
    "/I`"$env:ORT\include\onnxruntime\core\session`"",
    "/I`"$root\yawgpu\ffi\webgpu-headers`"", "/I`"$root\yawgpu-dawn\include`"",
    "/Fo`"$outDir\\`"", "/Fe`"$outDir\ort_webgpu.exe`"",
    '/link',
    "`"$target\release\yawgpu_dawn.lib`"", "`"$($tintLib.FullName)`"",
    "`"$env:ORT_BUILD\onnxruntime.lib`"",
    'kernel32.lib', 'ntdll.lib', 'userenv.lib', 'ws2_32.lib', 'dbghelp.lib'
) -join ' '
cmd /c "`"$vcvars`" >nul && $cl"
if ($LASTEXITCODE -ne 0) { throw "cl failed ($LASTEXITCODE)" }

# The Tint shim and ONNX Runtime are loaded from next to the executable.
Copy-Item -Force "$target\release\tint_shim.dll" $outDir
Copy-Item -Force "$env:ORT_BUILD\onnxruntime*.dll" $outDir
Write-Output "built $outDir\ort_webgpu.exe"
