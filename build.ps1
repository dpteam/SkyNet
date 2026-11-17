# build.ps1 - build SkyNet with whatever compiler is available.
#
# The point of this script is the portability check: every unit must build
# clean with -Wall -Wextra -pedantic under C89, and the selftest must pass,
# before anything is committed.

param(
    [ValidateSet("gcc", "tcc", "all")]
    [string]$Compiler = "all",
    [switch]$RunTests
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Push-Location $root

$src = @(
    "src/main.c", "src/sk_math.c", "src/sk_arena.c", "src/sk_clock.c",
    "src/sk_vault.c", "src/sk_sched.c", "src/sk_io.c", "src/sk_mem.c", "src/sk_selftest.c"
)

New-Item -ItemType Directory -Force build | Out-Null

function Invoke-Gcc {
    Write-Host "[build] gcc -std=c89 -pedantic -Wall -Wextra"
    & gcc -std=c89 -pedantic -Wall -Wextra -O2 -o skynet.exe @src
    if ($LASTEXITCODE -ne 0) { throw "gcc build failed" }
    & gcc -std=c89 -O2 -Isrc -o build\verify.exe tests\verify.c src\sk_clock.c
    if ($LASTEXITCODE -ne 0) { throw "verify build failed" }
    Write-Host "[build] gcc ok"
}

function Invoke-Tcc {
    $t = Get-Command tcc -ErrorAction SilentlyContinue
    if (-not $t) {
        Write-Host "[build] tcc not found, skipping" -ForegroundColor Yellow
        return
    }
    Write-Host "[build] tcc (portability check)"
    & tcc -o build\skynet_tcc.exe @src
    if ($LASTEXITCODE -ne 0) { throw "tcc build failed" }
    Write-Host "[build] tcc ok"
}

function Invoke-Unit32 {
    # Compile every unit for 32-bit without linking: this is what proves
    # "long is exactly 32 bits" is handled, since we have no 32-bit CRT here.
    Write-Host "[build] 32-bit unit compile"
    New-Item -ItemType Directory -Force build\obj32 | Out-Null
    foreach ($f in $src) {
        $name = [System.IO.Path]::GetFileNameWithoutExtension($f)
        & gcc -std=c89 -pedantic -Wall -Wextra -m32 -O2 -Isrc -c $f -o "build\obj32\$name.o"
        if ($LASTEXITCODE -ne 0) { throw "32-bit compile failed: $f" }
    }
    Write-Host "[build] 32-bit ok"
}

switch ($Compiler) {
    "gcc" { Invoke-Gcc }
    "tcc" { Invoke-Tcc }
    "all" { Invoke-Gcc; Invoke-Tcc; Invoke-Unit32 }
}

if ($RunTests) {
    Write-Host "[test] selftest"
    & .\skynet.exe selftest
    if ($LASTEXITCODE -ne 0) { throw "selftest failed" }
    if (Test-Path build\skynet_tcc.exe) {
        Write-Host "[test] selftest under tcc build"
        & .\build\skynet_tcc.exe selftest
        if ($LASTEXITCODE -ne 0) { throw "tcc selftest failed" }
    }
}

Pop-Location
Write-Host "[build] done"
