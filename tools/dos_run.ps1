# dos_run.ps1 - run the 16-bit build in DOSBox and capture its output.
#
# "It linked" is not "it runs". This is the only check that exercises the
# real target: real-mode segmentation, a 16-bit size_t, DOS file handles,
# no fsync, and 640 KB of conventional memory.
#
# Why a generated .conf instead of -c arguments: PowerShell mangles the
# nested quotes in mount paths, and DOSBox silently runs zero commands
# instead of failing, so the first two attempts produced no output and no
# error at all. A conf file with an [autoexec] block has no quoting layer.

param(
    [string]$Command = "skynet selftest",
    [string]$Exe = "build_dos\skynet.exe",
    [int]$TimeoutSec = 90,
    [switch]$Preserve,
    [string]$OutFile = "OUT.TXT"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$work = Join-Path $root "build\dostest"

# -Preserve keeps the mounted directory between invocations, which is the
# only way to test recovery: the second run must find the vault written by
# the first one and resume from it instead of starting over.
if (-not $Preserve) {
    if (Test-Path $work) { Remove-Item -Recurse -Force $work }
}
New-Item -ItemType Directory -Force $work | Out-Null

$exePath = Join-Path $root $Exe
if (-not (Test-Path $exePath)) { throw "missing $exePath - build it first (tools/test_watcom.sh)" }
# DOSBox needs an 8.3 name, so the file is always copied in as SKYNET.EXE
# and invoked as `skynet`.  Callers pass -Command with that name.
Copy-Item $exePath (Join-Path $work "SKYNET.EXE")

$conf = Join-Path $work "dosbox.conf"
$confText = @"
[sdl]
autolock=false
fullscreen=false
output=surface

[dosbox]
machine=svga_s3
memsize=16
ems=true
xms=true

[cpu]
core=auto
cycles=max

[autoexec]
mount c "$work"
c:
$Command > $OutFile
exit
"@
Set-Content -Path $conf -Value $confText -Encoding ASCII

Write-Host "work dir : $work"
Write-Host "command  : $Command"

$exe = "C:\Users\DartPower\scoop\apps\dosbox\0.74-3\DOSBox.exe"
if (-not (Test-Path $exe)) {
    $cmd = Get-Command dosbox -ErrorAction SilentlyContinue
    if ($cmd) { $exe = $cmd.Source } else { throw "DOSBox not found" }
}

$p = Start-Process -FilePath $exe -ArgumentList @("-conf", "`"$conf`"") -PassThru -WindowStyle Minimized
if (-not $p.WaitForExit($TimeoutSec * 1000)) {
    Write-Host "DOSBox did not exit in $TimeoutSec s, killing" -ForegroundColor Yellow
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

$out = Join-Path $work $OutFile
if (Test-Path $out) {
    Write-Host "--- output ---"
    Get-Content $out
    Write-Host "--- end ---"
    exit 0
} else {
    Write-Host "no $OutFile produced; files in the work dir:" -ForegroundColor Red
    Get-ChildItem $work | Select-Object Name, Length | Format-Table -AutoSize
    exit 1
}
