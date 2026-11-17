# crash_test.ps1 - kill SkyNet mid-run, restart, verify the output is
# still a contiguous prime sequence with no gaps and no duplicates.
#
# This is the M0 acceptance test: "a run may lose at most the work since
# the last checkpoint, and it must never resume from a corrupt state."
#
# Two things make this test honest:
#   - the killed rounds are UNBOUNDED, so the process can never finish on
#     its own and every kill really lands in the middle of the work;
#   - the final round is bounded to "a little more than we already have",
#     so it must resume from the vault rather than start over.

param(
    [int]$Rounds = 15,
    [int]$MinKillMs = 40,
    [int]$MaxKillMs = 400,
    [int]$Extra = 20000
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$exe  = Join-Path $root "skynet.exe"
$ver  = Join-Path $root "build\verify.exe"
$work = Join-Path $root "build\crashtest"

if (-not (Test-Path $exe) -or -not (Test-Path $ver)) {
    Write-Host "building first (build.ps1 -Compiler gcc)" -ForegroundColor Yellow
    & (Join-Path $root "build.ps1") -Compiler gcc
}

if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Force $work | Out-Null

$state = Join-Path $work "state"
$out   = Join-Path $work "primes.txt"

Write-Host "crash test: $Rounds unbounded kills, then resume"

$rng = New-Object System.Random 12345

for ($i = 1; $i -le $Rounds; $i++) {
    # --max 0 means "never stop", so this round cannot end by itself
    $p = Start-Process -FilePath $exe `
        -ArgumentList @("run","--quiet","--max","0","--dir",$state,"--out",$out) `
        -PassThru -WindowStyle Hidden
    Start-Sleep -Milliseconds ($rng.Next($MinKillMs, $MaxKillMs))
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    $p.WaitForExit()
    # counting lines with Get-Content on a growing multi-megabyte file is
    # the slowest thing in this script; File.ReadLines stops at EOF fast
    $n = 0
    if (Test-Path $out) {
        $n = [System.IO.File]::ReadAllLines($out).Length
    }
    Write-Host ("  round {0,2}: killed, {1} primes durable" -f $i, $n)
}

$before = 0
if (Test-Path $out) { $before = (Get-Content $out).Count }
$target = $before + $Extra
Write-Host "resuming from $before, running to $target"

& $exe run --quiet --max $target --dir $state --out $out | Out-Null

& $ver $out
exit $LASTEXITCODE
