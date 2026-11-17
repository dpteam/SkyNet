# dos_test.ps1 - acceptance test on the real target: 16-bit DOS in DOSBox.
#
# Everything here is something the host builds cannot prove:
#   - the program loads in real mode (69 KB, 640 KB conventional memory);
#   - a 16-bit size_t does not silently truncate an 8 MB arena request;
#   - the vault works without fsync;
#   - recovery resumes from state written by a *previous process* on a
#     real DOS filesystem, not from a file this script just created.
#
# The build is done by tools/test_watcom.sh (WSL or Linux) beforehand.

param(
    [string]$Exe = "build_dos\skynet.exe",
    [int]$TimeoutSec = 240
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$run = Join-Path $root "tools\dos_run.ps1"
$ver = Join-Path $root "build\verify.exe"

if (-not (Test-Path $ver)) { throw "missing build\verify.exe - run build.ps1 first" }

$fails = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" -ForegroundColor Green }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:fails++ }
}

Write-Host "=== DOS acceptance test ==="

# 1. selftest on DOS
Write-Host "[1] selftest"
$out = & powershell -ExecutionPolicy Bypass -File $run -Exe $Exe -Command "skynet selftest" -TimeoutSec $TimeoutSec -OutFile "SELF.TXT"
$text = $out -join "`n"
# Deliberately not a hardcoded count: it changes every time a check is
# added, and a test that has to be edited after each improvement is a test
# that will eventually be edited to pass.
Check "selftest reports zero failures" ($text -match "\d+ passed, 0 failed")
Check "selftest reached the end"       ($text -match "vault 9: close")

# 2. arena degradation: 8 MB requested on a 640 KB machine
Write-Host "[2] arena degradation"
$out = & powershell -ExecutionPolicy Bypass -File $run -Exe $Exe -Command "skynet run --quiet --max 100 --dir S --out P.TXT" -TimeoutSec $TimeoutSec -OutFile "RUN1.TXT"
$text = $out -join "`n"
Check "arena degraded instead of failing" ($text -match "of 8388608 bytes granted")
Check "the run still completed"           ($text -match "stopped at prime #100")

# 3. longer run with a checkpoint on every super-prime
Write-Host "[3] checkpoints"
$out = & powershell -ExecutionPolicy Bypass -File $run -Preserve -Exe $Exe -Command "skynet run --quiet --max 20000 --ckpt 1 --dir S --out P.TXT" -TimeoutSec $TimeoutSec -OutFile "RUN2.TXT"
$text = $out -join "`n"
Check "checkpoints were written" ($text -match "checkpoints")

# 4. recovery: the next process must resume, not restart
Write-Host "[4] recovery from a previous process"
$out = & powershell -ExecutionPolicy Bypass -File $run -Preserve -Exe $Exe -Command "skynet run --quiet --max 25000 --dir S --out P.TXT" -TimeoutSec $TimeoutSec -OutFile "RUN3.TXT"
$text = $out -join "`n"
Check "second process resumed" ($text -match "resumed at prime #")
Check "second process did not restart" (-not ($text -match "fresh start"))

# 5. the file the DOS program wrote must be mathematically exact
Write-Host "[5] verify the primes file"
$primes = Join-Path $root "build\dostest\P.TXT"
if (Test-Path $primes) {
    $v = & $ver $primes
    $vt = $v -join "`n"
    Check "primes file is an exact prefix" ($vt -match "PASS")
    Write-Host ($vt -split "`n" | Where-Object { $_ -match "lines|last value|mismatches" } | ForEach-Object { "        $_" })
} else {
    Check "primes file exists" $false
}

# 6. EMS: the number that lifts the 64 KB ceiling
Write-Host "[6] EMS memory report"
$out = & powershell -ExecutionPolicy Bypass -File $run -Preserve -Exe $Exe -Command "skynet mem" -TimeoutSec $TimeoutSec -OutFile "MEM.TXT"
$text = $out -join "`n"
Check "EMS driver detected"        ($text -match "ems   : v")
Check "EMS reports free pages"     ($text -match "pages")
Check "the 64 KB ceiling is lifted" ($text -match "no longer applies")
($text -split "`n" | Where-Object { $_ -match "ems|usable" } | ForEach-Object { Write-Host "        $_" })

# 7. paged store round-trip across page boundaries
Write-Host "[7] paged store round-trip"
$out = & powershell -ExecutionPolicy Bypass -File $run -Preserve -Exe $Exe -Command "skynet selftest" -TimeoutSec $TimeoutSec -OutFile "SELF2.TXT"
$text = $out -join "`n"
Check "pool opened in EMS"     ($text -match "pool: 48 KB in 3 EMS pages")
Check "pool round-trip passed" ($text -match "passed, 0 failed")
($text -split "`n" | Where-Object { $_ -match "pool:|maps" } | ForEach-Object { Write-Host "        $_" })

Write-Host ""
if ($fails -eq 0) {
    Write-Host "=== DOS ACCEPTANCE PASSED ===" -ForegroundColor Green
    exit 0
} else {
    Write-Host "=== $fails CHECK(S) FAILED ===" -ForegroundColor Red
    exit 1
}
