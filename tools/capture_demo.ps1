# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Capture demo_reproduce_0710.ps1 as a timed terminal recording.
#
# Each output line is written to demo_recording.tsv as "<ms since start>\t<line>", read from the
# child process as it appears, so the replay carries the run's real pacing rather than a guess.
# All of the child's streams are merged inside the child (*>&1) so a single reader cannot
# deadlock on an undrained stderr pipe.

param(
    [string]$Script = "demo_reproduce_0710.ps1",
    [string]$Out
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not $Out) {
    $Out = [System.IO.Path]::GetFileNameWithoutExtension($Script) -replace '^demo_reproduce_', 'demo_recording_'
    $Out = "$Out.tsv"
}
$out = Join-Path $root $Out
Remove-Item $out -ErrorAction SilentlyContinue

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "powershell"
$psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -Command `"& { .\$Script *>&1 }`""
$psi.WorkingDirectory = $root
$psi.RedirectStandardOutput = $true
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
if ($env:ROUTER_PY) { $psi.EnvironmentVariables["ROUTER_PY"] = $env:ROUTER_PY }

$writer = [System.IO.StreamWriter]::new($out, $false, [System.Text.UTF8Encoding]::new($false))
$proc = [System.Diagnostics.Process]::Start($psi)
$start = Get-Date
while ($true) {
    $line = $proc.StandardOutput.ReadLine()
    if ($null -eq $line) { break }
    $ms = [int]((Get-Date) - $start).TotalMilliseconds
    $writer.WriteLine("$ms`t$line")
    Write-Host $line
}
$proc.WaitForExit()
$writer.Close()

$total = [int]((Get-Date) - $start).TotalSeconds
$lines = (Get-Content $out).Count
Write-Host ""
Write-Host "captured $lines lines over $total s -> $out (child exit $($proc.ExitCode))"
exit $proc.ExitCode
