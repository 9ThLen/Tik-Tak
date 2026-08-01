$ErrorActionPreference = 'Stop'
$python = 'D:\Programs\D\OPen porject\harmonix_align_venv\Scripts\python.exe'
$repository = Split-Path -Parent $PSScriptRoot
$validator = Join-Path $PSScriptRoot 'harmonix_validate_ready.py'
$groundTruth = Join-Path $repository 'music\ground-truth'
$dataset = Join-Path $groundTruth 'sources\harmonix_annotations\dataset'
$report = Join-Path $groundTruth 'validation\harmonix_alignment'
$alignmentPidPath = Join-Path $report 'runner.pid'

if (Test-Path -LiteralPath $alignmentPidPath) {
    $alignmentPid = [int](Get-Content -LiteralPath $alignmentPidPath)
    while (Get-Process -Id $alignmentPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 15
    }
}

& $python $validator `
    --ready-dir (Join-Path $groundTruth 'audio\harmonix-ready') `
    --report-dir $report `
    --metadata (Join-Path $dataset 'metadata.csv') `
    --normalized-dir (Join-Path $groundTruth 'normalized\harmonix') `
    --melspec-dir (Join-Path $groundTruth 'sources\harmonix_melspecs\melspecs') `
    --manifest (Join-Path $groundTruth 'manifest.csv') `
    --validation-report (Join-Path $groundTruth 'validation_report.json') `
    --update-manifest
exit $LASTEXITCODE
