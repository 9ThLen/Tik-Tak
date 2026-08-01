param(
    [int]$Workers = 3,
    [switch]$Foreground
)

$ErrorActionPreference = 'Stop'
$python = 'D:\Programs\D\OPen porject\harmonix_align_venv\Scripts\python.exe'
$repository = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot 'harmonix_align_batch.py'
$groundTruth = Join-Path $repository 'music\ground-truth'
$dataset = Join-Path $groundTruth 'sources\harmonix_annotations\dataset'
$report = Join-Path $groundTruth 'validation\harmonix_alignment'

New-Item -ItemType Directory -Force -Path $report | Out-Null
$arguments = @(
    $script,
    'align',
    '--raw-dir', (Join-Path $groundTruth 'audio\harmonix-raw'),
    '--ready-dir', (Join-Path $groundTruth 'audio\harmonix-ready'),
    '--melspec-dir', (Join-Path $groundTruth 'sources\harmonix_melspecs\melspecs'),
    '--metadata', (Join-Path $dataset 'metadata.csv'),
    '--urls', (Join-Path $dataset 'youtube_urls.csv'),
    '--historical-scores', (Join-Path $dataset 'youtube_alignment_scores.csv'),
    '--report-dir', $report,
    '--workers', $Workers
)

$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

if ($Foreground) {
    & $python @arguments
    exit $LASTEXITCODE
}

$stdout = Join-Path $report 'runner.stdout.log'
$stderr = Join-Path $report 'runner.stderr.log'
$argumentLine = ($arguments | ForEach-Object {
    if ($_ -match '\s') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
}) -join ' '
$process = Start-Process -FilePath $python -ArgumentList $argumentLine -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content -LiteralPath (Join-Path $report 'runner.pid') -Value $process.Id -Encoding ascii
$process.Id
