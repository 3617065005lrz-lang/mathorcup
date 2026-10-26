$ErrorActionPreference = 'Continue'

$Python = 'D:\MathorCup_Data\ERA5_Land_Gyirong\.venv\Scripts\python.exe'
$GfsRoot = 'D:\MathorCup_Data\External_Gyirong\P0-5_GFS_historical_controls'
$SubsetScript = Get-ChildItem -LiteralPath $GfsRoot -Recurse -Filter 'download_gfs_subset.py' -File |
    Select-Object -First 1
if (-not $SubsetScript) { throw 'download_gfs_subset.py was not found below the GFS root.' }
$Scripts = $SubsetScript.Directory.FullName
$BundledDeps = Join-Path (Split-Path -Parent $Scripts) 'python_deps'
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$BundledDeps;$env:PYTHONPATH" } else { $BundledDeps }

Write-Output "[$(Get-Date -Format s)] GFS regional cycles started"
& $Python $SubsetScript.FullName --workers 6
$BaseExit = $LASTEXITCODE
Write-Output "[$(Get-Date -Format s)] GFS regional cycles exit=$BaseExit"

if ($BaseExit -eq 0) {
    Write-Output "[$(Get-Date -Format s)] GFS supplementary pressure levels started"
    & $Python (Join-Path $Scripts 'download_gfs_levels.py') --workers 6
    Write-Output "[$(Get-Date -Format s)] GFS supplementary pressure levels exit=$LASTEXITCODE"
} else {
    Write-Output 'Supplementary levels were not started because regional cycles remain incomplete.'
}
