$ErrorActionPreference = 'Continue'

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = 'D:\MathorCup_Data\ERA5_Land_Gyirong\.venv\Scripts\python.exe'
$Config = Join-Path (Split-Path -Parent $Here) 'config\07_cds_history_snow_deepsoil.json'

Write-Output "[$(Get-Date -Format s)] ERA5 snow/deep-soil stage started"
& $Python (Join-Path $Here '08_download_era5_land_official.py') --config $Config --workers 1 --max-attempts 8
$SnowExit = $LASTEXITCODE
Write-Output "[$(Get-Date -Format s)] ERA5 snow/deep-soil stage exit=$SnowExit"

if ($SnowExit -eq 0) {
    Write-Output "[$(Get-Date -Format s)] ERA5 pressure-level stage started"
    & $Python (Join-Path $Here '16_download_era5_pressure_levels.py') --max-attempts 8
    Write-Output "[$(Get-Date -Format s)] ERA5 pressure-level stage exit=$LASTEXITCODE"
} else {
    Write-Output 'Pressure-level stage was not started because the first stage did not complete successfully.'
}
