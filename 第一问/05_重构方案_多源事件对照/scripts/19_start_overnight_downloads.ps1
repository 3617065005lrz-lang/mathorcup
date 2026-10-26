$ErrorActionPreference = 'Stop'

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = 'D:\MathorCup_Data\ERA5_Land_Gyirong\.venv\Scripts\python.exe'
$LogRoot = Join-Path (Split-Path -Parent $Here) 'logs\overnight_downloads'
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

function Test-DownloaderRunning([string]$Needle) {
    $Match = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*$Needle*" -and $_.ProcessId -ne $PID }
    return [bool]$Match
}

function Start-LoggedProcess([string]$Name, [string]$Executable, [string[]]$Arguments, [string]$Needle) {
    if (Test-DownloaderRunning $Needle) {
        Write-Output "$Name already running; no duplicate started."
        return
    }
    $Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $StdOut = Join-Path $LogRoot "${Name}_${Stamp}.out.log"
    $StdErr = Join-Path $LogRoot "${Name}_${Stamp}.err.log"
    $Process = Start-Process -FilePath $Executable -ArgumentList $Arguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $StdOut -RedirectStandardError $StdErr
    Write-Output "$Name started: PID=$($Process.Id)"
    Write-Output "  stdout=$StdOut"
    Write-Output "  stderr=$StdErr"
}

Start-LoggedProcess 'IMERG' $Python @((Join-Path $Here '18_download_imerg_late_history_earthaccess.py')) '18_download_imerg_late_history_earthaccess.py'
Start-LoggedProcess 'ERA5_SNOW' $Python @(
    (Join-Path $Here '08_download_era5_land_official.py'),
    '--config', (Join-Path (Split-Path -Parent $Here) 'config\07_cds_history_snow_deepsoil.json'),
    '--workers', '2', '--max-attempts', '8'
) '08_download_era5_land_official.py'
Start-LoggedProcess 'ERA5_PRESSURE' $Python @((Join-Path $Here '16_download_era5_pressure_levels.py'), '--max-attempts', '8') '16_download_era5_pressure_levels.py'
Start-LoggedProcess 'SMAP' $Python @((Join-Path $Here '12_download_smap_l4_harmony.py')) '12_download_smap_l4_harmony.py'
Start-LoggedProcess 'GFS' 'powershell.exe' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $Here '19_gfs_overnight_chain.ps1')) '19_gfs_overnight_chain.ps1'
Start-LoggedProcess 'KEEP_AWAKE' 'powershell.exe' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $Here '19_keep_awake_while_downloading.ps1')) '19_keep_awake_while_downloading.ps1'

Write-Output "All independent chains were checked at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')."
Write-Output "Run 20_check_overnight_progress.ps1 at any time for a compact status report."
