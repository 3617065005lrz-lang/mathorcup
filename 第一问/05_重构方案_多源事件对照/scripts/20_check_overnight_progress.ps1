$ErrorActionPreference = 'SilentlyContinue'

$Imerg = 'D:\MathorCup_Data\External_Gyirong\P0-2_IMERG\data_late_history'
$EraSnow = 'D:\MathorCup_Data\ERA5_Land_Gyirong\raw_official\q1_history_snow_deepsoil'
$EraPressure = 'D:\MathorCup_Data\ERA5_Land_Gyirong\raw_official\q1_pressure_levels'
$Smap = 'D:\MathorCup_Data\External_Gyirong\P1-8_SMAP_L4_V8'
$GfsRoot = 'D:\MathorCup_Data\External_Gyirong\P0-5_GFS_historical_controls'
$GfsScript = Get-ChildItem -LiteralPath $GfsRoot -Recurse -Filter 'download_gfs_subset.py' -File | Select-Object -First 1
$Gfs = if ($GfsScript) { Join-Path $GfsScript.Directory.Parent.FullName 'P0-5_GFS' } else { $GfsRoot }

Write-Output "Checked: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output '--- active downloader processes ---'
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -match 'python|powershell' -and $_.CommandLine -match '18_download_imerg|19_era5_overnight|12_download_smap|19_gfs_overnight|download_gfs_' } |
    Select-Object ProcessId, Name, @{Name='Task'; Expression={
        if ($_.CommandLine -match '18_download_imerg') {'IMERG'}
        elseif ($_.CommandLine -match '12_download_smap') {'SMAP'}
        elseif ($_.CommandLine -match '19_gfs_overnight|download_gfs_') {'GFS'}
        elseif ($_.CommandLine -match '19_era5_overnight|08_download_era5|16_download_era5') {'ERA5'}
    }} | Format-Table -AutoSize

Write-Output '--- file progress ---'
$ImergCounts = @{}
Get-ChildItem -LiteralPath $Imerg -File | ForEach-Object {
    if ($_.Name -match '3IMERG\.(202[1-4])') { $ImergCounts[$Matches[1]] = 1 + [int]$ImergCounts[$Matches[1]] }
}
$ImergDone = ($ImergCounts.Values | Measure-Object -Sum).Sum
Write-Output ("IMERG: {0}/1536 ({1:N1}%) [{2}]" -f $ImergDone, (100*$ImergDone/1536), (($ImergCounts.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ', '))
Write-Output "ERA5 snow/deep-soil: $(@(Get-ChildItem -LiteralPath $EraSnow -Filter '*.grib' -File).Count)/16"
Write-Output "ERA5 pressure levels: $(@(Get-ChildItem -LiteralPath $EraPressure -Filter '*.grib' -File).Count)/20"

$StatePath = Join-Path $Smap 'harmony_jobs.json'
if (Test-Path -LiteralPath $StatePath) {
    $State = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
    $Entries = @($State.PSObject.Properties | ForEach-Object { $_.Value })
    $Downloaded = @($Entries | Where-Object { $_.downloaded }).Count
    $RemoteActive = @($Entries | Where-Object { -not $_.downloaded -and $_.status -match 'submitted|accepted|running|paused' }).Count
    Write-Output "SMAP months: downloaded=$Downloaded/20, remote-active=$RemoteActive, state-entries=$($Entries.Count)/20"
}
$GfsBase = @(Get-ChildItem -LiteralPath (Join-Path $Gfs 'data') -Filter 'gfs_2023*.nc' -File).Count
$GfsLevels = @(Get-ChildItem -LiteralPath (Join-Path $Gfs 'pressure_supplement') -Filter 'gfs_2023*.nc' -File).Count
Write-Output "GFS 2023 cycles: base=$GfsBase/20, extra-pressure=$GfsLevels/20"

Write-Output '--- newest log tails ---'
$LogRoot = Join-Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) 'logs\overnight_downloads'
Get-ChildItem -LiteralPath $LogRoot -Filter '*.log' -File | Sort-Object LastWriteTime -Descending | Select-Object -First 8 | ForEach-Object {
    Write-Output "[$($_.Name)]"
    Get-Content -LiteralPath $_.FullName -Tail 3
}
