param(
    [int]$Expected = 8,
    [int]$RefreshSeconds = 20,
    [string]$DataDirectory = 'D:\MathorCup_Data\ERA5_Land_Gyirong\raw_official',
    [switch]$Once
)

do {
    $Files = @(Get-ChildItem -LiteralPath $DataDirectory -Filter '*.grib' -File -ErrorAction SilentlyContinue)
    $Completed = $Files.Count
    $Percent = if ($Expected -gt 0) { [math]::Min(100, [math]::Round(100 * $Completed / $Expected, 1)) } else { 0 }

    if (-not $Once) { Clear-Host }
    Write-Host "ERA5-Land download progress: $Completed / $Expected ($Percent%)"
    Write-Host "Checked at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Data directory: $DataDirectory"
    Write-Host ''
    $Files |
        Sort-Object Name |
        Select-Object Name, @{Name = 'MB'; Expression = { [math]::Round($_.Length / 1MB, 2) }}, LastWriteTime |
        Format-Table -AutoSize

    if ($Completed -ge $Expected) {
        Write-Host 'All expected files are present.' -ForegroundColor Green
        break
    }
    if (-not $Once) {
        Write-Host "Refreshing every $RefreshSeconds seconds. Press Ctrl+C to stop watching."
        Start-Sleep -Seconds $RefreshSeconds
    }
} while (-not $Once)
