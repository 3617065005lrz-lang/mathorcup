$ErrorActionPreference = 'SilentlyContinue'

Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class AwakeState {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
'@

$Continuous = [uint32]0x80000000
$SystemRequired = [uint32]0x00000001
$AwayModeRequired = [uint32]0x00000040
$Needles = '18_download_imerg|19_era5_overnight|08_download_era5|16_download_era5|12_download_smap|19_gfs_overnight|download_gfs_|22_download_gee'

[AwakeState]::SetThreadExecutionState($Continuous -bor $SystemRequired -bor $AwayModeRequired) | Out-Null
try {
    while ($true) {
        $Active = Get-CimInstance Win32_Process |
            Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match $Needles }
        if (-not $Active) { break }
        Start-Sleep -Seconds 60
    }
}
finally {
    [AwakeState]::SetThreadExecutionState($Continuous) | Out-Null
}
