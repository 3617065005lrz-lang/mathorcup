$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$dataRoot = Join-Path $root '外源资料'
$eventDir = Join-Path $dataRoot '01_事件时序与机理证据'
$paramDir = Join-Path $dataRoot '03_传播模型参数'
$exposureDir = Join-Path $dataRoot '04_承灾体与人口'

New-Item -ItemType Directory -Force -Path $eventDir, $paramDir, $exposureDir | Out-Null

function Get-FileWithResume {
    param([string]$Url, [string]$Path)
    if ((Test-Path -LiteralPath $Path) -and ((Get-Item -LiteralPath $Path).Length -gt 0)) {
        Write-Host "Existing file kept: $Path"
        return
    }
    Write-Host "Downloading $Url"
    & curl.exe -L --fail --retry 5 --retry-delay 5 --output $Path $Url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url" }
}

# 事件时序、机理和传播参数的权威/技术材料。
Get-FileWithResume 'https://www.zgdzzhyfzxb.com/en/article/pdf/preview/10.16031/j.cghc.202610008.pdf' (Join-Path $eventDir 'Yin_et_al_2026_Gyirong_debris_flow.pdf')
Get-FileWithResume 'https://www.hirosaki-u.ac.jp/wp-content/uploads/2026/09/20260909_topics_a_02en_rev.pdf' (Join-Path $eventDir 'Hirosaki_2026_preliminary_assessment.pdf')
Get-FileWithResume 'https://hirisk.org/wp-content/uploads/2026/08/RHA_NP3_RasuwaFlood-1.pdf' (Join-Path $eventDir 'HiRISK_RasuwaFlood_2026.pdf')
Get-FileWithResume 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/detail/us7000tbwb.geojson' (Join-Path $eventDir 'USGS_us7000tbwb_event.geojson')
Get-FileWithResume 'https://ngf.earthscope.org/geophysical-event/us7000tbwb-nepal-glacial-collapse-and-debris-flow' (Join-Path $eventDir 'EarthScope_event_page.html')
Get-FileWithResume 'https://www.cgs.gov.cn/ywdt/ddyw/202608/t20260828_867531.html' (Join-Path $eventDir 'CGS_event_mechanism.html')

# 开放的地震反演与运动模拟资料；总计约 170 MB。
Get-FileWithResume 'https://zenodo.org/records/22884627/files/far_field_force_inversion.zip?download=1' (Join-Path $paramDir 'far_field_force_inversion.zip')
Get-FileWithResume 'https://zenodo.org/records/22884627/files/near-field-tracking.zip?download=1' (Join-Path $paramDir 'near-field-tracking.zip')
Get-FileWithResume 'https://zenodo.org/records/22884627/files/Sup%20Videos.zip?download=1' (Join-Path $paramDir 'Sup_Videos.zip')
Get-FileWithResume 'https://ramms.ch/ramms-debrisflow/friction-parameters/' (Join-Path $paramDir 'RAMMS_friction_parameters.html')

# 灾前人口和道路资料。OSM 使用 2026-08-01 快照，避免灾后补绘造成信息泄漏。
Get-FileWithResume 'https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/NPL/npl_ppp_2020_UNadj.tif' (Join-Path $exposureDir 'WorldPop_NPL_2020_UNadj_100m.tif')
Get-FileWithResume 'https://download.geofabrik.de/asia/nepal-260801.osm.pbf' (Join-Path $exposureDir 'nepal-260801.osm.pbf')
Get-FileWithResume 'https://download.geofabrik.de/asia/nepal-260801.osm.pbf.md5' (Join-Path $exposureDir 'nepal-260801.osm.pbf.md5')

Write-Host 'Core third-question materials downloaded.'

