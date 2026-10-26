# 电脑3外源资料交接（第二问）

更新日期：2026-10-26。负责人：电脑3（用户）。任务分配：P0-3 Sentinel-2、P1-3 Landsat、P1-7 WorldCover、P2-4 BIPAD。

## 当前交付

- **P1-7 WorldCover：已下载** ESA WorldCover 2020 v100 和 2021 v200 的官方 N27E084 瓦片。两份文件均为 36,000 × 36,000 像元、EPSG:4326、约 10 m，覆盖吉隆研究区；本地源文件放在 `D:\MathorCup_Data\External_Gyirong\P1-7_WorldCover\raw_official`，仓库副本位于 `data/worldcover/` 并由 Git LFS 管理。SHA-256、字节数和来源见 `download_manifest.csv`。
- **P0-3 Sentinel-2：部分完成** 已查 CDSE STAC，检出55个候选；已下载六个灾前 SAFE 产品（2026-06-03、2026-06-05、2026-06-15、2026-08-12，以及 2026-08-24 的 Sentinel-2B R076 和 Sentinel-2A R119），另有三景灾后独立验证产品（2026-08-27 Sentinel-2B R119、2026-09-03 Sentinel-2B R076、2026-09-08 Sentinel-2C R076），共 9 景，实际合计 7,692,647,053 bytes，均通过 ZIP CRC 与 SHA-256 校验，且包含 B02/B03/B04/B08/B11/B12 与 SCL。本地原始文件保存在 D 盘；仓库副本位于 `data/sentinel2/` 并由 Git LFS 管理。逐景哈希见 `download_manifest.csv`。六景灾前产品整景云量分别约36.11%、27.47%、18.42%、18.75%、38.57%、50.97%；三景灾后验证产品云量分别为78.47%、69.48%、62.75%。这些整景云量不能代替 AOI 内云量；解压后仍需检查 AOI SCL 有效像元。灾后影像仅独立验证。
- **P1-3 Landsat：部分完成** USGS LandsatLook C2 L2 SR 检出26个候选。仅作为 Sentinel-2 云/时间缺口的备份，不应不加筛选批量下载。目录内保留优先候选。灾后日期仅作验证；Landsat 温度波段是地表温度产品，不等同于2米气温。
- **P2-4 BIPAD：失败/待重试** 官方事件 API 在本次运行中超时，未获得或验证记录。附查询脚本和接口地址；网络恢复后重跑。不得据此声称“没有事件”。一般洪水、滑坡、雪崩记录不能自动标成泥石流。

GitHub 检索未找到可确认对应 2026 年 MathorCup 赛道B第二问的公开仓库，因此这里不引用未核实的他人代码/结论。

## 文件包

`data/sentinel2/` 包含已核验的 9 个 SAFE ZIP，`data/worldcover/` 包含 2020 与 2021 两个官方 GeoTIFF。原始栅格与影像通过 Git LFS 提交；仅拉取普通 Git 仓库不会下载这些大文件，需安装 Git LFS 并执行 `git lfs pull`。

## 目录

- `download_manifest.csv`：来源、状态、外部路径、校验与使用边界。
- `catalog/sentinel2_candidates.csv`、`catalog/landsat_candidates.csv`：官方 STAC 检索后的优先候选，不是完整检索结果。
- `catalog/worldcover_classes.csv`：WorldCover 官方类别值。
- `scripts/query_satellite_catalogs.py`：重跑 Sentinel-2 和 Landsat STAC 查询并输出完整检索清单。
- `scripts/query_bipad_events.py`：重试 BIPAD 官方事件接口并保存原始响应。
- `scripts/download_worldcover_tile.py`：必要时重取官方 WorldCover 瓦片并做大小/哈希校验。

## 复现

在仓库根目录运行：

```powershell
python "第二问/电脑3外源资料/scripts/query_satellite_catalogs.py" --out "第二问/电脑3外源资料/catalog"
python "第二问/电脑3外源资料/scripts/query_bipad_events.py" --out "D:/MathorCup_Data/External_Gyirong/P2-4_BIPAD"
python "第二问/电脑3外源资料/scripts/download_worldcover_tile.py" --out "D:/MathorCup_Data/External_Gyirong/P1-7_WorldCover/raw_official"
```

## 方法边界

AOI 目录检索框约为 WGS84 `[85.19, 28.25, 85.69, 28.56]`，用于找覆盖候选，不是精确裁剪范围。云量为整景属性，不能代替研究区云/阴影掩膜。Sentinel-2 建议使用 B02/B03/B04/B08/B11/B12 与 SCL；先按 AOI 有效像元筛选。时间窗：灾前 2026-06-01 至 2026-08-25；灾后 2026-08-26 至 2026-09-15。所有灾后数据均需标记 `post_event_validation_only=true`，不得混入灾前预测特征或训练标签泄漏。

WorldCover 2020 为 v100，2021 为 v200，产品版本不同；二者像元差异不能直接解释为地表变化。附件5中已有 `jilong_LUCD_2020..2023.tif`，其像元约30m且来源版本信息未在现有压缩包中说明；WorldCover 是独立的10m分类产品，可用于对照，使用时需报告分类体系和分辨率差异。ESA WorldCover 按 CC-BY 4.0 提供，引用产品名、年份、版本并注明 ESA WorldCover。

`downloaded`、`partial`、`requires_login`、`not_available`、`skipped_duplicate`、`not_needed`、`failed` 是本交接采用的状态枚举。超过单项2GB或计划总量10GB的下载先暂停并评估；不要把大体量原始影像放入仓库。
