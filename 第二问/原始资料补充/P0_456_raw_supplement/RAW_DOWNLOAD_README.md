# P0-4 至 P0-6 原始资料补充交接

交付根目录为本仓库的 `第二问/原始资料补充/P0_456_raw_supplement/`。本目录只登记与交付原始文件、官方元数据和完整性证据；没有生成建模结果或派生变量。

## 本轮结果

- 原有帮助交接 ZIP 未修改。原清单记录的实际 ZIP 路径为 本机工作区中的原始 ZIP（本分支不包含该 ZIP）（提示词所写 `D:\帮助_交接包_完善版.zip` 在本机不存在）。该实际 ZIP 开始前与结束时 SHA-256 均为 `fbf1b4690bb7ccd1554e2699ff530de1358b8f6f5acb42bf8197e0c5bbe6b947`（83,538,529 字节）。
- Sentinel-1：7 个原始分卷共 6,902,550,039 字节，逐个大小和 SHA-256 复核通过；原始分卷仍在清单登记的本机微信文件目录，未再次下载。4 个精密轨道 EOF 与每景成像起止时间匹配，EOF 哈希和有效时间覆盖复核通过。
- 4 个额外背景 GRD 仅列出候选产品、大小、轨道、极化、日期和覆盖范围，状态为 `awaiting_user_approval`，没有下载。SLC 未下载。
- NOAA GFS：820 个官方 `.inv` 对应 820 个起报时次/预报时效组合；共选出 3240 条消息。补齐后 `.grib2` 共 566,701,759 字节，文件数、消息数、计划字节数完全一致。每条消息均从 `GRIB` 开始、以 `7777` 结束，长度、原始字节区间及 SHA-256 均核验通过。本轮新写入 770 个原始文件；初始已存在且哈希有效的 50 个文件复用。
- GFS 实际存在项：地表反照率 `ALBDO` 800 条、雪深 `SNOD` 820 条、积雪水当量 `WEASD` 820 条、地表向下长波 `DLWRF` 800 条。`.inv` 中 `SNOC/SNOWC` 地表项目为 0 条，因此登记为 `not_available`，未猜测替代变量。
- GFS 非事件历史对照（2021–2024）只建立可用性与估算目录，状态为 `catalog_only`，没有下载。估算以一个 5 日目标窗口 566,701,759 字节为参照；具体日期和归档 `.inv` 尚未查询。
- Copernicus DEM：原有主 DSM、EDM/FLM/HEM/WBM、XML、KML 和许可 PDF 共 8 个资产的大小与 SHA-256 通过；5 个 TIFF 均可读取，均为 3600×3600、EPSG:4326、1 角秒。XML 记载水平 WGS84-G1150、垂直 WGS 84 Geoid EGM08，瓦片版本 1；许可证文件说明 Copernicus WorldDEM-30 为 full, free & open。相邻 8 个 1° 瓦片只列候选，预计每个主 DSM 约 38,402,839 字节（按现有瓦片大小估算，不含辅助层），均未下载。

## GFS 版本与发布时间说明

GFS 官方产品页把所用产品识别为 NCEI grid-004、0.5°；EMC 官方页面记录 GFS v16 的实施，保存的 NWS 通告包含 2025 年 v16.4.0 提案及 2026 年 v17 提案。现有公开材料不足以逐一证明 2026 年 8 月各周期的准确业务次版本，也不能从归档 HTTP `Last-Modified` 推断业务首次发布时间。因此准确次版本和实际首次可用时间登记为 `unresolved`；不以归档更新时间冒充首发时间。官方资料确认 GFS 每日 00/06/12/18 UTC 起报，但本次没有找到能证明每个周期实际发布时间延迟的正式材料。各 `.inv` / Range 下载响应的 `Last-Modified`、ETag、日期等原始 HTTP 响应头保存在对应 `.http.json` 或 GRIB provenance 中。

## 文件位置与核验

- `P0-4_Sentinel-1_raw/catalog/`：原始分卷位置/校验、轨道映射和额外背景影像候选表。
- `P0-4_Sentinel-1_raw/orbit/`：4 个 EOF 及来源 ZIP、响应头文件。
- `P0-5_GFS_raw_supplement/inventory/`：820 份原始 `.inv`、响应头、精选消息目录、变量可用性及历史对照计划。
- `P0-5_GFS_raw_supplement/grib2/`：逐起报时次/预报时效保存的 `.grib2` 与 `.provenance.json`；provenance 包含 `.inv` 行、源 URL、源字节起止位置、起报时间、时效、有效时间、单位、层次和消息哈希。
- `P0-5_GFS_raw_supplement/official_docs/`：NCEI/EMC 官方产品页与版本通告副本。
- `P0-6_DEM_metadata/`：DEM 原文件校验路径、可读性记录、原始元数据/许可副本、邻接瓦片计划。DEM TIFF 仍留在已验证的原交接目录中，未重复复制。
- `download_manifest_supplement.csv`：数据来源、产品、时间、格式、状态、体量和限制。
- `file_inventory_supplement.csv`：交付目录内文件的大小与 SHA-256；为避免自引用，该文件本身不列入自身清单。
- `logs/final_validation.json` 和 `logs/gfs_download_results.json`：完成情况和机器可读核验结果。

本轮新增下载约 532.8 MB（其中 GFS 原始 GRIB 为 532,509,344 字节，其余为少量 NOAA 官方文档）；低于 10 GiB 总量限制，单个输出文件低于 2 GiB。未使用账号凭证。
