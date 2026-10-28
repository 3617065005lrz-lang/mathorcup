# 第三问脚本顺序

|脚本|用途|是否联网|
|---|---|---|
|`01_download_core_materials.ps1`|下载事件、传播和人口道路辅助资料|是|
|`02_download_gefs_mean_spread.py`|下载7个起报时次、0—174 h的GEFS均值和离散度|是|
|`03_build_gefs_point_cache.py`|从GRIB提取源区点并进行文件质量审计|否|
|`04_run_trigger_forecast.py`|运行确定性基线、1000情景升级和五窗口敏感性|否|
|`05_run_propagation.py`|传播触发情景，计算口岸到达和前缘距离|否|
|`06_evaluate_models.py`|生成基线—升级对照及可靠性边界|否|
|`07_plot_results.py`|只从现有CSV绘制PNG|否|
|`08_build_warning_and_mechanism_outputs.py`|生成四色预警、滚动稳定性和双机制诊断|否|
|`09_audit_imerg_operational_availability.py`|按14 h发布延迟审计每次起报时可用的IMERG Late降雨证据|否|

一般不再单独执行脚本。缓存入口、模型入口和绘图入口分别为项目根目录下的`run_preprocessing.py`、`run_models.py`和`run_plots.py`。

