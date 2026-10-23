# 第一问绘图运行说明

本模块把“分析结果转缓存”和“读取缓存画图”分成两个独立脚本。缓存已经存在时，修改配色、标题、字号或图例后只运行绘图脚本，不会重新计算特征或48个敏感性场景。

绘图脚本只输出PNG，共4张图；不输出PDF、SVG和流程图。

## 1. 脚本和输出

- `scripts/01_prepare_plot_cache.py`：读取主分析结果并生成`plot_cache`。
- `scripts/02_plot_figures.py`：只读取`plot_cache`并输出PNG。
- `scripts/plot_style.py`：统一字体、颜色、坐标轴和PNG保存参数。
- `plot_cache/`：绘图所需的轻量缓存。
- `figures/`：最终PNG图片。

最终图片：

1. `fig01_top5_trajectories.png`：五个稳定特征在事件前7天的轨迹。
2. `fig02_factor_stability.png`：各物理因素进入前五的比例及排名范围。
3. `fig03_top5_sensitivity_heatmap.png`：前五因素对评价窗口和同期日期带的敏感性。
4. `fig04_event_vs_history.png`：2026灾前窗口与2021—2024历史同期分布比较。

## 2. VS Code中运行

建议直接在VS Code中打开文件夹：

```text
C:\Users\刘润洲\Documents\Codex\2026-10-23\we-j\MathorCup赛道B建模\第一问\03_可视化
```

在VS Code右下角或命令面板中选择已安装`pandas、numpy、matplotlib、pyarrow`的Python解释器，然后打开“终端”。终端提示符末尾应当是：

```text
...\第一问\03_可视化>
```

分析结果未变化且`plot_cache`已经存在时，直接画全部4张图：

```cmd
python .\scripts\02_plot_figures.py --fig all --dpi 300
```

只画一张图，例如图3：

```cmd
python .\scripts\02_plot_figures.py --fig 03 --dpi 300
```

`--fig`可选`01、02、03、04、all`。PNG最低允许300 DPI；如需更高分辨率，可把`300`改为`600`。

只有主分析输出、最终特征、事件时刻或历史年份发生变化时，才先刷新一次缓存：

```cmd
python .\scripts\01_prepare_plot_cache.py
python .\scripts\02_plot_figures.py --fig all --dpi 300
```

## 3. CMD中运行

从任意CMD窗口运行：

```cmd
cd /d "C:\Users\刘润洲\Documents\Codex\2026-10-23\we-j\MathorCup赛道B建模\第一问\03_可视化"
python .\scripts\02_plot_figures.py --fig all --dpi 300
```

需要刷新缓存时，在绘图命令前增加：

```cmd
python .\scripts\01_prepare_plot_cache.py
```

## 4. PowerShell中运行

```powershell
Set-Location -LiteralPath 'C:\Users\刘润洲\Documents\Codex\2026-10-23\we-j\MathorCup赛道B建模\第一问\03_可视化'
python .\scripts\02_plot_figures.py --fig all --dpi 300
```

需要刷新缓存时：

```powershell
python .\scripts\01_prepare_plot_cache.py
python .\scripts\02_plot_figures.py --fig all --dpi 300
```

## 5. 常见问题

### 提示找不到脚本

原因通常是终端仍停留在工作区根目录。先执行本说明中的`cd /d`或`Set-Location`，再运行相对路径命令。

### 提示尚未生成绘图缓存

先运行：

```cmd
python .\scripts\01_prepare_plot_cache.py
```

### 修改样式后是否需要刷新缓存

不需要。修改`scripts/plot_style.py`或`scripts/02_plot_figures.py`后，只运行绘图脚本。

### 输出在哪里

图片保存在：

```text
C:\Users\刘润洲\Documents\Codex\2026-10-23\we-j\MathorCup赛道B建模\第一问\03_可视化\figures
```
