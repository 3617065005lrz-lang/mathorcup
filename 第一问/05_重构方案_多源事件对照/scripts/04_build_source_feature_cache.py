from __future__ import annotations

"""由外部ERA5-Land处理结果构造源区日尺度特征缓存。

注意：当前输入来自公开复核项目的派生CSV，而不是本项目自行下载的ERA5-Land原始文件。
因此该缓存用于方法试跑和交叉核验；提交前应在获得CDS凭据后按相同定义自行重算。
"""

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent
INPUT_PATH = WORK_DIR / "external" / "gyirong-2026" / "data" / "processed" / "daily_series.csv"
CACHE_DIR = WORK_DIR / "cache"
OUTPUT_DIR = WORK_DIR / "outputs"
WINDOWS = [24, 48, 72, 120, 168]


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(INPUT_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    output = pd.DataFrame({"date": daily["date"]})
    definitions: list[dict[str, object]] = []

    def add(name: str, values: pd.Series, factor_id: str, factor: str, chinese: str, fields: str, definition: str, window: int, unit: str) -> None:
        output[name] = values
        definitions.append(
            {
                "衍生指标": name,
                "因素组ID": factor_id,
                "物理因素": factor,
                "指标中文名": chinese,
                "对应原始字段": fields,
                "透明定义": definition,
                "窗口小时": window,
                "单位": unit,
                "数据层": "源区局地（ERA5-Land派生）",
            }
        )

    for hours in WINDOWS:
        days = hours // 24
        suffix = f"_{hours}h"
        min_periods = max(1, int(np.ceil(days * 0.8)))
        roll = lambda series: series.rolling(days, min_periods=min_periods)
        add(f"src_tmean{suffix}", roll(daily["t_source"]).mean(), "TH01", "空气温度与热量累积", f"源区{hours}小时平均气温", "t_source", f"截止日前{days}个完整北京时间日的源区订正气温均值", hours, "degC")
        add(f"src_tmax{suffix}", roll(daily["t_source_max"]).max(), "TH01", "空气温度与热量累积", f"源区{hours}小时最高气温", "t_source_max", f"截止日前{days}个完整北京时间日的源区订正最高气温", hours, "degC")
        add(f"src_pdd{suffix}", roll(daily["t_source"].clip(lower=0)).sum(), "TH01", "空气温度与热量累积", f"源区{hours}小时近似正积温", "t_source", f"截止日前{days}个完整日max(日均温,0)之和", hours, "degC d")
        add(f"src_warm_days{suffix}", roll((daily["t_source"] > 0).astype(float)).sum(), "TH01", "空气温度与热量累积", f"源区{hours}小时正温日数", "t_source", "窗口内日均订正气温高于0摄氏度的日数", hours, "d")
        add(f"src_temp_anomaly_mean{suffix}", roll(daily["t_source_anom"]).mean(), "TH01", "空气温度与热量累积", f"源区{hours}小时气温距平", "t_source_anom", "相对逐日气候态的源区订正气温距平均值", hours, "degC")
        add(f"src_precip_sum{suffix}", roll(daily["precip"].clip(lower=0)).sum(), "PR01", "降水与液态水输入", f"源区区域{hours}小时累计降水", "precip", f"分析网格截止日前{days}个完整日平均降水之和", hours, "mm")
        add(f"src_freezing_level_mean{suffix}", roll(daily["freezing_level"]).mean(), "TH01", "空气温度与热量累积", f"源区区域{hours}小时0摄氏度层平均高度", "freezing_level", "温度—高程回归得到的0摄氏度等温线高度均值", hours, "m")
        add(f"src_lapse_rate_mean{suffix}", roll(daily["lapse_slope"]).mean() * 1000.0, "TH01", "空气温度与热量累积", f"源区区域{hours}小时气温垂直递减率", "lapse_slope", "逐日空间温度—高程回归斜率均值", hours, "K km-1")

    output.to_parquet(CACHE_DIR / "source_daily_features_provisional.parquet", index=False)
    pd.DataFrame(definitions).to_csv(OUTPUT_DIR / "09_源区ERA5衍生指标字典_暂定.csv", index=False, encoding="utf-8-sig")
    print(f"source daily cache={output.shape}; {output['date'].min().date()} to {output['date'].max().date()}")
    print("注意：这是公开派生结果的复核缓存，提交前需用官方ERA5-Land自行重算。")


if __name__ == "__main__":
    main()
