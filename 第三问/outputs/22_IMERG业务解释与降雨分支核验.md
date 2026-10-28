# IMERG业务可用性与独立降雨分支审计

## 1. 为什么需要单独审计

当前缓存是IMERG Late Run V07半小时产品。NASA说明Late Run的名义延迟约为14小时（近期常约12小时），因此它可以作为独立的近实时降雨证据，却不能在起报时使用尚未发布的未来资料。本文采用更保守的14小时截断：起报时刻`t`只允许使用`t-14 h`及以前的IMERG。

官方说明：https://gpm.nasa.gov/resources/faq/what-determines-latency-imerg

## 2. 五次起报的可用资料

 nominal_issue_h  issue_time_beijing  nominal_late_latency_h latest_observable_time_beijing  rain24_available  rain24_mm  rain24_valid_halfhours  rain24_expected_halfhours  rain24_completeness  rain72_available  rain72_mm  rain72_valid_halfhours  rain72_expected_halfhours  rain72_completeness  historical_q90_72h_mm  historical_q95_72h_mm rainfall_branch_supports_high_trigger rainfall_evidence_level               interpretation
             168 2026-08-19 08:00:00                      14            2026-08-18 18:00:00             False        NaN                       0                         48             0.000000             False        NaN                       0                        144             0.000000              39.830499              41.702999                                   NaN                  资料窗不完整              不得把缺测解释为零降雨或弱降雨
             120 2026-08-21 08:00:00                      14            2026-08-20 18:00:00             False        NaN                      20                         48             0.416667             False        NaN                      20                        144             0.138889              39.830499              41.702999                                   NaN                  资料窗不完整              不得把缺测解释为零降雨或弱降雨
              72 2026-08-23 08:00:00                      14            2026-08-22 18:00:00              True      2.130                      48                         48             1.000000             False        NaN                     116                        144             0.805556              39.830499              41.702999                                   NaN                  资料窗不完整              不得把缺测解释为零降雨或弱降雨
              48 2026-08-24 08:00:00                      14            2026-08-23 18:00:00              True      2.935                      48                         48             1.000000              True       5.86                     144                        144             1.000000              39.830499              41.702999                                 False                  未达到q90 可用完整72 h Late Run，仅作为已发生降雨证据
              24 2026-08-25 08:00:00                      14            2026-08-24 18:00:00              True      1.335                      48                         48             1.000000              True       6.40                     144                        144             1.000000              39.830499              41.702999                                 False                  未达到q90 可用完整72 h Late Run，仅作为已发生降雨证据

在具有完整72小时可用窗的2次起报中，达到历史q90（39.83 mm）的次数为0次。资料窗不完整的起报仅标记为“不可判”，不能解释成无雨。

## 3. 事件回溯核验

按真实启动时刻回溯的IMERG 72小时累计降雨为8.325 mm（完整度100.0%，144/144个半小时），低于历史q90=39.83 mm和q95=41.70 mm。这是对“2026事件非强降雨主导”的独立支持，但不证明降雨对坡面水分或河道汇流完全没有作用。

## 4. 论文边界

- IMERG Late是观测型证据，不承担未来1—7天降雨预报；未来降雨仍由GEFS情景提供。
- 该分支只验证降雨是否达到异常触发水平，不与热力分支竞争一套新的最终等级。
- 不能把Late Run的14小时延迟忽略，也不能用事后完整72小时雨量冒充较早起报时已知的信息。
- 不能把缺失窗口填成零降雨。
