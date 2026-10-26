# ETI 融水模型参数文献清单

## 1. 模型原始文献（已下载）

- Pellicciotti, F., Brock, B., Strasser, U., Burlando, P., Funk, M., & Corripio, J. (2005).
  *An enhanced temperature-index glacier melt model including the shortwave radiation balance*.
  Journal of Glaciology, 51(175), 573–587.
  DOI: <https://doi.org/10.3189/172756505781829124>
- 本地文件：`Pellicciotti_2005_ETI_model.pdf`
- 用途：给出 ETI 模型结构、变量定义、参数量纲和原始标定方法。
- 注意：文献中的 TF、SRF 和温度阈值来自阿尔卑斯冰川，不能直接当作吉隆源区的确定参数；只能作为先验中心或敏感性检验参照。

## 2. 高山亚洲/尼泊尔适用性文献（开放获取链接）

- Litt, M., Shea, J., Wagnon, P., et al. (2019).
  *Glacier ablation and temperature indexed melt models in the Nepalese Himalaya*.
  Scientific Reports, 9, 5264.
  DOI: <https://doi.org/10.1038/s41598-019-41657-5>
- 开放全文：<https://www.nature.com/articles/s41598-019-41657-5>
- 用途：说明高海拔喜马拉雅地区短波辐射、反照率和云量的重要性，以及 TI/ETI 参数具有明显场地依赖性。
- 建模含义：吉隆模型应对 TF、SRF、融化温度阈值和温度递减率设置概率分布或参数区间，不应照搬单一文献常数。

## 3. 本题参数处理原则

1. 模型形式采用温度项与净短波辐射项的可解释分解。
2. 文献参数只作为先验信息，不作为吉隆源区的“实测真值”。
3. 用蒙特卡洛或贝叶斯抽样传播 TF、SRF、温度阈值、反照率及温度订正的不确定性。
4. 分别报告温度项、辐射项及总融水量的贡献，避免得到一个无法解释的黑箱融水指数。
5. 用2021—2024年同期日作为历史背景；2025年不进入背景分布。
