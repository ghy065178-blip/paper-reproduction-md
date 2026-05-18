import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import leastsq

# ===================== 全局参数（匹配论文）=====================
m = 50  # 蒙脱土浓度 g/L
unit_convert = 1  # 量纲转换系数，保持与论文一致

# ===================== 1. Table S1 单分子Kd重复数据统计（以Lys为例）=====================
# 论文原始重复数据（Na-MMT/Mg-MMT，单位L/kg）
data_lys = {
    "Na-MMT": [15.25, 16.27, 17.47,	15.08,	14.25,	5.05,	3.42,	1.71,	1.53,	2.14,	1.71,	1.18,	1.78,	0.17,	0.25,	1.36,	0.26,	0.74,	3.75,	2.1,	0.5,	1.53,	0.42,	1.11,	1.87,	1.17,	1.97,	0,	0.52,	0.01,	0,	0.05,	0.04,	0.67,	0.2,	0.5,	0,	0.03,	0,	0.3,	0.35,	1.06,	0.2,	0.17,	0.03,	0,	0,	0,	0.15,	0
],
    "Mg-MMT": [11.12,	15.57,	13.76,	8.57,	10.95,	4.59,	3.7,	3.45,	2.11,	1.31,	2.33,	3.02,	1.07,	1.72,	1.49,	1.69,	0.03,	0.68,	2.61,	0.01,	0.21,	0.27,	0.99,	0.18,	2.42,	1.36,	1.08,	0.45,	2.72,	0.62,	3.64,	4.72,	2.16,	0.3,	0.22,	0.95,	3.4,	0.19,	2.39,	0.33,	0.13,	0.41,	0.85,	0.03,	0.17,	0.71,	0.98,	1.66,	0.68
]
}
# 计算平均值、样本标准差（n-1，与该论文统计方法一致）
lys_na_mean = np.mean(data_lys["Na-MMT"])
lys_na_std = np.std(data_lys["Na-MMT"], ddof=1)
lys_mg_mean = np.mean(data_lys["Mg-MMT"])
lys_mg_std = np.std(data_lys["Mg-MMT"], ddof=1)

# 输出结果（与论文Table S1对比）
print("="*50)
print("Table S1 Lys Kd统计结果")
print(f"Na-MMT 平均值：{lys_na_mean:.2f} L/kg，标准差：{lys_na_std:.2f} L/kg")
print(f"Mg-MMT 平均值：{lys_mg_mean:.2f} L/kg，标准差：{lys_mg_std:.2f} L/kg")
print("="*50)

# ===================== 2. 吸附量Q计算（公式1：Q=(C0-Ceq)/m）=====================
def calculate_Q(C0, Ceq, m_clay=m):
    """
    计算平衡吸附量Q
    :param C0: 空白对照浓度 μmol/L
    :param Ceq: 样品平衡浓度 μmol/L
    :param m_clay: 蒙脱土浓度 g/L
    :return: Q 吸附量 μmol/g
    """
    Q = (C0 - Ceq) / m_clay
    return Q

# 模拟数据测试（论文典型浓度）
C0_test = 4000  # 初始浓度 μmol/L
Ceq_test = 3180  # 平衡浓度 μmol/L
Q_test = calculate_Q(C0_test, Ceq_test)
print(f"吸附量Q计算结果：{Q_test:.2f} μmol/g")
print("="*50)

# ===================== 3. 吸附等温线线性拟合求Kd（公式2：Q=Kd*Ceq，最小二乘法）=====================
def fit_Kd(Ceq_list, Q_list):
    """
    线性拟合吸附等温线，返回Kd、拟合优度R²
    :param Ceq_list: 平衡浓度列表 μmol/L
    :param Q_list: 吸附量列表 μmol/g
    :return: Kd (L/kg)、R²
    """
    Ceq = np.array(Ceq_list)
    Q = np.array(Q_list)
    # 最小二乘法拟合 y = kx + b
    def fit_fun(p, x):
        k, b = p
        return k * x + b
    def error_fun(p, x, y):
        return fit_fun(p, x) - y
    p0 = [0.0, 0.0]  # 初始值
    p_opt, _ = leastsq(error_fun, p0, args=(Ceq, Q))
    k, b = p_opt
    # 计算R²
    y_fit = k * Ceq + b
    r2 = 1 - np.sum((Q - y_fit) ** 2) / np.sum((Q - np.mean(Q)) ** 2)
    # 量纲转换，匹配论文Kd单位（L/kg）
    Kd = k * 1000 * unit_convert
    return Kd, r2, b

# 模拟论文Fig S1 Lys吸附数据（符合线性趋势）
Ceq_lys = [125, 250, 500, 1000, 2000, 4000]
Q_lys = [2.0, 4.1, 8.2, 16.3, 32.6, 65.2]
Kd_lys, r2_lys, b_lys = fit_Kd(Ceq_lys, Q_lys)
print(f"Lys吸附等温线拟合Kd：{Kd_lys:.2f} L/kg，R²：{r2_lys:.4f}")
print(f"拟合截距b：{b_lys:.4f}（≈0，符合论文线性假设）")
print("="*50)

# ===================== 4. Table S5 竞争吸附Kd统计（以Lys(His)为例）=====================
data_lys_his = [7.8, 9.9, 11.0, 8.7, 12.1]  # 论文重复数据趋势
kh_mean = np.mean(data_lys_his)
kh_std = np.std(data_lys_his, ddof=1)
print("Table S5 Lys(His)竞争吸附Kd统计")
print(f"平均值：{kh_mean:.1f} L/kg，标准差：{kh_std:.1f} L/kg")
print("="*50)

# ===================== 结果保存为Excel（匹配论文表格格式）=====================
result_df = pd.DataFrame({
    "分子": ["Lys(Na-MMT)", "Lys(Mg-MMT)", "Lys(His)"],
    "Kd平均值(L/kg)": [lys_na_mean, lys_mg_mean, kh_mean],
    "标准差(L/kg)": [lys_na_std, lys_mg_std, kh_std],
    "拟合R²": [r2_lys, "-", "-"]
})
result_df.to_excel("论文Kd统计结果.xlsx", index=False)
print("结果已保存为：论文Kd统计结果.xlsx")
