# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:13:29 2024

@author: 18657
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

# 设置中文字体
font_path = "SimHei.ttf"
font_prop = FontProperties(fname=font_path)

# 确保 matplotlib 使用指定的字体
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加复杂的 CSS 样式，紫色高级风格，修复背景颜色问题
st.markdown("""
    <style>
 .main {
        background-color: #3E065F;
        background-image: url('https://www.transparenttextures.com/patterns/bedge-grunge.png');
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
 .title {
        font-size: 48px;
        color: #ffffff;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 3px 3px 10px #2E0854;
    }
 .subheader {
        font-size: 28px;
        color: #FFD700;
        margin-bottom: 25px;
        text-align: center;
        border-bottom: 2px solid #DDA0DD;
        padding-bottom: 10px;
        margin-top: 20px;
    }
 .input-label {
        font-size: 18px;
        font-weight: bold;
        color: #DDA0DD;
        margin-bottom: 10px;
    }
 .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 16px;
        color: #D8BFD8;
        background-color: #2E0854;
        padding: 20px;
        border-top: 1px solid #6A5ACD;
    }
 .button {
        background-color: #8A2BE2;
        border: none;
        color: white;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        margin: 20px auto;
        cursor: pointer;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.5);
        transition: background-color 0.3s, box-shadow 0.3s;
    }
 .button:hover {
        background-color: #6A5ACD;
        box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.7);
    }
 .stSelectbox,.stNumberInput,.stSlider {
        margin-bottom: 20px;
    }
 .stSlider > div {
        padding: 10px;
        background: #E6E6FA;
        border-radius: 10px;
    }
 .prediction-result {
        font-size: 24px;
        color: #ffffff;
        margin-top: 30px;
        padding: 20px;
        border-radius: 10px;
        background: #6A5ACD;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
    }
 .advice-text {
        font-size: 20px;
        line-height: 1.6;
        color: #ffffff;
        background: #8A2BE2;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

# 页面标题
st.markdown('<div class="title">五角场监测站交通污染预测</div>', unsafe_allow_html=True)

# 加载模型
file_path = r"XGBoost2020.pkl"
model = joblib.load(file_path)

if isinstance(model, xgb.XGBClassifier):
    # 尝试调整 booster 参数
    model.set_params(booster='gbtree')

# 定义特征名称
feature_names = ["CO", "FSP", "NO2", "O3", "RSP", "SO2"]

# 定义空气质量类别映射
category_mapping = {
    5: '严重污染',
    4: '重度污染',
    3: '重度污染',
    2: '轻度污染',
    1: '良',
    0: '优'
}

# Streamlit 用户界面
# 一氧化碳浓度
CO = st.number_input("一氧化碳的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0,
                    help="请输入该监测站检测到的一氧化碳在24小时内的平均浓度值，单位为毫克每立方米。")
if CO is None:
    st.warning("一氧化碳浓度输入为空，已将其从本次预测数据中删除。")
    CO = 0.0

# PM2.5浓度
FSP = st.number_input("PM2.5的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0,
                    help="请输入该监测站检测到的PM2.5在24小时内的平均浓度值，单位为毫克每立方米。")
if FSP is None:
    st.warning("PM2.5浓度输入为空，已将其从本次预测数据中删除。")
    FSP = 0.0

# 二氧化氮浓度
NO2 = st.number_input("二氧化氮的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0,
                    help="请输入该监测站检测到的二氧化氮在24小时内的可平均浓度值，单位为毫克每立方米。")
if NO2 is None:
    st.warning("二氧化氮浓度输入为空，已将其从本次预测数据中删除。")
    NO2 = 0.0

# 臭氧浓度
O3 = st.number_input("臭氧的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0,
                    help="请输入该监测站检测到的臭氧在24小时内的平均浓度值，单位为毫克每立方米。")
if O3 is None:
    st.warning("臭氧浓度输入为空，已将其从本次预测数据中删除。", min_value=0.0, value=0.0,
                    help="请输入该监测站检测到的臭氧在24小时内的平均浓度值，单位为毫克每立方米。")
if O3 is None:
    st.warning("臭氧浓度输入为空，已将其从本次预测数据中删除。")
    O3 = 0.0

# PM10浓度
RSP = st.number_input("PM10的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0,
                    help="请输入该监测站检测到的PM10在24小时内的平均浓度值，单位为毫克每立方米。")
if RSP is None:
    st.warning("PM10浓度输入为空，已将其从本次预测数据中删除。")
    RSP = 0.0

# 二氧化硫浓度
SO2 = st.number_input("二氧化硫的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0,
                    help="请输入该监测站检测到的二氧化硫在24小时内的平均浓度值，单位为毫克每立方米。")
if SO2 is None:
    st.warning("二氧化硫浓度输入为空，已将其从本次预测数据中删除。")
    SO2 = 0.0

# 处理输入并进行预测
feature_values = [CO, FSP, NO2, O3, RSP, SO2]
features = np.array([feature_values])
st.write("feature_values after input processing:", feature_values)

if st.button("预测"):
    try:
        if model is not None:
            # 预测类别和概率
            try:
                predicted_class = model.predict(features)[0]
                predicted_proba = model.predict_proba(features)[0]
                st.write("predicted_class:", predicted_class)
                st.write("predicted_proba:", predicted_proba)

                # 检查预测概率是否包含None值
                if any(p is None for p in predicted_proba):
                    st.error("预测概率结果中包含None值，请检查模型或数据！")
                    raise ValueError("预测概率不能包含None值。")

                # 显示预测结果
                st.markdown(f"<div class='prediction-result'>预测类别：{category_mapping[predicted_class]}</div>", unsafe_allow_html=True)

                # 根据预测结果生成建议
                probability = predicted_proba[predicted_class] * 100
                advice = {
                    '严重污染': f"根据我们的库，该日空气质量为严重污染。模型预测该日为严重污染的概率为 {probability:.1f}%。建议采取防护措施，减少户外活动。",
                    '重度污染': f"根据我们的库，该日空气质量为重度污染。模型预测该日为重度污染的概率为 {probability:.1f}%。建议减少外出，佩戴防护口罩。",
                    '中度污染': f"根据我们的库，该日空气质量为中度污染。敏感人群应减少户外活动。",
                    '轻度污染': f"根据我们的库，该日空气质量为轻度污染。模型预测该日为轻度污染的概率为 {probability:.1f}%。可以适当进行户外活动，但仍需注意防护。",
                    '良': f"根据我们的库，此日空气质量为良。模型预测此日空气质量为良的概率为 {probability:.1f}%。可以正常进行户外活动。",
                    '优': f"根据我们的库，该日空气质量为优。模型预测该日空气质量为优的概率为 {probability:.1f}%。空气质量良好，尽情享受户外时光。",
                }[category_mapping[predicted_class]]
                st.markdown(f"<div class='advice-text'>{advice}</div>", unsafe_allow_html=True)

                # 计算SHAP值并绘制shap瀑布图
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
                    st.write("shap_values after calculation:", shap_values)

                    # 递归删除shap_values中所有可能存在的None值
                    shap_values = _recursive_remove_none(shap_values)
                    st.write("shap_values after recursive None value removal:", shap_values)

                    st.write("Shape of shap_values:", np.shape(shap_values))
                    shap_values_2d = np.squeeze(shap_values, axis=0)
                    st.write("shap_values_2d after squeezing:", shap_values_2d)

                    st.write("SHAP values for the first class:")
                    st.write(shap_values[0, 0, :])

                    st.write("First few elements of shap_values:", shap_values_2d[:3])

                    # 获取base_value（通过解释器计算得到）
                    base_value = explainer.expected_value
                    st.write("base_value after calculation:", base_value)
                    if base_value is None:
                        st.error("计算得到的base_value为None，请检查模型或数据！")
                        raise ValueError("base_value不能为None。")

                    # 准备绘制瀑布图的数据
                    feature_name_mapping = {
                        "CO": "一氧化碳浓度",
                        "FSP": "PM2.5浓度",
                        "NO2": "二氧化氮浓度",
                        "O3": "臭氧浓度",
                        "RSP": "PM10浓度",
                        "SO2": "二氧化硫浓度"
                    }
                    features = [feature_name_mapping[f] for f in shap_values_2d.columns.tolist()]  # 获取特征名称
                    contributions = shap_values_2d[predicted_class].values  # 获取特征贡献度

                    # 确保瀑布图的数据是按贡献度绝对值降序排列的
                    sorted_indices = np.argsort(np.abs(contributions))[::-1]
                    features_sorted = [features[i] for i in sorted_indices]
                    contributions_sorted = contributions[sorted_indices]

                    # 初始化绘图
                    fig, ax = plt.subplots(figsize=(14, 8))

                    # 初始化累积值
                    start = 0
                    prev_contributions = [start]  # 起始值为0

                    # 计算每一步的累积值
                    for i in range(1, len(contributions_sorted)):
                        prev_contributions.append(prev_contributions[-1] + contributions_sorted[i - 1])

                    # 绘制瀑布图
                    for i in range(len(contributions_sorted)):
                        color = '#ff5050' if contributions_sorted[i] < 0 else '#66b3ff'  # 负贡献使用红色，正贡献使用蓝色
                        if i == len(contributions_sorted) - 1:
                            # 最后一个条形带箭头效果，表示最终累积值
                            ax.barh(features_sorted[i], contributions_sorted[i], left=prev_contributions[i], color=color, edgecolor='black', height=0.5, hatch='/')
                        else:
                            ax.barh(features_sorted[i], contributions_sorted[i], left=prev_contributions[i], color=color, edgecolor='black', height=0.5)

                        # 在每个条形上显示数值
                        plt.text(prev_contributions[i] + contributions_sorted[i] / 2, i, f"{contributions_sorted[i]:.2f}", 
                                ha='center', va='center', fontsize=10, fontproperties=font_prop, color='black')

                    # 设置图表属性
                    plt.title(f'{category_mapping[predicted_class]} 的特征贡献度瀑布图', fontsize=18, fontproperties=font_prop)
                    plt.xlabel('贡献度 (SHAP值)', fontsize=14, fontproperties=font_prop)
                    plt.ylabel('特征', fontsize=14, fontproperties=font_prop)
                    plt.yticks(fontsize=12, fontproperties=font_prop)
                    plt.xticks(fontsize=12, fontproperties=font_prop)
                    plt.grid(axis='x', linestyle='--', alpha=0.7)

                    # 增加边距避免裁剪
                    plt.xlim(left=0, right=max(prev_contributions) + max(contributions_sorted) * 1.0)
                    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

                    plt.tight_layout()

                    # 保存并在 Streamlit 中展示
                    plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=1200)
                    st.image("shap_waterfall_plot.png")
                except Exception as e:
                    st.error(f"SHAP值计算过程中出现错误：{e}")
                    raise
            except Exception as e:
                st.error(f"预测过程中出现错误：{e}")
                raise
        else:
            st.write("模型加载失败，无法进行预测。")
    except Exception as e:
        st.error(f"整个预测相关操作出现错误：{e}")
        raise


def _recursive_remove_none(data):
    """
    递归删除数据结构中所有的None值
    """
    if isinstance(data, list):
        return [_recursive_remove_none(item) for item in data if item is not None]
    elif isinstance(data, np.ndarray):
        return np.array([_recursive_remove_none(item) for item in data if item is not None])
    elif isinstance(data, tuple):
        return tuple(_recursive_remove_none(item) for item in data if item is not None)
    elif isinstance(data, dict):
        return {key: _recursive_remove_none(value) for key, value in data.items() if value is not None}
    return data

# 页脚
st.markdown('<div class="footer">© 2024 All rights reserved.</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
