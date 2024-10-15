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
import matplotlib.pyplot as plt

# 加载模型
file_path = r"C:/Users/18657/Desktop/人工智能/XGBoost.pkl"
model = joblib.load(file_path)

# 定义特征名称
feature_names = ['CO', 'FSP', 'NO2', 'O3', 'RSP', 'SO2']

# Streamlit用户界面
st.title("五角场监测站交通污染预测app")

# 一氧化碳浓度
CO = st.number_input("一氧化碳的24小时平均浓度（毫克每立方米）：", min_value=0, value=0)


# PM2.5浓度
FSP = st.number_input("PM2.5的24小时平均浓度（毫克每立方米）：", min_value=0, value=0)


# 二氧化氮浓度
NO2= st.number_input("二氧化氮的24小时平均浓度（毫克每立方米）：", min_value=0, value=0)

# 臭氧浓度
O3 = st.number_input("臭氧的24小时平均浓度（毫克每立方米）：", min_value=0, value=0)

# PM10浓度
RSP = st.number_input("PM10的24小时平均浓度（毫克每立方米）：", min_value=0, value=0)

# 二氧化硫浓度
SO2 = st.number_input("二氧化硫的24小时平均浓度（毫克每立方米）：", min_value=0, value=0)

# 处理输入并进行预测
feature_values = [CO, FSP, NO2, O3, RSP, SO2]
features = np.array([feature_values])

if st.button("预测"):
    # 预测类别和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**预测类别：** {predicted_class}")
    st.write(f"**预测概率：** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 6:
        advice = (
            f"根据我们的模型，该日空气质量为严重污染。"
            f"模型预测该日为严重污染的概率为 {probability:.1f}%。"
            "建议采取防护措施，减少户外活动。"
        )
    elif predicted_class == 5:
        advice = (
            f"根据我们的模型，该日空气质量为重度污染。"
            f"模型预测该日为重度污染的概率为 {probability:.1f}%。"
            "建议减少外出，佩戴防护口罩。"
        )
    elif predicted_class == 4:
        advice = (
            f"根据我们的模型，该日空气质量为中度污染。"
            f"模型预测该日为中度污染的概率为 {probability:.1f}%。"
            "敏感人群应减少户外活动。"
        )
    elif predicted_class == 3:
        advice = (
            f"根据我们的模型，该日空气质量为轻度污染。"
            f"模型预测该日为轻度污染的概率为 {probability:.1f}%。"
            "可以适当进行户外活动，但仍需注意防护。"
        )
    elif predicted_class == 2:
        advice = (
            f"根据我们的模型，该日空气质量为良。"
            f"模型预测该日空气质量为良的概率为 {probability:.1f}%。"
            "可以正常进行户外活动。"
        )
    else:
        advice = (
            f"根据我们的模型，该日空气质量为优。"
            f"模型预测该日空气质量为优的概率为 {probability:.1f}%。"
            "空气质量良好，尽情享受户外时光。"
        )

    st.write(advice)

    # 计算SHAP值并显示力图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(
        pd.DataFrame([feature_values], columns=feature_names)
    )

    shap.force_plot(
        explainer.expected_value, shap_values[0],
        pd.DataFrame([feature_values], columns=feature_names),
        matplotlib=True
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")