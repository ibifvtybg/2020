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

# 加载模型
file_path = r"XGBoost2020.pkl"
model = joblib.load(file_path)

if isinstance(model, xgb.XGBClassifier):
    # 尝试调整 booster 参数
    model.set_params(booster='gbtree')

# 定义特征名称
feature_names = ['CO', 'FSP', 'NO2', 'O3', 'RSP', 'SO2']

# Streamlit 用户界面
st.title("五角场监测站交通污染预测")

# 一氧化碳浓度
CO = st.number_input("一氧化碳的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0)

# PM2.5浓度
FSP = st.number_input("PM2.5的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0)

# 二氧化氮浓度
NO2 = st.number_input("二氧化氮的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0)

# 臭氧浓度
O3 = st.number_input("臭氧的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0)

# PM10浓度
RSP = st.number_input("PM10的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0)

# 二氧化硫浓度
SO2 = st.number_input("二氧化硫的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0)

# 处理输入并进行预测
feature_values = [CO, FSP, NO2, O3, RSP, SO2]
features = np.array([feature_values])

if st.button("预测"):
    # 预测类别和概率
    if model is not None:
        try:
            predicted_class = model.predict(features)[0]
            predicted_proba = model.predict_proba(features)[0]

            # 显示预测结果
            st.write(f"**预测类别：** {predicted_class}")
            st.write(f"**预测概率：** {predicted_proba}")

            # 根据预测结果生成建议
            probability = predicted_proba[predicted_class] * 100

            if predicted_class == 5:
                advice = (
                    f"根据我们的模型，该日空气质量为严重污染。"
                    f"模型预测该日为严重污染的概率为 {probability:.1f}%。"
                    "建议采取防护措施，减少户外活动。"
                )
            elif predicted_class == 4:
                advice = (
                    f"根据我们的模型，该日空气质量为重度污染。"
                    f"模型预测该日为重度污染的概率为 {probability:.1f}%。"
                    "建议减少外出，佩戴防护口罩。"
                )
            elif predicted_class == 3:
                advice = (
                    f"根据我们的模型，该日空气质量为中度污染。"
                    f"敏感人群应减少户外活动。"
                )
            elif predicted_class == 2:
                advice = (
                    f"根据我们的模型，该日空气质量为轻度污染。"
                    f"模型预测该日为轻度污染的概率为 {probability:.1f}%。"
                    "可以适当进行户外活动，但仍需注意防护。"
                )
            elif predicted_class == 1:
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

            # 计算SHAP值并绘制shap瀑布图
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
                base_value = explainer.expected_value

                st.write("Shape of shap_values:", np.shape(shap_values))
                st.write("First few elements of shap_values:", shap_values[:3])

                if len(shap_values) > 0:
                    for i in range(len(shap_values)):
                        shap_exp = shap.Explanation(shap_values[i][0], base_value[i], data=pd.DataFrame([feature_values], columns=feature_names))
                        shap.plots.waterfall(shap_exp)
                        plt.savefig(f"shap_waterfall_plot_{i}.png", bbox_inches='tight', dpi=1200)
                        st.image(f"shap_waterfall_plot_{i}.png")
                else:
                    st.write("无法计算 SHAP 值。")
            except Exception as e:
                st.write(f"SHAP 值计算过程中出现错误：{e}")
        except Exception as e:
            st.write(f"预测过程中出现错误：{e}")
    else:
        st.write("模型加载失败，无法进行预测。")
