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
    try:
        if model is not None:
            # 预测类别和概率
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
                        f"根据我们的库，该日空气质量为严重污染。"
                        f"模型预测该日为严重污染的概率为 {probability:.1f}%。"
                        "建议采取防护措施，减少户外活动。"
                    )
                elif predicted_class == 4:
                    advice = (
                        f"根据我们的库，该日空气质量为重度污染。"
                        f"模型预测该日为重度污染的概率为 {probability:.1f}%。"
                        "建议减少外出，佩戴防护口罩。"
                    )
                elif predicted_class == 3:
                    advice = (
                        f"根据我们的库，该日空气质量为中度污染。"
                        f"敏感人群应减少户外活动。"
                    )
                elif predicted_class == 2:
                    advice = (
                        f"根据我们的库，该日空气质量为轻度污染。"
                        f"模型预测该日为轻度污染的概率为 {probability:.1f}%。"
                        "可以适当进行户外活动，但仍需注意防护。"
                    )
                elif predicted_class == 1:
                    advice = (
                        f"根据我们的库，此日空气质量为良。"
                        f"模型预测此日空气质量为良的概率为 {probability:.1f}%。"
                        "可以正常进行户外活动。"
                    )
                else:
                    advice = (
                        f"根据我们的库，该日空气质量为优。"
                        f"模型预测该日空气质量为优的概率为 {probability:.1f}%。"
                        "空气质量良好，尽情享受户外时光。"
                    )

                st.write(advice)

                # 计算SHAP值并绘制shap瀑布图
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
                    st.write("Shape of shap_values:", np.shape(shap_values))
                    shap_values_2d = np.squeeze(shap_values, axis=0)
                    st.write("Shape of shap_values_2d:", np.shape(shap_values_2d))

                    st.write("SHAP values for the first class:")
                    st.write(shap_values[0, 0, :])

                    st.write("First few elements of shap_values:", shap_values_2d[:3])

                    # 获取base_value（通过解释器计算得到）
                    base_value = explainer.expected_value

                    # 只绘制第一个样本（索引为0）的第predicted_class + 1个类别
                    if 0 <= predicted_class < shap_values_2d.shape[1] - 1:
                        sample_idx = 0
                        class_idx = predicted_class 
                        shap_value_param = shap_values_2d[sample_idx][class_idx]
                        base_value_param = base_value[sample_idx]
                        data_param = pd.DataFrame([feature_values], columns=feature_names)
                        shap_value_param = np.array([shap_value_param])

                        # 根据模型类型确定用于绘制瀑布图的SHAP值及创建正确的Explanation对象
                        if isinstance(model, xgb.XGBClassifier) and hasattr(model, 'n_classes_'):
                            # 对于多输出模型（这里判断是否是XGBClassifier且有n_classes_）
                            if model.n_classes_ > 1:
                                shap_values_for_plot = shap_values[0, 0]
                                shap_plot_values = shap.Explanation(
                                    values=shap_values_for_plot,
                                    data=pd.DataFrame([feature_values], columns=feature_names),
                                    feature_names=feature_names
                                )
                            else:
                                shap_values_for_plot = shap_values[0]
                                shap_plot_values = shap.Explanation(
                                    values=shap_values_for_plot,
                                    data=pd.DataFrame([feature_values], columns=feature_names),
                                    feature_names=feature_names
                                )
                        else:
                            shap_plot_values = shap_exp
                            # 这里可以添加一些检查确保shap_exp的有效性，比如
                            if not isinstance(shap_exp, shap.Explanation):
                                raise ValueError("shap_exp is not a valid shap.Explanation object!")

                        try:
                            shap.plots.waterfall(shap_plot_values)
                            plt.savefig(f"shap_waterfall_plot_{sample_idx}_{class_idx}.png", bbox_inches='tight', dpi=1200)
                            st.image(f"shap_waterfall_plot_{sample_idx}_{class_idx}.png")
                        except Exception as e:
                            st.write(f"绘制瀑布图过程中出现错误：{e}")
                    else:
                        st.write("指定的类别索引超出范围，请检查预测类别值。")
                except Exception as e:
                    st.write(f"SHAP值计算过程中出现错误：{e}")
            except Exception as e:
                st.write(f"预测过程中出现错误：{e}")
        else:
            st.write("模型加载失败，无法进行预测。")
    except Exception as e:
        st.write(f"整个预测相关操作出现错误：{e}")
