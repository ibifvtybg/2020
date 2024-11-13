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
feature_names = ["CO", "FSP", "NO2", "O3", "RSP", "SO2"]

# Streamlit 用户界面
st.title("五角场监测站交通污染预测")

# 一氧化碳浓度
CO = st.number_input("一氧化碳的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0)
if CO is None:
    st.warning("一氧化碳浓度输入为空，已将其从本次预测数据中删除。")
    CO = 0.0

# PM2.5浓度
FSP = st.number_input("PM2.5的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0)
if FSP is None:
    st.warning("PM2.5浓度输入为空，已将其从本次预测数据中删除。")
    FSP = 0.0

# 二氧化氮浓度
NO2 = st.number_input("二氧化氮的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0)
if NO2 is None:
    st.warning("二氧化氮浓度输入为空，已将其从本次预测数据中删除。")
    NO2 = 0.0

# 臭氧浓度
O3 = st.number_input("臭氧的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0)
if O3 is None:
    st.warning("臭氧浓度输入为空，已将其从本次预测数据中删除。")
    O3 = 0.0

# PM10浓度
RSP = st.number_input("PM10的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0)
if RSP is None:
    st.warning("PM10浓度输入为空，已将其从本次预测数据中删除。")
    RSP = 0.0

# 二氧化硫浓度
SO2 = st.number_input("二氧化硫的24小时平均浓度（毫克每立方米）：", min_value=0.0, value=0.0)
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
                                shap_values_for_plot = _recursive_remove_none(shap_values_for_plot)
                                shap_plot_values = shap.Explanation(
                                    values=shap_values_for_plot,
                                    data=pd.DataFrame([feature_values], columns=feature_names),
                                    feature_names=feature_names
                                )
                            else:
                                shap_values_for_plot = shap_values[0]
                                shap_values_for_plot = _recursive_remove_none(shap_values_for_plot)
                                shap_plot_values = shap.Explanation(
                                    values=shap_values_for_plot,
                                    data=pd.DataFrame([feature_values], columns=feature_names),
                                    feature_names=feature_names
                                )
                        else:
                            shap_plot_values = shap_exp
                            # 这里可以添加一些检查确保shap_exp的有效性，比如
                            if not isinstance(shap_exp, shap.Explanation):
                                raise ValueError("shap_exp is not a  valid shap.Explanation object!")

                        # 检查并删除shap_plot_values中可能存在的None值
                        if shap_plot_values is not None:
                            shap_plot_values = shap.Explanation(
                                values=[value for value in shap_plot_values.values if value is not None],
                                data=shap_plot_values.data if shap_plot_values.data is not None else pd.DataFrame([], columns=feature_names),
                                feature_names=shap_plot_values.feature_names if shap_plot_values.feature_names is not None else []
                            )
                            shap_plot_values = _recursive_remove_none(shap_plot_values)
                            st.write("shap_plot_values after recursive None value removal:", shap_plot_values)

                        # 检查data属性对应的DataFrame内是否有None值
                        if shap_plot_values.data is not None:
                            for col in shap_plot_values.data.columns:
                                if any(shap_plot_values.data[col].isnull()):
                                    st.error(f"shap_plot_values的data属性中{col}列存在None值，请检查模型或数据！")
                                    raise ValueError(f"shap_plot_values的data属性中{col}列不能有None值。")

                        try:
                            shap.plots.waterfall(shap_plot_values)
                            plt.savefig(f"shap_waterfall_plot_{sample_idx}_{class_idx}.png", bbox_inches='tight', dpi=1200)
                            st.image(f"shap_waterfall_plot_{sample_idx}_{class_idx}.png")
                        except TypeError as e:
                            st.write(f"绘制瀑布图时发生类型错误：{e}")
                        except ValueError as e:
                            st.write(f"绘制瀑布图时发生值错误：e")
                        except Exception as e:
                            st.write(f"绘制瀑布图过程中出现其他未分类错误：e")
                    else:
                        st.write("指定的类别索引超出范围，请检查预测类别值。")
                except Exception as e:
                    st.write(f"SHAP值计算过程中出现错误：e")
            except Exception as e:
                st.write(f"预测过程中出现错误：e")
        else:
            st.write("模型加载失败，无法进行预测。")
    except Exception as e:
        st.write(f"整个预测相关操作出现错误：e")


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
