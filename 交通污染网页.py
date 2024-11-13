import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from xgboost import XGBClassifier
import xgboost as xgb

# 设置中文字体
font_path = "SimHei.ttf"
font_prop = FontProperties(fname=font_path)

# 确保matplotlib使用指定的字体
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加蓝色主题的CSS样式，修复背景颜色问题
st.markdown("""
    <style>
   .main {
        background-color: #007BFF;
        background-image: url('https://www.transparenttextures.com/patterns/light_blue_fabric.png');
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
   .title {
        font-size: 48px;
        color: #ffffff;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 3px 3px 10px #0056b3;
    }
   .subheader {
        font-size: 28px;
        color: #99CCFF;
        margin-bottom: 25px;
        text-align: center;
        border-bottom: 2px solid #80BFFF;
        padding-bottom: 10px;
        margin-top: 20px;
    }
   .input-label {
        font-size: 18px;
        font-weight: bold;
        color: #ADD8E6;
        margin-bottom: 10px;
    }
   .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 16px;
        color: #D8BFD8;
        background-color: #0056b3;
        padding: 20px;
        border-top: 1px solid #6A5ACD;
    }
   .button {
        background-color: #0056b3;
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
        background-color: #003366;
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
        background: #4682B4;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
    }
   .advice-text {
        font-size: 20px;
        line-height: 1.6;
        color: #ffffff;
        background: #5DADE2;
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

# 加载XGBoost模型
try:
    model = joblib.load('XGBoost2020.pkl')
except Exception as e:
    st.write(f"<div style='color: red;'>Error loading model: {e}</div>", unsafe_allow_html=True)
    model = None

# 获取模型输入特征数量及顺序
model_input_features = ["CO", "FSP", "NO2", "O3", "RSP", "SO2"]
expected_feature_count = len(model_input_features)

# 定义空气质量类别映射
category_mapping = {
    5: '严重污染',
    4: '重度污染',
    3: '重度污染',
    2: '轻度污染',
    1: '良',
    0: '优'
}

# Streamlit界面设置
st.markdown('<div class="subheader">请填写以下信息以进行交通污染预测：</div>', unsafe_allow_html=True)

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

def predict():
    try:
        # 检查模型是否加载成功
        if model is None:
            st.write("<div style='color: red;'>模型加载失败，无法进行预测。</div>", unsafe_allow_html=True)
            return

        # 获取用户输入并构建特征数组
        user_inputs = {
            "CO": int(CO),
            "FSP": int(FSP),
            "NO2": int(NO2),
            "O3": int(O3),
            "RSP": int(RSP),
            "SO2": int(SO2)
        }

        feature_values = [user_inputs[feature] for feature in model_input_features]
        features_array = np.array([feature_values])

        # 使用XGBoost模型进行预测
        predicted_class = model.predict(features_array)[0]
        predicted_proba = model.predict_proba(features_array)[0]

        # 显示预测结果
        st.markdown(f"<div class='prediction-result'>预测类别：{category_mapping[predicted_class]}</div>", unsafe_allow_html=True)

        # 根据预测结果生成建议
        probability = predicted_proba[predicted_class] * 100
        advice = {
                    '严重污染': f"根据我们的库，该日空气质量为严重污染。模型预测该日为严重污染的概率为 {probability:.1f}%。建议采取防护措施，减少户外活动。",
                    '重度污染': f"根据我们的库，该日空气质量为重度污染。模型预测该日为重度污染
