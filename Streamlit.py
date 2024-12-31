import math
import pandas as pd
import pandas
import numpy as np
import streamlit as st
# Information
st.sidebar.markdown("## Authors' information")
st.sidebar.markdown("Authors: Thai-Hoang Nguyen,Van-Thanh Pham, Quang-Viet Vu, and Viet-Ngoc Pham")
st.sidebar.caption("Faculty of Civil Engineering, Thuyloi University, Vietnam")
st.sidebar.caption("Emails: phamthanhwru@gmail.com (V-T Pham)")

#Tên bài báo
st.title ("Hybrid machine learning with FHO algorithm and WERCS method for predicting fire resistance of timber columns") 

# Chèn sơ đồ nghiên cứu
# st.header("1. Layout of this study")
# check1=st.checkbox('1.1 Display layout of this investigation')
# if check1:
#    st.image("Fig. 1.jpg", caption="Layout of this study")
# # Hiển thị dữ liệu
# st.header("2. Dataset")
# check2=st.checkbox('2.1 Display dataset')
# if check2:
#    Database="Dataset.csv"
#    df = pandas.read_csv(Database)
#    df.head()
#    st.write(df)
# st.header("3. Modeling approach")
# check3=st.checkbox('3.1 Display structure of Random Forest model')
# if check3:
#    st.image("Fig2.jpg", caption="Overview on structure of Random Forest model") 
	
#Make a prediction
st.header("Predicting the fire resistance of timber columns")
st.subheader("Input variables")
col1, col2, col3, col4 =st.columns(4)
with col1:
   X1 = st.slider("D (mm)", 75.00, 400.00)
   X2 = st.slider("W (mm)", 75.00, 400.00)
	
with col2:	

   X3= st.slider("Density (kg/m3)", 310.00, 590.00)
   X4 = st.slider("Compre. stre. (MPa)", 7.90, 62.00)

with col3:		
   X5 = st.slider("E (MPa)", 6895.00, 18616.00)
   X6 = st.slider("L (mm)", 1800.00, 3658.00)
   	
	
with col4:	
   X7 = st.slider("C (kN)", 106.00, 5592.00)
   X8 = st.slider("P (%)", 9.10, 60.00)




from sklearn.model_selection import train_test_split, KFold
#from sklearn.ensemble import GradientBoostingRegressor
#import catboost as cb
from catboost import CatBoostRegressor

#Model

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df = pd.read_csv('Timber_Thanh_WERCS.csv')
X_ori = df[['D','W','den','fc','E','L','C','P']].values
y = df['R'].values

feature_name = ['D','W',r'$\rho$',r'$f_{c}$','E','L','C','P']

X = X_ori
X_train=X
y_train=y

n_estimators = 180
learning_rate = 0.225
depth = 6
l2_leaf_reg = 8
        
cat_clf_n = CatBoostRegressor(n_estimators =n_estimators,learning_rate = learning_rate, depth=depth,l2_leaf_reg=l2_leaf_reg, random_state=42)
cat_clf_n.fit(X_train, y_train)


Inputdata = [X1, X2, X3, X4, X5, X6, X7,X8]


from numpy import asarray
Newdata1 = asarray([Inputdata])
print(Newdata1)
Newdata=scaler.transform(Newdata1)

fc_pred2 = cat_clf_n.predict(Newdata)

st.subheader("Output variable")
if st.button("Predict"):
    import streamlit as st
    import time
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
       time.sleep(0.01)
       my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    st.success(f"Your predicted fire resistance (minutes) obtained from FHO-CGB model is: {(fc_pred2)}")

