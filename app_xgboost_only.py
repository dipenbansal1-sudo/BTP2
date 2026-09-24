import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title='Nonwoven Absorption Predictor', page_icon='🧪', layout='centered')

ROWS = [
    ('A1',0.0392,5.80,0.0097,12.91,0.065),('A2',0.0392,5.72,0.0196,6.96,0.027),
    ('A3',0.0392,6.56,0.0242,5.42,0.032),('A4',0.0392,6.61,0.0315,4.54,0.028),
    ('A5',0.0392,6.41,0.0401,4.78,0.019),('B1',0.0248,5.80,0.0097,28.55,0.078),
    ('B2',0.0248,5.56,0.0195,21.24,0.052),('B3',0.0248,5.60,0.0303,18.45,0.044),
    ('B4',0.0248,5.82,0.0388,16.04,0.035),('B5',0.0248,5.67,0.0511,14.40,0.033),
    ('C1',0.0124,5.50,0.0097,29.76,0.198),('C2',0.0124,4.83,0.0195,26.86,0.099),
    ('C3',0.0124,5.55,0.0235,21.99,0.064),('C4',0.0124,5.55,0.0303,19.35,0.079),
    ('C5',0.0124,5.45,0.0359,17.11,0.081)
]

DF = pd.DataFrame(ROWS, columns=['code','d_f_mm','thickness_mm','mu','capacity_gg','rate_ggs'])
FEATURES = ['d_f_mm', 'mu']

@st.cache_resource
def train_xgboost(target_col):
    X = DF[FEATURES]
    y = DF[target_col].values
    use_log = target_col == 'rate_ggs'
    y_train = np.log(y) if use_log else y

    model = XGBRegressor(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1,
        objective='reg:squarederror',
        random_state=42,
        verbosity=0
    )
    model.fit(X, y_train)
    return model, use_log

st.title('🧪 Nonwoven Absorption Predictor')
st.caption('XGBoost model using fibre diameter and packing density')

target_col = st.radio(
    'What do you want to predict?',
    ['capacity_gg', 'rate_ggs'],
    format_func=lambda x: 'Absorption capacity (g/g)' if x == 'capacity_gg' else 'Absorption rate (g/g·s)'
)

model, use_log = train_xgboost(target_col)

st.subheader('Enter input values')
col1, col2 = st.columns(2)
with col1:
    fibre_diameter = st.number_input('Fibre diameter (mm)', min_value=0.001, max_value=0.100, value=0.0248, step=0.0001, format='%.4f')
with col2:
    packing_density = st.number_input('Packing density (μ)', min_value=0.001, max_value=0.100, value=0.0303, step=0.0001, format='%.4f')

if st.button('Predict', type='primary', use_container_width=True):
    input_data = pd.DataFrame([[fibre_diameter, packing_density]], columns=FEATURES)
    prediction = float(model.predict(input_data)[0])
    if use_log:
        prediction = float(np.exp(prediction))

    unit = 'g/g' if target_col == 'capacity_gg' else 'g/g·s'
    st.success(f'Predicted value: {prediction:.5f} {unit}')

st.divider()
st.subheader('Training dataset')
st.dataframe(DF, use_container_width=True, hide_index=True)
st.caption('The model is trained on all 15 observations. It does not automatically learn from new data.')
