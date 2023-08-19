# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd 
import numpy as np
import pickle  #to load a saved modelimport base64  #to open .gif files in streamlit app
from xgboost import XGBRegressor
import json
import category_encoders
import scipy.stats as stats

st.header('HDB Price Prediction Application')
st.write('by Alfred Tang, Aug 2023')
st.warning('v0.2 -- Demo version. Everything is a WIP!')

wdir = './'

df = pd.read_csv(wdir + 'files/full_df_clean.csv')

# should gather this section into 1 config.py...
with open(wdir + 'files/hdb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(wdir + 'files/cat_encoder.pkl', 'rb') as f:
    enc = pickle.load(f)

with open(wdir + 'files/flat_type_ranks.json', 'rb') as f:
    flat_type_ranks = json.load(f)

with open(wdir + 'files/selection_params.json', 'rb') as f:
    selection_params = json.load(f)

# # Sanity checks for make sure all items are loaded properly    
# print(xgbr)
# print(enc)
# print(flat_type_ranks)

town = st.selectbox('Town', selection_params['town'])

def get_flat_types(df, flat_type_ranks):
    town_flat_type_nums = list(df[df['town'] == town]['flat_type_num'].unique())
    town_flat_types = [key for key, val in flat_type_ranks.items() if val in town_flat_type_nums]
    return town_flat_types
    
flat_type = st.selectbox('Type of Flat',
                         get_flat_types(df, flat_type_ranks)
                         )

flat_type_num = flat_type_ranks[flat_type]

# def get_flat_models(df, town, flat_type):
    

flat_model = st.selectbox('Flat Model', 
                          list(df[(df['town'] == town) & (df['flat_type_num'] == flat_type_num)]['flat_model'].unique())
                          )



# floor_area = st.slider('Floor Area (square metres)', 
#                         int(selection_params['floor_area_sqm'][0]), 
#                         int(selection_params['floor_area_sqm'][1])
#                         )

floor_area = st.number_input('Floor Area (square metres)', 
                             int(selection_params['floor_area_sqm'][0]), 
                             int(selection_params['floor_area_sqm'][1])
                             )

# storey = st.slider('Storey', 
#                     int(selection_params['storey'][0]), 
#                     int(selection_params['storey'][1])
#                     )

storey = st.number_input('Storey', 
                         int(selection_params['storey'][0]), 
                         int(selection_params['storey'][1])
                         )

# remaining_lease_years = st.slider('Remaining Lease (Years)', 
#                                   float(selection_params['remaining_lease_months'][0]/12.0), 
#                                   float(selection_params['remaining_lease_months'][1])/12.0
#                                    )

remaining_lease_years = st.number_input('Remaining Lease (Years)', 
                                  float(selection_params['remaining_lease_months'][0]/12.0), 
                                  float(selection_params['remaining_lease_months'][1])/12.0
                                  )

remaining_lease_months = remaining_lease_years * 12.0

# encode features into a format compatible with the model
categoricals = pd.DataFrame({'town':[town],
                             'flat_model':[flat_model]
                             })
transformed_categoricals = enc.transform(categoricals)
town_encoded = list(transformed_categoricals['town'])[0]
flat_model_encoded = list(transformed_categoricals['flat_model'])[0]

# convert inputs to array
feature_list = [town_encoded, 
                flat_model_encoded, 
                flat_type_num,
                floor_area, 
                storey, 
                remaining_lease_months]

single_pred = np.array(feature_list).reshape(1,-1)

if st.button('Predict!'):
    
    pred = model.predict(single_pred)
    st.write('''
             ## Predicted HDB Resale Price
             ''')
    st.success('S${0:,.2f}'.format(pred[0]))

st.warning('NOTE: This application is for educational and demo purposes only and does not consitute\
           financial advice or proper valuation of your Housing Development Board (HDB) flat.\
           The predictive model backing this application is a work in progress.\
           Please only use this as a reference point and seek official valuation from HDB.'
           )