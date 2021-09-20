import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from flask import abort

raw_input = {'person_age': 28,
  'person_income': 30000,
  'person_home_ownership': 'RENT',
  'person_emp_length': 0.0,
  'loan_intent': 'MEDICAL',
  'loan_grade': 'D',
  'loan_amnt': 11500,
  'loan_int_rate': 14.46,
  'loan_percent_income': 0.38,
  'cb_person_default_on_file': 'Y',
  'cb_person_cred_hist_length': 5}


test_input = {'person_age': 28,
  'person_income': 30000,
  'person_home_ownership': 'RENT',
  'person_emp_length': 0.0,
  'loan_intent': 'MEDICAL',
  'loan_grade': 'D',
  'loan_amnt': 11500,
  'loan_int_rate': 14.46,
  'loan_percent_income': 0.38,
  'cb_person_default_on_file': 'Y',
  'cb_person_cred_hist_length': 5,
  'person_home_ownership_OTHER': 0,
  'person_home_ownership_OWN': 0,
  'person_home_ownership_RENT': 1,
  'loan_intent_EDUCATION': 0,
  'loan_intent_HOMEIMPROVEMENT': 0,
  'loan_intent_MEDICAL': 1,
  'loan_intent_PERSONAL': 0,
  'loan_intent_VENTURE': 0,
  'loan_grade_B': 0,
  'loan_grade_C': 0,
  'loan_grade_D': 1,
  'loan_grade_E': 0,
  'loan_grade_F': 0,
  'loan_grade_G': 0,
  'cb_person_default_on_file_Y': 1,
  'person_age_WOE': -0.057,
  'loan_int_rate_WOE': 0.026,
  'loan_percent_income_WOE': 2.136,
  'person_income_WOE': 0.837,
  'loan_amnt_WOE': 0.069,
  'person_emp_length_WOE': 0.16,
  'cb_person_cred_hist_length_WOE': -0.071,
  'Risk': 0.0,
  'score_proba': 0.7462351777564744
  }



with open("[MODULE 8] Rapidminer\WOE-1.0.0.pkl", "rb") as f:
	dict_woe = pickle.load(f)

with open("[MODULE 8] Rapidminer\OHE-1.0.0.pkl", "rb") as f:
	ohe = pickle.load(f)

with open("[MODULE 8] Rapidminer\COL-NAME1.0.0.pkl", "rb") as f:
	cat_columns = pickle.load(f)

with open("[MODULE 8] Rapidminer\Mei-LR-1.0.0.pkl", "rb") as f:
	modelLR = pickle.load(f)

'''
  cat_column : 
  'person_home_ownership',
 'loan_intent',
 'loan_grade',
 'cb_person_default_on_file'
'''

def preprocess(data):
  
  data = pd.DataFrame([data])
  
  for feature, woe_info in dict_woe.items():
    data[f'{feature}_WOE'] = pd.cut(data[feature], bins=woe_info['binning'], labels=woe_info['labels'])
    data[f'{feature}_WOE'] = data[f'{feature}_WOE'].values.add_categories('Nan').fillna('Nan') 
    data[f'{feature}_WOE'] = data[f'{feature}_WOE'].replace('Nan', 0) # ubah nilai nan menjadi 0
    data[f'{feature}_WOE'] = data[f'{feature}_WOE'].astype(float)
	
  data_ohe = ohe.transform(data[cat_columns]).toarray()
  
  column_name = ohe.get_feature_names(cat_columns)
  
  data_one_hot_encoded = pd.DataFrame(data_ohe, columns=column_name, index=data[cat_columns].index).astype(int)
  
  data = pd.concat([data, data_one_hot_encoded], axis=1).reset_index(drop=True)
  
  return data


def pred(data):
  model = modelLR
  all_features = ['person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT', 'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 'loan_grade_E', 'loan_grade_F', 'loan_grade_G', 'cb_person_default_on_file_Y', 'person_age_WOE', 'loan_int_rate_WOE', 'loan_percent_income_WOE', 'person_income_WOE', 'loan_amnt_WOE', 'person_emp_length_WOE', 'cb_person_cred_hist_length_WOE']
  
  print('cek')
  print(data.shape)
  pred_proba = model.predict_proba(data[all_features])[:,1]
  threshold = 0.189765
  prediction = (pred_proba > threshold).astype(int)
  return { "data": [ { "probability to default ": float(pred_proba[0]), "prediction": int(prediction[0])} ] }

def prediction(raw_input):
  # data = formatting_data(raw_input)
  data = preprocess(raw_input)
  prediction = pred(data)
  return prediction


if __name__ == "__main__":
	result = pred(pd.DataFrame([test_input]))
	print(result)