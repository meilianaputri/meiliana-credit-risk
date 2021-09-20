# meiliana-credit-risk

## Dataset
This modelling is using credit_risk_dataset.csv which is contain 12 variabels. The main goal is to predict that someone will default or not default based on their information (such as age, income) that contain in dataset. This dataset contain the following varible:

1. person_age 
2. person_income 
3. person_home_ownership (RENT, OWN, MORTGAGE, OTHER)
4. person_emp_length 
5. loan_intent (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION)
6. loan_grade (A, B, C, D, E, F, G)
7. loan_amnt 
8. loan_int_rate
9. loan_status
10. loan_percent_income
11. cb_person_default_on_file (Y/N)
12. cb_erson_credit_hist_length

Numerical variable:
['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

Categorical variable:
['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']

Outliers and Missing Value:
- All numerical varible have outliers
- Person_emp_length and loan_int_rate have missing value

## Handling Missing Value and Handlig Outlier
Both of handling missing value and handling oulier using woe method.

## Modelling
This step using K-vold stratified because of imbalanced data, and then do hyperparameter tunning to find best parameter.
There are to model that used in this modelling step, logistic regression and random forest. 

Performance for Random Forest:
![image](https://user-images.githubusercontent.com/76585709/133971034-aa4ce2e8-d854-48d6-8d00-c9535572110f.png)

Performance for Logistic Regression:
![image](https://user-images.githubusercontent.com/76585709/133971106-03f890d9-9bb9-4cd1-8970-7c11cfdb4f7b.png)

Model uses random forest give better performance than logistic regression, so better to choose random forest modelling than logistic regression. But if the goal of modelling is to know how many times the effect of the variable on the target compared to other variables (using coefisient regression), better use logistic regression.

## Input Format
{'person_age': 28,
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
  
  expected result :
  - probability = 0.746
  - prediction = 1
 
 ## Output Format
 { "model": "RF-WOE-meiliana", "prediction": 0, "version": "1.0.0" }
 
The output only prediction just to simplified the conclusion.

## HTTP Method Rule
- You need to install all of the packages by call the requirements.txt on your command prompt
- Open app_final.py in the main repository and run it
- Open Postman apps to overview the final results
- Set the URL http://127.0.0.1:5000/predict-credit-risk using 'POST' method and put the input in the body section
- Change the content type to JSON
- note: make sure the input is in JSON format (use {} and "...")
