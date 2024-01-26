import requests
import json

data_0 = {
    "age": 18,
    "workclass": "Local_gov",
    "fnlgt": 1.794650e05,
    "education": "10th",
    "education_num": 8,
    "marital_status": "Never_married",
    "occupation": "Farming_fishing",
    "relationship": "Not_in_family",
    "race": "Black",
    "sex": "Female",
    "capital_gain": 100,
    "capital_loss": 50,
    "hours_per_week": 0,
    "native_country": "Mexico",
}

data_1 = {
    "age": 50,
    "workclass": "Private",
    "fnlgt": 1.880050e05,
    "education": "Masters",
    "education_num": 12,
    "marital_status": "Married_civ_spouse",
    "occupation": "Exec_managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 40174,
    "capital_loss": 195,
    "hours_per_week": 50,
    "native_country": "United_States",
}

def live_api_call(data):
    headers = {'Content-Type': 'application/json'}
    url = 'https://cicd-api-cloud.onrender.com/predict'
    print('Posting prediction to: ', url)
    response = requests.post(url, data=json.dumps(data), headers=headers)
    print('Status code: ', response.status_code)
    print('Prediction: ', response.json()['prediction'])

if __name__ == '__main__':
    live_api_call(data_0)
    live_api_call(data_1)
