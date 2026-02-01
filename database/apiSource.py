import requests
from dotenv import load_dotenv
import os

load_dotenv()

api_base = os.getenv('base_url')

def api_call(endpoint, filename):
    try: 
        api = f'{api_base}{endpoint}'
        print(api)

        response = requests.get(api)

        if response.status_code == 200:
            data = response.content

            file_path = os.path.join("dataset", filename)

            with open(file_path, "wb") as file:
                file.write(data)
        else: 
            print("We are not able to call the endpoint")
    except Exception as e:
        print(str(e))


api_data = {
    'admissions/csv': 'main_admissions.csv',
    'daily-metrics/csv': 'main_daily_metrics.csv'
}

for endpoint, filename in api_data.items():
    api_call(endpoint, filename)