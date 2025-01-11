import requests
from src.strom import utils
from datetime import date

query = "https://web-api.tp.entsoe.eu/api?documentType=A44"
timeframe = "&periodStart=202501110100&periodEnd=202501120100"
domain = "&out_Domain=10YAT-APG------L&in_Domain=10YAT-APG------L"
security = "&securityToken="
apikey = utils.get_api_key('../config/apiKey.txt') #please see readme to see how to create your config folder with the API key

response = requests.get(query + timeframe + domain + security + apikey)

print(response.text)

with open("../data/" + str(date.today()) + "-API-response.json", "wb") as file:  # Use "wb" to write in binary mode
    file.write(response.content)