import requests
from src.strom import utils


apikey = utils.get_api_key('../config/apiKey.txt') #please see readme to see how to create your config folder with the API key
query = "https://web-api.tp.entsoe.eu/api?documentType=A44&periodStart=202501102200&periodEnd=202501112200&out_Domain=10YAT-APG------L&in_Domain=10YAT-APG------L&securityToken="

response = requests.get(query + apikey)

print(response.text)

with open("../data/output_file.txt", "wb") as file:  # Use "wb" to write in binary mode
    file.write(response.content)