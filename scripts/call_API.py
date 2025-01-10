import requests
from src.strom import utils


apikey = utils.get_api_key('../config/apiKey.txt')
query = "https://web-api.tp.entsoe.eu/api?documentType=A65&processType=A16&outBiddingZone_Domain=10YCZ-CEPS-----N&periodStart=202303030000&periodEnd=202303060000&securityToken="

response = requests.get(query + apikey)

print(response.text)

with open("../data/output_file.txt", "wb") as file:  # Use "wb" to write in binary mode
    file.write(response.content)