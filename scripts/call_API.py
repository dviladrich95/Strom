import requests
from strom import utils
from datetime import date
import os
import xml.etree.ElementTree as ET
import pandas as pd

query = "https://web-api.tp.entsoe.eu/api?documentType=A44"
timeframe = "&periodStart=202501110100&periodEnd=202501120100"
domain = "&out_Domain=10YAT-APG------L&in_Domain=10YAT-APG------L"
security = "&securityToken="
os.chdir(utils.find_root_dir())
apikey = utils.get_api_key('./config/apiKey.txt') #please see readme to see how to create your config folder with the API key

response = requests.get(query + timeframe + domain + security + apikey)

#print(response.text)

with open("./data/" + str(date.today()) + "-API-response.txt", "wb") as file:  # Use "wb" to write in binary mode
    file.write(response.content)

#root = ET.fromstring(response.text)

# Parse the XML content directly
root = ET.fromstring(response.content)

# Extract the time interval information
time_interval = root.find(".//ns0:timeInterval", namespaces={'ns0': root.tag.split('}')[0].strip('{')})

# Extract the start and end timestamps
start_time = time_interval.find('ns0:start', namespaces={'ns0': root.tag.split('}')[0].strip('{')}).text
end_time = time_interval.find('ns0:end', namespaces={'ns0': root.tag.split('}')[0].strip('{')}).text

# Extract all 'Point' elements (ignoring namespaces in the tags)
points = root.findall(".//ns0:Point", namespaces={'ns0': root.tag.split('}')[0].strip('{')})

# List to hold the data for DataFrame
data = []

# Loop through each 'Point' and extract the 'position', 'price.amount'
for point in points:
    position = point.find('ns0:position', namespaces={'ns0': root.tag.split('}')[0].strip('{')}).text
    price = point.find('ns0:price.amount', namespaces={'ns0': root.tag.split('}')[0].strip('{')}).text
    data.append({'position': int(position), 'price.amount': float(price), 'start_time': start_time, 'end_time': end_time})

# Create a DataFrame from the data
df = pd.DataFrame(data)

print(df.head())