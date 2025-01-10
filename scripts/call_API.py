import requests

response = requests.get("https://example.com")
print(response.text)

with open("output_file.txt", "wb") as file:  # Use "wb" to write in binary mode
    file.write(response.content)