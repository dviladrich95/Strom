# Function to read the API key from a text file
def get_api_key():
    with open('apiKey.txt', 'r') as file:
        api_key = file.read().strip()  # Read the file
    return api_key

# Get the API key
api_key = get_api_key()

# Print the API key (just to confirm it's working) remove this before using real API key
print(api_key)
