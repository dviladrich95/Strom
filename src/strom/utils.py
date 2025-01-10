def call_api():
    price = 1
    return price

def get_api_key(key_path):
    with open(key_path, 'r') as file:
        api_key = file.read().strip()  # Read the file
    return api_key

def data_analysis(data,decision = True,reverse = True):
    if reverse:
        decision = not(decision)
    return decision
