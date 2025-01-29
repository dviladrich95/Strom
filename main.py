from strom import utils

def print_price():
    price = utils.call_api()
    print(f'Test price of {price}')

if __name__ == '__main__':
    print_price()

