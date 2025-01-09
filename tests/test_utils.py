from src.strom import utils


def test_get_api_key():
    test_key_path = './test_apiKey.txt'
    api_key = utils.get_api_key(test_key_path)
    assert api_key == 'test123'