import yaml

class Configuration(object):
    def __init__(self):
        pass
    
    @staticmethod
    def get_config(file_path: str) -> dict:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.SafeLoader)