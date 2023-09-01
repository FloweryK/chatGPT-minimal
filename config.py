import json


class Config:
    def __init__(self, path_config):
        with open(path_config, 'r') as j:
            config = json.load(j)
        
        for key, value in config.items():
            setattr(self, key, value)