import yaml

def load_config():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)
