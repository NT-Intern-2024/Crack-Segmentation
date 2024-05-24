import yaml
from .constants import CONFIG_FILE
from .file_utils import check_path_exists, load_file


class Config:
    def __init__(self, config_file=CONFIG_FILE):
        self.config_file = config_file
        self.config = self.__load_config()

        self.path = self.get('dataset', 'path')
        self.path_to_train = self.get('dataset', 'path_to_train')
        self.path_to_validated = self.get('dataset', 'path_to_validated')
        self.path_to_test = self.get('dataset', 'path_to_test')

    def __load_config(self):
        check_path_exists(self.config_file)
        config = load_file(self.config_file)
        return config

    def get(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)

    def set(self, section, key, value):
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value

    def save_config(self):
        with open(self.config_file, 'w') as file:
            yaml.dump(self.config, file)

    def __str__(self):
        return str(self.config)
