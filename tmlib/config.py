import configparser
import logging
import os
CONFIG_FILE = 'config.ini'


def get_config(section, key):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(dir_path, CONFIG_FILE)
    print config_file
    config = configparser.ConfigParser()
    try:
        config.read(config_file)
        value = config[section][key]
        return value
    except ValueError:
        logging.error('Section %s or key %s not in config.ini file', section, key)


def edit_config(section, key, value):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(dir_path, CONFIG_FILE)
    config = configparser.ConfigParser()
    try:
        config.read(config_file)
        value = config[section][key] = value
        with open(config_file, 'w') as configfile:
            config.write(configfile)
    except Exception as ve:
        logging.error(ve)