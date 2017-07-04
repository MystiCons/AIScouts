import configparser
import os

from numpy.distutils.command import config, config


def create_config(path, configs):
    if not os.path.exists(path):
        config = configparser.ConfigParser()
        config.add_section("Settings")
        for i in configs:
            config.set("Settings", i, i[0])
        with open(path, "w") as config_file:
            config.write(config_file)


def update_value(path, key, value):
    if os.path.exists(path):
        config = configparser.ConfigParser()
        config.read(path)
        config.set("Settings", key, value)
        with open(path, "w") as config_file:
            config.write(config_file)
    else:
        print("Config file not found")


def read_value(path, key):
    config = configparser.ConfigParser()
    config.read(path)
    return config.get("Settings", key)

