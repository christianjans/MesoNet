import argparse
import yaml


def config_to_namespace(config: str) -> argparse.Namespace:
    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)
