import argparse
from ultralytics import YOLO, settings
from clearml import Task

from src.config import Config
import logging
from src.constants import EXPERIMENTS_PATH


def arg_parse() -> argparse.Namespace:
    """Parses CLI input

    Returns:
        argparse.Namespace: parse result
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')

    return parser.parse_args()


def train(config: Config):
    settings.update({"runs_dir": EXPERIMENTS_PATH, "tensorboard": True, "clearml": True})
    task = Task.init(
        project_name=config.project_name,
        task_name=f'{config.experiment_name}',
        auto_connect_frameworks=True,
    )

    model = YOLO(f"{config.model}.pt")
    model.train(data=config.config, epochs=config.n_epochs)
    model.val()
    model.export(format="onnx")


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=logging.INFO)

    config = Config.from_yaml(args.config_file)

    train(config)

