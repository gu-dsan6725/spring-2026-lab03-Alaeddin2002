import json
import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
    )


def log_dict(data: dict):
    logging.info(json.dumps(data, indent=2, default=str))
