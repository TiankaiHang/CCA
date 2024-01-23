import os

import functools
import logging


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(
    name="editing-agents",
    output=None,
):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # plain_formatter = logging.Formatter(
    #     "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    # )
    # formatter with file name and line number
    plain_formatter = logging.Formatter('%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s') 

    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)
    else:
        file_handler = None

    return logger
