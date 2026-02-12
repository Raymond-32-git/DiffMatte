def setup_logger(*args, **kwargs):
    import logging
    logger = logging.getLogger("detectron2")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[MockD2] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
