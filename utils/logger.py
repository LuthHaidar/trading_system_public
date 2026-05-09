import logging
import os


DEFAULT_FORMAT = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
DEFAULT_DATEFMT = '%Y-%m-%d %H:%M:%S'


def _create_formatter() -> logging.Formatter:
    return logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)


def _build_handlers(
    log_file: str = None,
    console_level: int = logging.WARNING,
    file_level: int = logging.INFO,
    file_mode: str = 'a',
):
    handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(_create_formatter())
    handlers.append(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(_create_formatter())
        handlers.append(file_handler)

    return handlers


def setup_logger(
    name: str,
    log_file: str = None,
    level=logging.INFO,
    console_level: int = logging.WARNING,
    file_mode: str = 'a',
):
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        console_level: Console handler logging level
        file_mode: File mode for log file ('a' append, 'w' overwrite)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Reset root handlers so repeated setup calls do not duplicate output.
    existing_count = len(root_logger.handlers)
    if existing_count > 0:
        root_logger.warning(
            "setup_logger reconfiguring root logger with %d existing handler(s); "
            "discarding previous handler configuration",
            existing_count,
        )
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    for handler in _build_handlers(
        log_file,
        console_level=console_level,
        file_level=level,
        file_mode=file_mode,
    ):
        root_logger.addHandler(handler)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True

    return logger


def get_logger(name: str):
    """Get existing logger or create new one with default settings"""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(_build_handlers(console_level=logging.WARNING, file_level=logging.INFO)[0])

    logger = logging.getLogger(name)
    logger.propagate = True
    return logger
