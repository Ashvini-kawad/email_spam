import os
import logging


def log_function(file_name):
    try:
        # Ensure the "logs" directory exists
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)

        # Logging configuration
        logger = logging.getLogger(file_name)
        logger.setLevel(logging.DEBUG)

        # Avoid adding duplicate handlers if function is called multiple times
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)

            # File handler
            log_file_path = os.path.join(log_dir, f"{file_name}.log")
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.DEBUG)

            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            # Add handlers
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger

    except Exception as e:
        print("Error in log_function:", e)
        return None
