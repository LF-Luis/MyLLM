import os
import logging
from datetime import datetime
from src.utils.root import create_temp_data_file


def setup_logging(log_file='project_logs.log'):
    # Retrieve the logging level from the environment variable
    log_level = os.getenv('LOG_LEVEL', 'DEBUG').upper()  # Default to 'DEBUG' if not set
    
    # Verify if log level is valid, else fallback to DEBUG and log an error
    level = getattr(logging, log_level, logging.DEBUG)
    if level == logging.DEBUG and log_level != 'DEBUG':
        logging.basicConfig(level=logging.DEBUG)
        logging.error(f'Invalid LOG_LEVEL "{log_level}" in environment; defaulting to DEBUG.')

    # Create the log directory we will write checkpoints to and log to
    current_time = datetime.now().astimezone()
    current_time = current_time.strftime('%Y_%m_%d-%H_%M_%Z')
    log_file = create_temp_data_file(path='logs/log_' + current_time, file_name='log.txt')

    # Configure the root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),     # Write to log file
            logging.StreamHandler()            # Also output to console
        ]
    )
