import logging
import os

def setup_logging(log_directory="c:/Users/User/Downloads/film/Rossmann-Pharmaceuticals-Store-Sales-Prediction/logs", log_file="eda_log.txt"):
    """Set up logging configuration with a specified directory and file."""

    # Create log directory if it doesn't exist
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Full path to the log file
    log_file_path = os.path.join(log_directory, log_file)

    # Configuring the logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),  # Write logs to specified file
            logging.StreamHandler()  # Optional: Output logs to the console
        ]
    )
    logging.info('Logging configured at %s', log_file_path)