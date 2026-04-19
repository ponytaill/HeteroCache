import logging
import os
import sys

# Remove explicit dependency on global variable _LOGGER, rely on logging module's internal singleton management

def create_log(args):
    """
    Configure the logger named "logger".
    """

    logger = logging.getLogger("logger")
    

    if logger.handlers:
        return

    logger.setLevel(logging.INFO)
    

    log_dir = "./results/latency_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    method_key = args.method.lower()
    method_label = 'HeteroCache' if method_key in ['heterocache'] else args.method
    if method_key in ['heterocache']:
        log_filename = os.path.join(log_dir, f"{method_label}_c{args.num_clusters}_d{args.decode_step}_{'real' if args.real_offload else 'sim'}.log")
    else:
        log_filename = os.path.join(log_dir, f"{method_label}.log")


    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
  
    logger.info(f"Log initialized. Saving to: {log_filename}")
    logger.info(f"Args: {args}")

def get_logger():
    """
    Safely retrieve the logger instance.
    """
    return logging.getLogger("logger")