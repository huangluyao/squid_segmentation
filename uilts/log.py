import os, time
import logging, sys


def get_logger(logdir):

    # create log directory
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # create log file
    logname = f'fun-{time.strftime("%Y-%m-%d-%H-%M")}.log'
    log_file = os.path.join(logdir, logname)

    # create log
    logger = logging.getLogger('trian')
    logger.setLevel(logging.INFO)

    # set log format
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # StreamHandler: Output the log to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler: Output the log to log_file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger