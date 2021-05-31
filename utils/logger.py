import logging
import os


def get_logger(command_argv, log_file_path=None):
    """add log
    Args:
        command_argv (list): sys.argv
        log_file_path (str): output path of log file, do not save log file when it's not specified

    Returns:
        logging.logger: logger
    """
    program = os.path.basename(command_argv[0])
    logger = logging.getLogger(program)

    # 设置输出的日志级别
    logger.setLevel(level=logging.INFO)
    # 设置日志输出内容：时间%Y-%m-%d %H:%M:%S，日志级别INFO,WARNING,ERROR，日志信息
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    if log_file_path is None:
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        logger.addHandler(chlr)

    if log_file_path:
        # 设置日志输出文件
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        fhlr = logging.FileHandler(log_file_path)
        fhlr.setFormatter(formatter)
        logger.addHandler(fhlr)

    # 输出日志信息内容
    logger.info("running %s" % ' '.join(command_argv))

    return logger
