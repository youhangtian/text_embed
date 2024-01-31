import os
import sys
import yaml
from yacs.config import CfgNode
import logging, logging.handlers
import numpy as np 

def get_config_from_yaml(yaml_file):
    def get_node(dict_val):
        if not isinstance(dict_val, dict):
            return dict_val 
        
        res = CfgNode()
        for key, val in dict_val.items():
            res[key] = get_node(val)
        return res 

    with open(yaml_file) as f:
        cfg_dict = yaml.safe_load(f.read())
        cfg_node = get_node(cfg_dict)

    return cfg_node.clone() 


def get_logger(log_dir, log_file='log.txt'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    filename = os.path.join(log_dir, log_file)
    log_format = '%(asctime)s %(message)s'

    logger = logging.getLogger(log_file.split('.')[0])
    logger.setLevel(level=logging.INFO)

    #file_handler = logging.FileHandler(filename)
    file_handler = logging.handlers.RotatingFileHandler(filename, maxBytes=100*1024*1024, backupCount=9)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)    

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(stream_handler)

    return logger


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     lock_epochs=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep 
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(final_value, base_value, warmup_iters)

    lock_schedule = np.array([])
    lock_iters = lock_epochs * niter_per_ep 
    if lock_epochs > 0:
        lock_schedule = np.array([base_value] * lock_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, lock_schedule, schedule))
    assert len(schedule) == (epochs + lock_epochs) * niter_per_ep 
    return schedule


class AverageMeter():
    def __init__(self, len=100):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

        self.len = len
        self.arr = []

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1

        if self.len > 0:
            self.arr.append(val)
            if len(self.arr) > self.len:
                self.sum -= self.arr[0]
                self.arr.pop(0)
                self.count -= 1

        self.avg = self.sum / self.count
        