import logging

logging.basicConfig(level=logging.DEBUG)

def log_and_import():
    logging.debug('Before importing encoding module')
    from encoding.nn import LabelSmoothing, NLLMultiLabelSmooth
    from encoding.utils import accuracy, AverageMeter, MixUpWrapper, LR_Scheduler, torch_dist_sum
    logging.debug('After importing encoding module')

log_and_import()
