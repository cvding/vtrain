import numpy as np
from .callback import CallBack

class EarlyStopping(CallBack):
    def __init__(self,
                 min_delta=0,
                 patience=0,
                 mode='min',
                 baseline=None, 
                 watch_value:str=None):
        super(EarlyStopping, self).__init__()

        self.watch_value = watch_value
        self.patience = patience
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0

        if mode not in ['min', 'max']:
            raise ValueError('not support mode: ', mode) 
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
    
    def on_validation_epoch_end(self, **kwargs):
        progress = kwargs.get('progress', None) 
        if self.glob_rank<= 0:
            try:
                val = progress.get(self.watch_value)
            except:
                return False
            
            flag = self.__step(val)

            return flag
        
        return False

    def reset(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None

    def __step(self, value):
        stop_training = False
        current = value

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                stop_training = True
        return stop_training