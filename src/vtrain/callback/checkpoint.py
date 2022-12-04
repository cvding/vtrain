import os
import numpy as np
from .callback import CallBack
from ..utils import FIFO, save_spec_wgts


class CheckPoint(CallBack):
    def __init__(self, save_each_iter=None, save_maxs=None, watch_value: str = None, best_type='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        best_func = {'max': max, 'min': min}
        best_base = {'max': -np.inf, 'min': np.inf}

        self.save_each_iter = save_each_iter

        self.qsize = save_maxs or 1000
        self.save_queue = FIFO(maxsize=self.qsize)
        self.watch_value = watch_value
        self.best_base = best_base[best_type]
        self.best_func = best_func[best_type]

    def is_best(self, logbar, **kwargs):
        conf = kwargs.get('cconf', None)
        if conf['mode'] != 'train' or self.watch_value is None:
            return False

        try:
            val = logbar.get(self.watch_value)
        except KeyError:
            return False

        bval = self.best_func(self.best_base, val)
        if bval != val:
            self.best_base = bval
            return True

        return False

    def on_train_step(self, datas, **kwargs):
        conf = kwargs.get('cconf', None)
        save_each_iter = self.save_each_iter or kwargs.get('iters_per_epoch', np.inf)

        if self.glob_rank <= 0:
            glob_step = self.get_glob_step()
            epoch_index = self.get_epoch_index()

            if (glob_step + 1) % save_each_iter == 0:
                nets = kwargs.get("nets", None)
                paths = kwargs.get('paths', None)

                job_name = conf['job_name']
                model_path = os.path.join(paths['model'], job_name + '_%d.pth' % (epoch_index))
                save_spec_wgts(model_path, nets, glob_step, epoch_index)
                self.logger.info('Model :[%3d]\t%8d\t|saved|\t%s' % (epoch_index, glob_step, model_path))
                jtem = self.save_queue.put(model_path)

                if jtem is not None:
                    os.remove(jtem)
                    self.logger.info('Model :[%3d]\t|removed|\t%s' % (epoch_index, jtem))

    def on_validation_epoch_end(self, **kwargs):
        progress = kwargs.get('progress', None)

        if self.glob_rank <= 0:
            if self.is_best(progress, **kwargs):
                nets = kwargs.get("_nets", None)
                paths = kwargs.get('paths', None)
                conf = kwargs.get('cconf', None)
                glob_step = self.get_glob_step()
                epoch_index = self.get_epoch_index()

                job_name = conf['job_name']
                model_path = os.path.join(paths['model'], job_name + '.epoch_best.pth')
                save_spec_wgts(model_path, nets, glob_step, epoch_index)
                self.logger.info('Model :[%3d]\t%f\t|best|\t%s' % (epoch_index, self.best_base, model_path))
