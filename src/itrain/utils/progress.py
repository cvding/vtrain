from collections import OrderedDict
from copy import deepcopy as dcopy

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', hall=True):
        self.name = name
        self.fmt = fmt
        self.reset()
        if hall:
            self.fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        else:
            self.fmtstr = '{name} {avg' + self.fmt + '}'

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def get(self):
        return self.avg if self.count else None
    
    def last(self):
        return self.val if self.count else None
    
    def empty(self):
        return self.count == 0

    def __str__(self):
        if self.count != 0:
            return self.fmtstr.format(**self.__dict__)
        else:
            return ""


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.meters = OrderedDict()
        self.add(meters)
        self.set(num_batches=num_batches, prefix=prefix)
    
    def reset(self):
        for key in self.meters.keys():
            self.meters[key].reset()
    
    def add(self, meters):
        if isinstance(meters, AverageMeter) and \
            meters.name not in self.meters.keys():
            self.meters[meters.name] = dcopy(meters)
        elif isinstance(meters, list):
            for meter in meters:
                if isinstance(meter, AverageMeter) and \
                    meter.name not in self.meters.keys():
                    self.meters[meter.name] = dcopy(meter) 

    def set(self, num_batches=-1, prefix=""):
        if num_batches > 0:
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        
        if prefix != "":
            self.prefix = prefix
    
    def update(self, name, value, n=1):
        self.meters[name].update(value, n)
    
    def get(self, name):
        return self.meters[name].get()

    def display(self, batch_index):
        entries = [self.prefix + self.batch_fmtstr.format(batch_index)]
        entries += [str(meter) for meter in self.meters.values()]
        show = '\t'.join(entries)
        return show

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
    def __str__(self):
        entries = [self.prefix]
        for meter in self.meters.values():
            if str(meter) != '':
                entries.append(str(meter))
        show = '\t'.join(entries)
        return show

    def update_meters(self, ins_src):
        if isinstance(ins_src, ProgressMeter):
            for key in ins_src.meters.keys():
                if not ins_src.meters[key].empty():
                    self.meters[key] = dcopy(self.meters[key])