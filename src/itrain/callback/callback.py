


class CallBack(object):
    _EPOCH_INDEX = 0
    _GLOB_STEP = 0
    def __init__(self, logger=None, local_rank=-1, glob_rank=-1):
        self.logger = logger
        self.local_rank = local_rank
        self.glob_rank = glob_rank

    def set(self, logger, local_rank, glob_rank):
        self.logger = logger
        self.local_rank = local_rank
        self.glob_rank = glob_rank

    def get_epoch_index(self):
        return CallBack._EPOCH_INDEX

    def get_glob_step(self):
        return CallBack._GLOB_STEP

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_train_epoch_begin(self, **kwargs):
        pass

    def on_train_epoch_end(self, **kwargs):
        pass

    def on_validation_epoch_begin(self, **kwargs):
        pass

    def on_validation_epoch_end(self, **kwargs):
        pass

    def on_test_epoch_begin(self, **kwargs):
        pass

    def on_test_epoch_end(self, **kwargs):
        pass

    def on_train_step(self, datas, **kwargs):
        pass

    def on_validation_step(self, datas, **kwargs):
        pass

    def on_test_step(self, datas, **kwargs):
        pass


class CallBackList(CallBack):
    def __init__(self, cbs:list=[], logger=None, local_rank=-1, glob_rank=-1):
        super().__init__()
        __cbs = []
        for cb in cbs:
            cb.set(logger, local_rank, glob_rank)
            __cbs.append(cb)
        self.__cbs = __cbs

        self.reset(logger=logger, local_rank=local_rank, glob_rank=glob_rank)

    def _update_epoch_index(self, step=1):
        CallBack._EPOCH_INDEX += step

    def _update_glob_step(self, step=1):
        CallBack._GLOB_STEP += step

    def _nbacks(self):
        return len(self.__cbs)

    def reset(self, logger=None, local_rank=-1, glob_rank=-1):
        for cb in self.__cbs:
            cb.set(logger, local_rank, glob_rank)
        self.logger = logger
        self.local_rank = local_rank
        self.glob_rank = glob_rank

    def _push(self, cb):
        assert isinstance(cb, CallBack)
        cb.set(self.logger, self.local_rank, self.glob_rank)
        self.__cbs.append(cb)

    def _on_train_begin(self, **kwargs):
        flag = False
        for cb in self.__cbs:
            r = cb.on_train_begin(**kwargs)
            flag |= bool(r)
        r = self.on_train_begin(**kwargs)
        return flag | bool(r)

    def _on_train_end(self, **kwargs):
        flag = False
        for cb in self.__cbs:
            r = cb.on_train_end(**kwargs)
            flag |= bool(r)
        r = self.on_train_end(**kwargs)
        return flag | bool(r)

    def _on_train_epoch_begin(self, **kwargs):
        flag = False
        for cb in self.__cbs:
            r = cb.on_train_epoch_begin(**kwargs)
            flag |= bool(r)
        r = self.on_train_epoch_begin(**kwargs)
        return flag | bool(r)

    def _on_train_epoch_end(self, **kwargs):
        flag = False
        for cb in self.__cbs:
            r = cb.on_train_epoch_end(**kwargs)
            flag |= bool(r)
        r = self.on_train_epoch_end(**kwargs)
        return flag | bool(r)

    def _on_validation_epoch_begin(self, **kwargs):
        flag = False
        for cb in self.__cbs:
            r = cb.on_validation_epoch_begin(**kwargs)
            flag |= bool(r)
        r = self.on_validation_epoch_begin(**kwargs)
        return flag | bool(r)

    def _on_validation_epoch_end(self, **kwargs):
        flag = False
        for cb in self.__cbs:
            r = cb.on_validation_epoch_end(**kwargs)
            flag |= bool(r)
        r = self.on_validation_epoch_end(**kwargs)
        return flag | bool(r)

    def _on_test_epoch_begin(self, **kwargs):
        flag = False
        for cb in self.__cbs:
            r = cb.on_test_epoch_begin(**kwargs)
            flag |= bool(r)
        r = self.on_test_epoch_begin(**kwargs)
        return flag | bool(r)

    def _on_test_epoch_end(self, **kwargs):
        flag = False
        for cb in self.__cbs:
            r = cb.on_test_epoch_end(**kwargs)
            flag |= bool(r)
        r = self.on_test_epoch_end(**kwargs)
        return flag | bool(r)

    def _on_train_step(self, datas, **kwargs):
        flag = self.on_train_step(datas, **kwargs)
        for cb in self.__cbs:
            cb.on_train_step(datas, **kwargs)

        return flag

    def _on_validation_step(self, datas, **kwargs):
        flag = self.on_validation_step(datas, **kwargs)
        for cb in self.__cbs:
            cb.on_validation_step(datas, **kwargs)

        return flag

    def _on_test_step(self, datas, **kwargs):
        flag = self.on_test_step(datas, **kwargs)
        for cb in self.__cbs:
            cb.on_test_step(datas, **kwargs)

        return flag
