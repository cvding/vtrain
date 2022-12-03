import os

os.environ['OPENBLAS_NUM_THREADS'] = '2'
import argparse
import random
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from copy import deepcopy as dcopy
import torch.multiprocessing as mp

from .. import YamlConf
from ..utils import AverageMeter, ProgressMeter, load_spec_wgts, model_parameters, find_free_port, ctip, finetune_layers
from ..callback.callback import CallBackList
from ..callback.checkpoint import CheckPoint
from torch.utils.tensorboard import SummaryWriter
from loguru import logger


class EngineFileSystem:
    def __init__(self) -> None:
        self.init = False
        self.args = self.__args()
        self.capp = YamlConf(self.args.conf)

    def deploy(self):
        cconf = self.capp['common']
        paths = None
        job_name = '{}.{}'.format(cconf.job_name, cconf.backend)

        def create_vesion(path):
            idx = -1
            if not os.path.exists(path):
                return "version_1"
            for file_name in os.listdir(path):
                if 'version' in file_name:
                    _, vid = file_name.split('_')
                    vid = int(vid)
                    if vid > idx:
                        idx = vid
            if idx < 0:
                idx = 0

            return "version_%d" % (idx + 1)


        # default version "version_1"
        root = os.path.join(cconf.save_root, job_name)
        version = create_vesion(root) 
        save_root = os.path.join(root, version)

        paths = {
            "root": save_root,
            "summary": os.path.join(save_root, "summary"),
            "model": os.path.join(save_root, 'models')
        }

        ctip('yellow', '|wait for director builder...', show=True)
        if self.init:
            for name in paths.keys():
                os.makedirs(paths[name])
            ctip('green', '|director build done.', show=True)
            ctip("cyan", "|model save root: %s" % (paths['root']), show=True)

        while True:
            if os.path.exists(save_root):
                break

        self.capp.conf['common']['paths'] = paths
        self.capp.conf['common']['local_rank'] = self.args.local_rank
        self.capp.conf['common']['glob_rank'] = self.args.glob_rank
        self.capp.conf['common']['gpus'] = self.args.gpu_idx
        self.capp.conf['common']['num_machine'] = self.args.num_machine
        self.capp.conf['common']['version'] = version.split('_')[-1]

        if self.capp.conf['common']['backend'] == 'ddp':
            self.capp.conf['common']['ddp']['dist_url'] = "%s:%d" % (self.args.master_addr, self.args.master_port)
            self.capp.conf['common']['ddp']['nprocs'] = self.args.nprocs
            ctip('green', 'bold', "master address: %s" % (self.capp.conf['common']['ddp']['dist_url']), show=True)
        return self.capp

    @staticmethod
    def parse_local(gpu_idxs):
        gpus = {}
        is_multi = isinstance(gpu_idxs[0], list)
        is_each = 1 if not is_multi else len(gpu_idxs[0])
        for mdx, gpu in enumerate(gpu_idxs):
            flag = isinstance(gpu, list)
            if flag:
                gpus[mdx] = gpu
                assert is_each == len(gpu), "Error: GPU each machine must be equal!"
            assert is_multi == flag, "Error: GPU Index set Wrong!"

        if not is_multi:
            gpus[0] = gpu_idxs

        return len(gpus), gpus

    def __args(self, desc='vengine one worker...'):
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument('-c', '--conf', default='./conf.yaml', help='input your config path')

        parser.add_argument('-r', '--local_rank', default=-1, type=int, help='local process idx for distributed training|-1')
        parser.add_argument('-g', '--glob_rank', default=-1, type=int, help='global process idx for distributed training|-1')
        parser.add_argument('-a', '--master_addr', default='127.0.0.1', help='input your master address for distributed training')
        parser.add_argument('-p', '--master_port', type=int, default=-1, help='input your master port for distributed training|-1')

        args = parser.parse_args()
        conf = YamlConf.load(args.conf)

        assert len(conf['common']['gpu_idx']) > 0, 'please config the [common][gpu_idx] parameter'

        num_machine, gpus = self.parse_local(conf['common']['gpu_idx'])
        if num_machine <= 1:
            args.glob_rank = args.local_rank

        if conf['common']['backend'] == 'ddp':
            rank = os.getenv('RANK', None) or args.glob_rank
            local_rank = os.getenv('LOCAL_RANK', None) or args.local_rank
            master_port = os.getenv('MASTER_PORT', None) or args.master_port
            master_addr = os.getenv("MASTER_ADDR", None) or args.master_addr

            rank = int(rank)
            local_rank = int(local_rank)
            master_port = int(master_port)

            if master_port < 0:
                master_port = find_free_port()

            if conf['vtraining']['enable']:
                assert rank >= 0 and local_rank >= 0 and master_port > 0 and master_addr

            args.glob_rank = rank if num_machine > 1 else local_rank          # 全局进程索引
            args.local_rank = local_rank    # 本地进程索引
            args.master_addr = "tcp://" + master_addr
            args.master_port = master_port

            #end_gpu_idx = num_machine*len(gpus[0]) - 1
            end_gpu_idx = len(gpus[0]) - 1
            if conf['vtraining']['enable']:
                self.init = args.glob_rank == end_gpu_idx # vtraining 脚本要求
            else:
                self.init = args.glob_rank <= 0
        else:
            args.glob_rank = -1
            self.init = True

        pidx = 0 if args.glob_rank < 0 else args.glob_rank // len(gpus[0])
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpus[pidx]])
        args.num_machine = num_machine
        args.nprocs = num_machine * len(gpus[pidx])
        args.conf = conf
        args.gpu_idx = [i for i in range(len(gpus[pidx]))]
        assert len(args.gpu_idx) <= torch.cuda.device_count(), 'Error: GPU Resource is not Right!'

        return args

class Engine(CallBackList):
    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.local_rank <= 0:
            self.logger.info(str(conf))
        self.conf = YamlConf(conf)

        flag = self.initialize()

        if flag is not None:
            tips = '1. make sure your self.initialize() function is right.\n \
                    2. make sure to use vengine.init in self.initialize() function.'
            raise ValueError(tips)
        if self.cconf.mode == 'train':
            self._push(CheckPoint(logger=self.logger, **self.cconf.checkpoint))

    def initialize(self):
        self.cconf = self.conf['common']

        self._nets = OrderedDict()

        self.__init_dist()
        self.__init_seed()

        self.__progress = ProgressMeter(0, meters=[AverageMeter('Loss', ':.6f', False)])

        self.iters_per_epoch = 0

        self.tag_map = {'map': {}, 'names': []}

        if self.glob_rank <= 0 and self.cconf.mode == 'train':
            self.__board = SummaryWriter(log_dir=self.cconf.paths['summary'])

        return True

    # def __del__(self):
    #     try:
    #         if self.cconf.backend == 'ddp' and self.cconf.mode in ['train', 'test']:
    #             dist.destroy_process_group()
    #     except Exception as e:
    #             print(e)

    def __get_info(self):
        ret = {
            'nets': self._nets,
            'paths': self.cconf.paths,
            'iters_per_epoch': self.iters_per_epoch,
            'progress': self.__progress,
            'cconf': self.cconf
        }
        return ret

    def on_train_step(self, datas, **kwargs):
        images, labels = datas[:2]

        preds = self.forward(images)
        loss = self.loss(preds, labels)

        loss.backward()

        self._step()

        return loss

    def on_validation_step(self, datas, **kwargs):
        """
        evalue your network on test so you can log every images or videos
        """
        images, labels = datas[:2]
        preds = self.forward(images)
        loss = self.loss(preds, labels)

        return loss

    def _on_train_epoch_begin(self, **kwargs):
        for name in self._nets.keys():
            if self._nets[name]['learnable']:
                self.set_net_state(name, True)
            else:
                self.set_net_state(name, False)
        return super()._on_train_epoch_begin(**kwargs)

    def _on_train_epoch_end(self, **kwargs):

        for tag_key in self.tag_map['map'].keys():
            kv = {}
            for name in self.tag_map['map'][tag_key]:
                value = self.__progress.get(name)
                if value is not None:
                    kv[name] = value
            self.watch_value(tag_key, kv, self.get_epoch_index())

        for name in self._nets.keys():
            self.watch_value('lr/'+name, self.get_lrs(name), self.get_epoch_index())
            self.watch_value('loss', {"train": self.__progress.get('Loss')}, self.get_epoch_index())
        
        if self.local_rank <= 0:
            self.logger.info(str(self.__progress))
        return super()._on_train_epoch_end(**kwargs)

    def _on_validation_epoch_end(self, **kwargs):
        for tag_key in self.tag_map['map'].keys():
            kv = {}
            for name in self.tag_map['map'][tag_key]:

                value = self.__progress.get(name)
                if value is not None:
                    kv[name] = value

            self.watch_value(tag_key, kv, self.get_epoch_index())
        return super()._on_validation_epoch_end(**kwargs)

    def _train(self, train_loader, test_loader, epoch_start, epoch_stop):
        """训练多个epoch

        Args:
            dataset (Dataset): 输入torch类型的Dataset,做训练集
            testset (Dataset): 输入torch类型的Dataset,做测试集
            epoch_start (int): 开始的epoch索引
            epoch_stop (int): 结束的epoch索引
        """
        epochs = epoch_stop - self.get_epoch_index()
        total = len(train_loader)

        disable = self.glob_rank > 0
        self.iters_per_epoch = len(train_loader)
        self._on_train_begin(**self.__get_info())
        desc = self.cconf.job_name+'-v'+self.cconf.version
        with tqdm(total=epochs, desc=desc, leave=True, disable=disable, colour='red') as ebar:
            for i in range(epoch_start, epoch_stop):
                self.__progress.set(num_batches=total, prefix="Train :[%3d]" % (self.get_epoch_index()))
                self.__progress.reset()

                train_loader.sampler.set_epoch(i)
                with tqdm(total=total, desc='train', leave=False, disable=disable, colour='cyan') as pbar:
                    self._on_train_epoch_begin(**self.__get_info())

                    for datas in train_loader:
                        datas = self._cuda_data(datas)
                        for key in self._nets.keys():
                            if self._nets[key]['learnable']:
                                self._nets[key]['optimizer'].zero_grad()

                        loss = self._on_train_step(datas, **self.__get_info())
                        self.__progress.update('Loss', loss.item())

                        # write loss into graph
                        if self.get_glob_step() % self.cconf.verbose_interval == 0:
                            self.watch_value('train/loss', loss.item(), self.get_glob_step())
                            pbar.set_postfix_str('loss:%.6f' % (self.__progress.get('Loss')))

                        self._update_glob_step()
                        pbar.update()

                    self._on_train_epoch_end(**self.__get_info())

                    flag = self.__validation(test_loader, pbar)
                    if flag:
                        self.logger.info('early stopping... epoch_index: %d ' % (self.get_epoch_index()))
                        ebar.set_description_str('early stopping...')
                        break

                ebar.update()
                self._update_epoch_index()
        self._on_train_end(**self.__get_info())

    def _test(self, test_loader):
        """测试

        Args:
            dataset (Dataset): 输入torch类型的dataset，做测试集
        """
        info = self.__get_info()
        self._on_test_epoch_begin(**info)
        self.glob_step = 1
        self.__validation(test_loader)
        self._on_test_epoch_end(**info)

    def __validation(self, loader, pbar=None):
        if loader is not None:
            total = len(loader)
            if pbar is not None:
                pbar.reset(total=total)
                pbar.set_description(desc="validation")
            else:
                pbar = tqdm(total=total, desc='validation', leave=True, disable=(self.glob_rank > 0), colour='green')

            for name in self._nets.keys():
                self.set_net_state(name, False)

            copy_prog = dcopy(self.__progress)

            self.__progress.set(num_batches=total, prefix='Test  :[%3d]' % (self.get_epoch_index()))
            self.__progress.reset()

            self._on_validation_epoch_begin(**self.__get_info())
            _val_fun = self._on_test_step if self.cconf.mode == 'test' else self._on_validation_step
            with torch.no_grad():
                for datas in loader:
                    info = self.__get_info()
                    datas = self._cuda_data(datas)
                    loss = _val_fun(datas, **info)

                    if loss is not None:
                        self.__progress.update('Loss', loss.item(), 1)
                        pbar.set_postfix_str("loss: %.6f" % self.__progress.get('Loss'))
                    pbar.update()
            value = self.__progress.get('Loss')
            if value is not None:
                self.watch_value('loss', {"test": value}, self.get_epoch_index())
            if self.local_rank <= 0:
                self.logger.info(str(self.__progress))
            self.__progress.update_meters(copy_prog)
        return self._on_validation_epoch_end(**self.__get_info())

    def __init_dist(self):
        if self.cconf.backend == 'ddp':
            # torch.set_num_threads(1)
            # self.logger.info("RANK: %d GANK: %d NODE: %d" % (self.local_rank, self.glob_rank, self.cconf.ddp.nprocs))
            ctip('cyan', 'bold', "RANK: %d GANK: %d NODE: %d" % (self.local_rank, self.glob_rank, self.cconf.ddp.nprocs), show=True)
            dist.init_process_group(backend='nccl', init_method=self.cconf.ddp.dist_url, world_size=self.cconf.ddp.nprocs, rank=self.glob_rank)

        self.gpu_idx = self.cconf.gpus#[i for i in range(len(self.cconf.gpu_idx))]

    def __init_seed(self):
        seed = self.cconf.seed
        cuda_deterministic = self.cconf.cuda_deterministic

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) # sets the seed for generating random numbers.
        torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
        torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

        if cuda_deterministic: # slower, more deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else: # faster, less deterministic
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    def __create_network(self, net, learnable=True):
        """根据类型创建网络

        Args:
            net (torch.nn.Module): torch网络
            learnable (bool): 是否可以学习

        Returns:
            nn.Module: DP,DDP or Single GPU Network
        """
        if self.cconf.backend == 'ddp' and self.cconf.mode != 'export':
            torch.cuda.set_device(self.local_rank)
            if learnable:
                net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
                net.cuda(self.local_rank)

                net = torch.nn.parallel.DistributedDataParallel(net,
                                                                device_ids=[self.local_rank],
                                                                find_unused_parameters=self.cconf['ddp']['find_unused_parameters'])
            else:
                net.cuda(self.local_rank)
        elif self.cconf.backend == 'dp' and self.cconf.mode != 'export':
            torch.cuda.set_device(self.gpu_idx[0])
            net.cuda(self.gpu_idx[0])
            if learnable:
                net = nn.DataParallel(net, device_ids=self.gpu_idx, output_device=self.gpu_idx[0])
        else:
            torch.cuda.set_device(self.gpu_idx[0])
            net.cuda(self.gpu_idx[0])
        return net

    def get_name(self, idx=0):
        """获取网络名称

        Args:
            idx (int, optional): 网络的索引. Defaults to 0.

        Returns:
            str: 网络名称
        """
        names = list(self._nets.keys())[idx]
        return names

    def get_net(self, name=None):
        """根据名称获取网络

        Args:
            name (str, optional): 输入名称. Defaults to None.

        Returns:
            nn.Module: torch类型的网络
        """
        names = list(self._nets.keys())
        name = name or names[0]
        return self._nets[name]['model']

    def add_net(self, name, net, learnable=True):
        """新增网络到流程中

        Args:
            name (str): 网络名称
            net (nn.Module): torch类型网络
            learnable (bool): 网络是否可以学习
        """
        assert name not in self._nets.keys() and isinstance(net, nn.Module)
        self._nets[name] = {'model': net, 'bind': False, 'learnable': learnable}
        if not learnable:
            finetune_layers(self._nets[name]['model'])

        param = model_parameters(net)
        if self.cconf.mode != 'train':
            info = '[%s(%s)] all param: %.6f(MB)' % (name, type(net).__name__, param['all'])
        else:
            info = '[%s(%s)] all param: %.6f learnable: %.6f' % (name, type(net).__name__, param['all'], param['learnable'])
        if self.local_rank <= 0:
            self.logger.info(info)

    def get_opt(self, name=None):
        names = list(self._nets.keys())
        name = name or names[0]
        return self._nets[name]['optimizer']

    def get_sch(self, name=None):
        names = list(self._nets.keys())
        name = name or names[0]
        return self._nets[name]['scheduler']

    def set_net_state(self, name, mode:bool):
        """设置网络的状态

        Args:
            name (str): 网络名称
            mode (bool): 模式（训练|测试）
        """
        self._nets[name]['model'].train(mode=mode)

    def set_opt_sch(self, name, opt, sch):
        """设置优化器和lr调试器

        Args:
            name (str): 网络名称
            opt (nn.Module): torch类型优化器
            sch (nn.Module): torch类型Scheduler
        """
        if opt is not None:
            self._nets[name]['optimizer'] = opt
        if sch is not None:
            self._nets[name]['scheduler'] = sch

    def add_watch(self, name, fmt=':6f', tag=None, have_average=False):
        """增加监控变量

        Args:
            name (str): 变量名称
            fmt (str, optional): 浮点格式. Defaults to ':6f'.
            tag (str, optional): 同tensorboard中tag. Defaults to None.
            have_average (bool, optional): 是否取平均值. Defaults to False.
        """
        assert name not in self.tag_map['names']
        _tag = tag or name
        self.tag_map['names'].append(name)
        if _tag not in self.tag_map['map'].keys():
            self.tag_map['map'][_tag] = []
        self.tag_map['map'][_tag].append(name)

        self.__progress.add(AverageMeter(name, fmt=fmt, hall=have_average))

    def upd_watch(self, name, value, size=1):
        """更新监控变量

        Args:
            name (str): 变量名称
            value (float): 变量的值
            size (int, optional): 变量的数量. Defaults to 1.
        """
        try:
            self.__progress.update(name, value, n=size)
        except KeyError:
            self.logger.error('Error[%d] %s key not in watcher.' % (name, self.local_rank))

    def get_watch(self, name):
        """获取变量的值

        Args:
            name (str): 变量名称

        Returns:
            float: 变量的值
        """
        try:
            return self.__progress.get(name)
        except KeyError:
            self.logger.error('Error[%d] %s key not in watcher.' % (name, self.local_rank))
        return None

    def get_lrs(self, name=None, idx=0):
        """获取当前优化器的学习率

        Args:
            name (str, optional): 网络名称. Defaults to None.
            idx (int, optional): 参数组索引. Defaults to 0.

        Returns:
            float: 学习率
        """
        # get learning rate: default is first net's lr
        assert self.cconf.mode == 'train'

        names = list(self._nets.keys())
        name = name or names[0]
        lr = 0 if not self._nets[name]['learnable'] else self._nets[name]['optimizer'].param_groups[idx]['lr']
        return lr

    def _lr_step(self, name=None, **kwargs):
        names = list(self._nets.keys())
        model_name = name or names[0]
        if 'scheduler' in self._nets[model_name].keys():
            self._nets[model_name]['scheduler'].step(**kwargs)

    def clip_grad(self, cval, name=None):
        if cval is not None:
            net = self.get_net(name)
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, net.parameters()),
                            max_norm=cval)

    def get_data(self, idx=0, *vars):
        assert self.cconf.mode == 'train', 'Error: get_data just support in "mode: train"'
        new_var = []
        start = idx * self.grad_step
        stop = min(start + self.grad_step, self.batch_size)
        for data in vars:
            new_var.append(data[start:stop])

        return new_var

    def forward_withpart(self, index, name=None, *vars, **invars):
        """
            vars: 表示顺序输入的变量（都是可分割的）
            invars: 表示不可变量，以字典形式输入
        """

        net = self.get_net(name=name)
        if self.cconf.backend == 'ddp':
            with net.no_sync():
                inp = self.get_data(index, *vars)
                outs = net(*inp, **invars)
        else:
            inp = self.get_data(index, *vars)
            outs = net(*inp, **invars)
        return outs

    def forward(self, *args, **kwargs):
        names = list(self._nets.keys())
        model_name = kwargs.get('name', None) or names[0]
        kwargs.pop('name', None)

        net = self._nets[model_name]['model']
        return net(*args, **kwargs)

    def _step(self, name=None, clip_mval=None):
        names = list(self._nets.keys())
        name = name or names[0]

        opt = self._nets[name]['optimizer']
        self.clip_grad(clip_mval, name)
        opt.step()
        opt.zero_grad()

    def watch_value(self, tag, scalar_value, global_step=None, walltime=None):
        """将value写入summarywriter
        Args:
            tag (str): 标记名称
            scalar_value (float) : 写入的值
            global_step (int): y轴坐标
            walltime (float): 等待刷新时间
        """
        if self.glob_rank <= 0 and self.cconf.mode == 'train':
            if isinstance(scalar_value, dict):
                self.__board.add_scalars(tag, scalar_value, global_step=global_step)
            else:
                self.__board.add_scalar(tag, scalar_value, global_step=global_step)

    def watch_model(self, input_tensor, model_name=None):
        """将模型写入summarywriter

        Args:
            input_tensor (torch.Tensor): 输入特征张量
            model_name (str, optional): 模型名称. Defaults to None.
        """
        if self.glob_rank <= 0:
            net = self.get_net(name=model_name)
            self.__board.add_graph(net, input_tensor)

    def configure_optimizers(self):
        raise NotImplementedError

    def _cuda_data(self, datas):
        if isinstance(datas, dict):
            ds = {}
            for key in datas.keys():
                ds[key] = datas[key].cuda(non_blocking=True)
        else:
            ds = []
            for data in datas:
                try:
                    data = data.cuda(non_blocking=True)
                except:
                    pass
                ds.append(data)
        return ds

    def __resume(self, model_path):
        if self.local_rank <= 0 and os.path.isfile(model_path):
            strict = False if self.cconf.mode == 'train' else True
            ret = load_spec_wgts(model_path, self._nets, strict, show=True)
            try:
                glob_index, epoch_index = ret
                self._update_epoch_index(epoch_index+1)
                self._update_glob_step(glob_index+1)
            except:
                pass
            self.logger.info('%s : model load success' % (model_path))

    def init(func):
        def warp(self, *args, **kwargs):
            Engine.initialize(self)
            func(self, *args, **kwargs)
            Engine.__backend(self)
        return warp

    def __backend(self):
        mode = self.cconf.mode
        self.__resume(self.cconf.ckpt_path)

        #create network
        for name in self._nets.keys():
            if not self._nets[name]['bind']:
                self._nets[name]['model'] = self.__create_network(self._nets[name]['model'], self._nets[name]['learnable'])
                self._nets[name]['bind'] = True

        # wait for create net work
        if mode == 'train':
            self.configure_optimizers()
            for name in self._nets.keys():
                if self._nets[name]['learnable']:
                    self._nets[name]['optimizer'].zero_grad()

    def create_loader(self):
        raise NotImplementedError

    def run(self):
        if self.local_rank <= 0:
            ctip("cyan", "|begin to load dataset...", show=True)
        train_loader, test_loader = self.create_loader()
        if self.cconf.mode == 'test':
            assert len(test_loader) > 0
            self._test(test_loader)
        elif self.cconf.mode == 'train':
            assert len(train_loader) > 0
            if self.cconf.backend != 'ddp':
                train_loader.sampler.set_epoch = lambda i: i
            self._train(train_loader, test_loader, 0, self.cconf.max_epoch)
        else:
            if self.local_rank <= 0:
                self.logger.error("please set mode to [test, train]")


def vrun_func(local_rank, glob_rank, conf, cbs, Module):
    cconf = conf['common']
    if cconf['ddp']['use_spawn']:
        glob_rank = local_rank
    app_name = '%s_%s.log' % (cconf.job_name, cconf.mode)
    logf = os.path.join(cconf['paths']['root'], app_name)
    logger.remove(handler_id=None)
    #logger.add(logf, rotation='50 MB', enqueue=cconf.backend=='ddp')
    logger.add(logf, rotation='50 MB', enqueue=(cconf.backend == 'ddp'),
                format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>')

    module = Module(conf=conf, cbs=cbs, logger=logger, local_rank=local_rank, glob_rank=glob_rank)

    module.run()

def vrun(Module, cbs=[]):
    ef = EngineFileSystem()

    conf = ef.deploy()

    cconf = conf['common']

    if conf['vtraining']['enable']:
        assert not cconf['ddp']['use_spawn'], "vtraining can not with use_spawn together"
    else:
        conf.remove('vtraining')

    if cconf.backend == 'ddp':
        if cconf['ddp']['use_spawn']:
            mp.spawn(fn=vrun_func, nprocs=cconf.ddp.nprocs, args=(cconf.glob_rank, conf, cbs, Module), join=True)
        else:
            vrun_func(cconf.local_rank, cconf.glob_rank, conf, cbs, Module)
    else:
        vrun_func(cconf.local_rank, cconf.glob_rank, conf, cbs, Module)
