import re
import time
import torch
import socket
import torch.nn as nn
from collections import OrderedDict


def show_time(show):
    def func_time(func):
        def inner(*args, **kw):
            start_time = time.time()
            func(*args, **kw)
            end_time = time.time()
            rtime = end_time - start_time
            if show:
                print('函数运行时间为：', rtime, 's')
            return rtime

        return inner

    return func_time


def load_wgts(net, state_dict, strict=True, show=True):
    """加载权重

    Args:
        net (nn.Module): 输入网络
        state_dict (dict): 从torch.load()加载的权重
        strict (bool, optional): 是否严格匹配字典. Defaults to True.
        show (bool, optional): 是否显示信息. Defaults to True.

    Raises:
        ValueError: 如果严格控制，不能加载部分权重

    Returns:
        nn.Module: 输出网络
    """
    device = next(net.parameters()).device
    dst = net.state_dict()
    src = state_dict
    pretrained_dict = {}
    for k, v in dst.items():
        mk = k[7:]
        if k in src and src[k].size() == v.size():
            # k in state_dict (same as net)
            pretrained_dict[k] = src[k].to(device)
        elif mk in src and src[mk].size() == v.size():
            # mk in state_dict (not same as net)
            pretrained_dict[k] = src[mk].to(device)
        else:
            pass

    if len(pretrained_dict) == len(dst):
        print("%s : All parameters loading." % type(net).__name__)
    else:
        if strict:
            raise ValueError("Strict Parameter load failed")

        nkey = 0
        not_loaded_keys = []
        for k in dst.keys():
            if k not in pretrained_dict.keys():
                not_loaded_keys.append(k)
            else:
                nkey += 1

        if show:
            print('%s: Some params were not loaded:' % type(net).__name__)
            # print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
            print("load %d keys of %d" % (len(pretrained_dict), len(src)))
        if nkey == 0:
            return None

    dst.update(pretrained_dict)
    net.load_state_dict(dst)

    return net


def save_wgts(net, path, odict={}):
    """保存权重

    Args:
        net (nn.Module): 输入网络
        path (str): 保存路径
        odict (dict, optional): 保存的额外参数. Defaults to {}.
    """
    state_dict = net.state_dict()
    save_dict = OrderedDict()
    for key in state_dict.keys():
        mkey = key[7:] if 'module' in key else key
        save_dict[mkey] = state_dict[key].cpu()

    save_dict = {'state_dict': save_dict}
    save_dict.update(odict)
    torch.save(save_dict, path)


def save_spec_wgts(path, nets: dict, glob_step, epoch_index):
    save_dict = {"glob_step": glob_step, "epoch_index": epoch_index}
    for name in nets.keys():
        state_dict = nets[name]['model'].state_dict()
        temp_dict = OrderedDict()
        for key in state_dict.keys():
            mkey = key[7:] if 'module' in key else key
            temp_dict[mkey] = state_dict[key].cpu()

        save_dict[name] = {"state_dict": temp_dict}

    torch.save(save_dict, path)


def load_spec_wgts(path, nets: dict, strict=True, show=True):
    ckpts = torch.load(path, map_location='cpu')

    for key in nets.keys():
        load_wgts(nets[key]['model'], ckpts[key]['state_dict'], strict=strict, show=show)

    glob_step = ckpts['glob_step']
    epoch_index = ckpts['epoch_index']

    return glob_step, epoch_index


def model_parameters(model):
    """计算模型的参数量

    Args:
        model (nn.Module): 输入待计算的模型

    Returns:
        dict: {"all": 所有参数量, "learnable": 可学习的参数量}
    """
    mb = 1000000

    params = {"all": 0, "learnable": 0}

    for p in model.parameters():
        np = p.numel()
        if p.requires_grad:
            params['learnable'] += np
        params['all'] += np

    params['all'] /= mb  # MB
    params['learnable'] /= mb

    return params


def frozen_layers(model, layer_names=[]):
    """冻结模型中某一层或几层的参数

    Args:
        model (nn.Module): 输入模型
        layer_names (list, optional): 需要冻结的层. Defaults to [].
    Returns:
        list: 输出finetune的层
    """
    pattern = re.compile('|'.join(layer_names), re.I)

    names = []
    for name, param in model.named_parameters():
        if re.search(pattern, name) and len(layer_names) != 0:
            param.requires_grad = False
            names.append(name)
        else:
            param.requires_grad = True
    return names


def finetune_layers(model, layer_names=[]):
    """finetune模型中某一层或几层的参数

    Args:
        model (nn.Module): 输入模型
        layer_names (list, optional): 需要冻结的层. Defaults to [].
    Returns:
        list: 输出finetune的层
    """
    pattern = re.compile('|'.join(layer_names), re.I)

    names = []
    for name, param in model.named_parameters():
        if re.search(pattern, name) and len(layer_names) != 0:
            param.requires_grad = True
            names.append(name)
        else:
            param.requires_grad = False
    return names


def model_layers(model):
    """打印模型所有参数

    Args:
        model (nn.Module): 输入模型
    """
    for name, param in model.named_parameters():
        print(name, list(param.size()))


def split_param(net, layer_names=[]):
    """分割参数

    Args:
        net (nn.Module): 输入网络
        layer_names (list, optional): 待分割的网络层名称. Defaults to [].

    Returns:
        tuple: (输入网络的参数, 升入网络的参数)
    """
    idparam = []
    for name, param in net.named_parameters():
        idp = id(param)
        if name in layer_names:
            idparam.append(idp)

    oparams = filter(lambda x: x not in idparam, net.parameters())
    iparams = filter(lambda x: x in idparam, net.parameters())

    return oparams, iparams


def count_param(parameters):
    return len(list(map(id, parameters)))


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for datas in dataloader:
        inputs = datas[0]
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('localhost', 0))
    sockname = sock.getsockname()
    sock.close()
    return sockname[1]
