"""
Network Initializations
"""

import logging
import importlib
import torch
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.panoptic_deeplab import (
    add_panoptic_deeplab_config
)
import _init_paths
import d2

def get_net(args, criterion):
    """
    Get Network Architecture based on arguments provided
    """
    if args.config_file is not None and not args.config_file == "":
        cfg = get_cfg()
        add_panoptic_deeplab_config(cfg)
        cfg.merge_from_file(args.config_file)
        net = build_model(cfg)

    else:
        net = get_model(network=args.arch, num_classes=args.dataset_cls.num_classes,
                        criterion=criterion)
    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.1f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net


def wrap_network_in_dataparallel(net, use_apex_data_parallel=False):
    """
    Wrap the network in Dataparallel
    """
    if use_apex_data_parallel:
        import apex
        net = apex.parallel.DistributedDataParallel(net)
    else:
        net = torch.nn.DataParallel(net)
    return net


def get_model(network, num_classes, criterion):
    """
    Fetch Network Function Pointer
    """
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(num_classes=num_classes, criterion=criterion)
    return net
