import logging
import torch.optim as optim
from robustbench.data import load_cifar100c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy
from conf_demo import cfg, load_cfg_fom_args
import vmp as mop
import sys
import time
logger = logging.getLogger(__name__)

def evaluate(description):
    print('='*100)
    start_t = time.time()
    start_str = time.strftime("Start Time: %Y-%m-%d %H:%M:%S", time.localtime())
    print(start_str) 

    load_cfg_fom_args(description)
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    model = setup_mop(base_model)

    err_list = []
    for severity in cfg.CORRUPTION.SEVERITY:
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            if i_c == 0:
                try:
                    model.reset()
                    logger.info("resetting model")
                except:
                    a =1
            else:
                a=1
            x_test, y_test = load_cifar100c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            acc = accuracy(model, x_test, y_test, cfg.batch_size)
            err = 1. - acc
            err_list.append(err)
            print(f"[{i_c}] error % [{corruption_type}{severity}]: {err:.2%}")
            sys.stdout.flush()
    mean_err = sum(err_list) / len(err_list)
    print(err_list)
    print(f"mean error: {mean_err:.2%}")
    end_t = time.time()
    last_time = (end_t - start_t)/60
    end_str = time.strftime("End Time: %Y-%m-%d %H:%M:%S", time.localtime())
    print('!!!'+start_str) 
    print('!!!'+end_str)
    print('!!!Last Time: {:.1f}min'.format(last_time))
    print('='*120)

    sys.stdout.flush()

def setup_mop(model):
    params,cls_params = mop.variational_fisher(model,cfg)
    optimizer = setup_optimizer_mop(params,cls_params)
    mop_model = mop.Mop(model, optimizer, cfg)
    print(f"optimizer for adaptation: ", optimizer)
    return mop_model

def setup_optimizer_mop(params,cls_params):
    param_group = [
        {'params': params, 'lr': cfg.lr},
        {'params': cls_params, 'lr':  cfg.lr2}
    ]

    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(param_group,
                    lr=cfg.lr,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(param_group,
                   lr=cfg.lr,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation.')
