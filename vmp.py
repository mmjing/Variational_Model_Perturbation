from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from functools import partial
import types

class Mop(nn.Module):
    def __init__(self, model, optimizer,cfg):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = cfg.MODEL.EPISODIC
        self.var_scal = cfg.var_scal
        self.kl_par = cfg.kl_par
        self.entr_par = cfg.entr_par

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    @torch.enable_grad()
    def forward_and_adapt(self,x, model, optimizer):
        outputs = model(x)
        loss = self.entr_par * softmax_entropy(outputs).mean(0)
        kl_loss = self.kl_par * get_kl_loss(model,self.var_scal)
        loss += kl_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return outputs

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def get_kl_loss(model,variance_scaling):
    modules = [x for x in model.modules() if hasattr(x, 'logvar0')]
    component1 = 0
    component2 = 0
    k = 0
    for x in modules:
        k += x.weight.numel()
        p_var = variance_scaling * torch.ones_like(x.weight).float() * x.sigma0
        component1 += (p_var.log() - x.logvar0).sum()
        component2 += (x.logvar0.exp()/p_var).sum()
    kl_loss = 0.5 *(component1 -k + component2)/k
    return kl_loss

def variational_forward(module, input):
    var = module.logvar0.expand_as(module.weight).exp()
    if isinstance(module, torch.nn.modules.conv.Conv2d):
        output = F.conv2d(input,module.weight, module.bias, module.stride,
                          module.padding, module.dilation, module.groups)
        output_var = F.conv2d(input ** 2 + 1e-2, var, None, module.stride,
                              module.padding, module.dilation, module.groups)
    elif isinstance(module, torch.nn.modules.linear.Linear):
        output = F.linear(input,  module.weight, module.bias)
        output_var = F.linear(input ** 2 + 1e-2, var, None)
    else:
        raise NotImplementedError("Module {} not implemented.".format(type(module)))
    eps = torch.empty_like(output).normal_()
    return output + torch.sqrt(output_var) * eps

def get_variational_vars(model):
    """Returns all variables involved in optimizing the hessian estimation."""
    result = []
    if hasattr(model, 'logvar0'):
        result.append(model.logvar0)
    for l in model.children():
        result += get_variational_vars(l)
    return result

def _add_logvar(module, args,is_bt=False):
    learn_dims = args.learn_dims
    variance_scaling = args.var_scal_init
    if not hasattr(module, 'weight'):
        return
    w = module.weight.data
    if w.dim() < 2:
        return
    if not hasattr(module, 'logvar0'):
        if learn_dims == 'all' or is_bt:
            var = w.view(w.size(0), -1).var(dim=1).view(-1, *([1] * (w.dim() - 1)))
            logvar_expand = (var * variance_scaling + 1e-10).log().clone().expand_as(module.weight)
            empty_var = torch.ones_like(w).float()
            module.logvar0 = Parameter(empty_var)
            module.logvar0.data[:] = logvar_expand

        elif learn_dims == 'one':
            var = w.view(w.size(0), -1).var(dim=1).view(-1, *([1] * (w.dim() - 1)))
            module.logvar0 = Parameter(var.log())
            module.logvar0.data[:] = (var * variance_scaling + 1e-10).log()

        module.sigma0 = w.var(dim=list(range(args.pr_dim_start,w.dim())),keepdim=True).expand_as(w)
        module.sigma0.requires_grad = False

def make_variational(model,args,is_bt=False):
    """Replaces the forward pass of the model layers to add noise."""
    model.apply(partial(_add_logvar, args=args,is_bt=is_bt))
    for m in model.modules():
        if hasattr(m, 'logvar0'):
            m.forward = types.MethodType(variational_forward, m)

def variational_fisher(model, args):
    model.train()
    model.requires_grad_(False)
    print("hook logvar...")
    parameters = []
    cls_parameters = []

    for m in model.layers[0:-1]:
        if isinstance(m, nn.Module) and (not isinstance(m, nn.BatchNorm2d)):
            make_variational(m,args)
            parameters += get_variational_vars(m)

    make_variational(model.layers[-1],args,is_bt=True)
    cls_parameters += get_variational_vars(model.layers[-1])

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            parameters += list(m.parameters())

    print("hook complete!")
    return parameters,cls_parameters



