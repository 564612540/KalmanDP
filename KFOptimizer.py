import torch
from torch.optim.optimizer import Optimizer, required
import torch.autograd as ta
from collections import defaultdict
from typing import Callable, List, Optional, Union

class KFOptimizer(Optimizer):
    def __init__(self, params, optimizer:Optimizer, kappa = 0.9, gamma = 1.0):
        '''
        # wrapping up the optimizer with
        optimizer = KFOptimizer(model.parameters(), optimizer, sigma_H, sigma_g)
        # before the first step of gradient accumulation:
        if t % acc_step == 0 and hasattr(optimizer, 'prestep'):
            optimizer.prestep()
        '''
        if gamma ==0:
            gamma = (1-kappa)/kappa
            self.compute_grad = False
        elif abs(gamma - (1-kappa)/kappa) <1e-3:
            gamma = (1-kappa)/kappa
            self.compute_grad = False
        else:
            self.scaling_factor = (gamma*kappa+kappa-1)/(1-kappa)
            self.compute_grad = True
        defaults = dict(kappa = kappa, gamma=gamma)
        self.optimizer = optimizer
        # if nesterov and (momentum <= 0 or dampening != 0):
            # raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(KFOptimizer, self).__init__(params, defaults)

    # def __setstate__(self, state):
    #     super(KFOptimizer, self).__setstate__(state)

    @property
    def param_groups(self) -> List[dict]:
        """
        Returns a list containing a dictionary of all parameters managed by the optimizer.
        """
        return self.original_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups: List[dict]):
        """
        Updates the param_groups of the optimizer.
        """
        self.original_optimizer.param_groups = param_groups

    @property
    def state(self) -> defaultdict:
        """
        Returns a dictionary holding current optimization state.
        """
        return self.original_optimizer.state

    @state.setter
    def state(self, state: defaultdict):
        """
        Updates the state of the optimizer.
        """
        self.original_optimizer.state = state

    @property
    def defaults(self) -> dict:
        """
        Returns a dictionary containing default values for optimization.
        """
        return self.original_optimizer.defaults

    @defaults.setter
    def defaults(self, defaults: dict):
        """
        Updates the defaults of the optimizer.
        """
        self.original_optimizer.defaults = defaults

    def prestep(self, closure=required):
        loss = None
        for group in self.param_groups:
            gamma = group['gamma']
            break
        if self.compute_grad:
            with torch.enable_grad():
                loss = closure() # compute grad
        # totoal_grad = 0
        for group in self.param_groups:
            gamma = group['gamma']
            for p in group['params']:
                state = self.state[p]
                if 'kf_d_t' not in state:
                    continue
                # perturb
                p.data.add_(state['kf_d_t'], alpha = gamma)
                if self.compute_grad:
                    if hasattr(p, 'private_grad'):
                        p.private_grad.mul_(self.scaling_factor)
                    elif p.grad is not None:
                        p.grad.mul_(self.scaling_factor)
                    else:
                        raise RuntimeError("Must have either grad or private_grad!")
        with torch.enable_grad():
            if self.compute_grad:
                closure()
            else:
                loss = closure()
        for group in self.param_groups:
            gamma = group['gamma']
            for p in group['params']:
                state = self.state[p]
                if 'kf_d_t' not in state:
                    continue
                # perturb back
                p.data.add_(state['kf_d_t'], alpha = -gamma)
                if self.compute_grad:
                    if hasattr(p, 'private_grad'):
                        p.private_grad.div_(self.scaling_factor)
                    elif p.grad is not None:
                        p.grad.div_(self.scaling_factor)
        return loss
            

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        scaling_factor = 0.0
        for group in self.param_groups:
            kappa = group['kappa']
            for p in group['params']:
                has_private_grad = False
                if hasattr(p, 'private_grad'):
                    grad = p.private_grad
                    has_private_grad = True
                elif p.grad is not None:
                    grad = p.grad
                else:
                    continue
                if self.compute_grad:
                    grad.div_(1+1/self.scaling_factor)
                state = self.state[p]
                if 'kf_d_t' not in state:
                    state['kf_d_t'] = torch.zeros_like(p.data).to(p.data)
                    state['kf_m_t'] = grad.clone().to(p.data)
                state['kf_m_t'].lerp_(grad, weight = kappa)
                if has_private_grad:
                    p.private_grad = state['kf_m_t'].clone().to(p.data)
                else:
                    p.grad = state['kf_m_t'].clone().to(p.data)
                    scaling_factor += p.grad.norm().pow(2)
                state['kf_d_t'] = -p.data.clone().to(p.data)
        if scaling_factor > 0 and not has_private_grad:
            scaling_factor = scaling_factor.sqrt()
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.div_(scaling_factor)
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    self.state[p]['kf_d_t'].add_(p.data, alpha = 1)
        return loss