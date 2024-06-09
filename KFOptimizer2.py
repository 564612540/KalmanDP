import torch
from torch.optim.optimizer import Optimizer, required
import torch.autograd as ta

class KFOptimizer2(Optimizer):
    def __init__(self, params, optimizer:Optimizer, sigma_H=0, sigma_g=1e-5):
        '''
        # wrapping up the optimizer with
        optimizer = KFOptimizer(model.parameters(), optimizer, sigma_H, sigma_g)
        # before the first step of gradient accumulation:
        if t % acc_step == 0 and hasattr(optimizer, 'prestep'):
            optimizer.prestep()
        '''
        defaults = dict(sigma_H=sigma_H, sigma_g=sigma_g)
        self.optimizer = optimizer
        # if nesterov and (momentum <= 0 or dampening != 0):
            # raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(KFOptimizer2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(KFOptimizer2, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            sigma_g = group['sigma_g']
            sigma_H = group['sigma_H']
            for p in group['params']:
                has_private_grad = False
                if not p.requires_grad:
                    continue
                else:
                    p.previous_data=p.data.clone()
                if hasattr(p, 'private_grad'):
                    grad = p.private_grad
                    has_private_grad = True
                elif p.grad is not None:
                    grad = p.grad
                else:
                    continue
                if 'kf_beta_t' not in self.state[p]:
                    self.state[p]['kf_beta_t'] = sigma_g**2 #??? #t=0
                    # self.state[p]['kf_d_t'] = torch.zeros_like(p.data).to(p.data)
                    self.state[p]['kf_m_t'] = grad.clone().to(p.data) # t=-1??? not mention in paper
                beta_t = self.state[p]['kf_beta_t'] + sigma_H**2 
                k_t = beta_t/(beta_t + sigma_g**2 - sigma_H**2) #t=0
                self.state[p]['kf_beta_t'] = (1-k_t)*beta_t #t=0
                
                self.state[p]['kf_m_t'].lerp_(grad, weight = (1-k_t)/k_t) #t>=0
        
                if has_private_grad:
                    p.private_grad = self.state[p]['kf_m_t'].clone().to(p.data)
                else:
                    p.grad = self.state[p]['kf_m_t'].clone().to(p.data)

        for group, group_orig in zip(self.param_groups,self.optimizer.param_groups):
            group['lr']=(1-(1-k_t)/k_t)*group_orig['lr']

        loss = self.optimizer.step(closure)

        return loss
