import torch
from torch.optim.optimizer import Optimizer, required
import torch.autograd as ta

class KFAdam(Optimizer):
    def __init__(self, params, lr=required, betas = (0.9, 0.999), weight_decay = 0, sigma_dp = 0):
        '''
        # wrapping up the optimizer with
        optimizer = KFOptimizer(model.parameters(), optimizer, sigma_H, sigma_g)
        # before the first step of gradient accumulation:
        if t % acc_step == 0 and hasattr(optimizer, 'prestep'):
            optimizer.prestep()
        '''
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr = lr, betas=betas, weight_decay = weight_decay, sigma_dp=sigma_dp)
        # if nesterov and (momentum <= 0 or dampening != 0):
            # raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(KFAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(KFAdam, self).__setstate__(state)

    # def prestep(self):
    #     for group in self.param_groups:
    #         sigma_g = group['sigma_g']
    #         sigma_H = group['sigma_H']
    #         for p in group['params']:
    #             if not p.requires_grad:
    #                 continue
    #             if 'kf_beta_t' not in p_state:
    #                 continue
    #             beta_t = p_state['kf_beta_t'] + sigma_H**2
    #             k_t = beta_t/(beta_t + sigma_g**2 - sigma_H**2 )
    #             k_1 = (1-k_t)/k_t
    #             p.data.add_(p_state['kf_d_t'], alpha = k_1)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            sigma_dp = group['sigma_dp']
            for p in group['params']:
                has_private_grad = False
                if not p.requires_grad:
                    continue
                if hasattr(p, 'private_grad'): # I think there should only be grad (privatized gradient)
                    grad = p.private_grad
                    has_private_grad = True
                elif p.grad is not None:
                    grad = p.grad
                else:
                    continue
                p_state = self.state[p]
                if 'kf_beta_t' not in p_state: #first iteration
                    p_state['exp_avg'] = torch.zeros_like(p.data).to(p.data)
                    p_state['exp_avg_sq'] = torch.zeros_like(p.data).to(p.data)
                    p_state['kf_beta_t'] = torch.zeros_like(p.data).to(p.data)
                    p_state['kf_m_t'] = grad.clone().to(p.data)
                
                beta_t_ = p_state['kf_beta_t'] + 
                k_t = beta_t/(beta_t + sigma_g**2 - sigma_H**2)
                k_1 = (1-k_t)/k_t
                p_state['kf_beta_t'] = (1-k_t)*beta_t
                p.data.add_(p_state['kf_d_t'], alpha = -k_1)
                p_state['kf_m_t'].lerp_(grad, weight = k_t)
                if has_private_grad:
                    p.private_grad = p_state['kf_m_t'].clone().to(p.data)
                else:
                    p.grad = p_state['kf_m_t'].clone().to(p.data)
                p_state['kf_d_t'] = -p.data.clone().to(p.data)
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p_state['kf_d_t'].add_(p.data, alpha = 1)
        return loss
    