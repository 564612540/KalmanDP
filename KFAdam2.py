import torch
from torch.optim.optimizer import Optimizer, required
import torch.autograd as ta

class KFAdam(Optimizer):
    def __init__(self, params, lr=required, betas = (0.9, 0.999), weight_decay = 0, gamma=1e-2, sigma_dp = 0):
        '''
        # Kalman filter + Adam, with bias correction and variance estimation. 
        # Does not need to tune any parameter except the learning rate
        # before the first step of gradient accumulation:
        if t % acc_step == 0 and hasattr(optimizer, 'prestep'):
            optimizer.prestep()
        '''
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {}".format(betas[0]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr = lr, betas=betas, weight_decay = weight_decay, gamma=gamma, sigma_dp=sigma_dp)
        # if nesterov and (momentum <= 0 or dampening != 0):
            # raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(KFAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(KFAdam, self).__setstate__(state)

    def prestep(self, closure=required):
        for group in self.param_groups:
            sigma_dp = group['sigma_dp']
            beta1,beta2 = group['betas']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                if 'kf_beta_t' not in self.state[p]:
                    continue
                p_state = self.state[p]
                bias_correction_1 = 1 - beta1**p_state['step']
                beta_t_ = p_state['kf_beta_t'] + p_state['kf_sigma_H_sq'].divide(bias_correction_1)
                k_t = beta_t_/(beta_t_ + sigma_dp**2)
                gamma = (1-k_t)/k_t
                if isinstance(gamma, torch.Tensor):
                    gamma = gamma.item()
                p.data.add_(self.state[p]['kf_d_t'], alpha = gamma)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        for group in self.param_groups:
            sigma_dp = group['sigma_dp']
            beta1,beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                if hasattr(p, 'private_grad'): # I think there should only be grad (privatized gradient)
                    grad = p.private_grad
                elif p.grad is not None:
                    grad = p.grad
                else:
                    continue
                p_state = self.state[p]
                if 'kf_beta_t' not in p_state: #first iteration, initialization
                    p_state['step'] = 0
                    p_state['exp_avg_sq'] = torch.zeros(1).to(p.data)
                    p_state['kf_beta_t'] = sigma_dp**2
                    p_state['kf_m_t'] = grad.clone().to(p.data)
                    p_state['kf_d_t'] = torch.zeros_like(p.data).to(p.data)
                    p_state['kf_sigma_H_sq'] = 0
                p_state['step'] += 1
                bias_correction_1 = 1 - beta1**p_state['step']                
                beta_t_ = p_state['kf_beta_t'] + p_state['kf_sigma_H_sq']/bias_correction_1
                k_t = beta_t_/(beta_t_ + sigma_dp**2)
                gamma = (1-k_t)/k_t
                p_state['kf_beta_t'] = (1-k_t)*beta_t_
                if isinstance(gamma, torch.Tensor):
                    gamma = gamma.item()

                p.data.add_(p_state['kf_d_t'], alpha = -gamma)
                bias_correction_2 = 1 - beta2**p_state['step']
                exp_avg_sq_hat = torch.divide(p_state['exp_avg_sq'], bias_correction_2).subtract(sigma_dp**2).clamp_min(1e-8)
                g_avg_sq = torch.norm(p_state['kf_m_t']).pow(2).div(torch.numel(grad)).subtract(p_state['kf_beta_t']).clamp_min(0)
                p_state['exp_avg_sq'].mul_(beta2).add_(torch.norm(grad).pow(2).div(torch.numel(grad)), alpha= 1-beta2)
                p_state['kf_sigma_H_sq'] = beta1*p_state['kf_sigma_H_sq'] + exp_avg_sq_hat.subtract(g_avg_sq).clamp_min(1e-8).multiply(1-beta1)

                p_state['kf_m_t'].lerp_(grad, weight = k_t)
                # if has_private_grad:
                #     p.private_grad = p_state['kf_m_t'].clone().to(p.data)
                # else:
                #     p.grad = p_state['kf_m_t'].clone().to(p.data)
                p_state['kf_d_t'] = -p.data.clone().to(p.data)

                # begin update
                if weight_decay != 0:
                    p.data.mul_(1 - group['lr'] * weight_decay)
                
                p.data.add_(p_state['kf_m_t'], alpha = -lr/(exp_avg_sq_hat.sqrt().item()))

                p_state['kf_d_t'].add_(p.data, alpha = 1)
        return loss
    