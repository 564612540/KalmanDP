import torch
from torch.optim.optimizer import Optimizer, required
import torch.autograd as ta

class KFAdam2(Optimizer):
    def __init__(self, params, lr=required, beta2 = 0.999, weight_decay = 0, gamma=1e-2, sigma_dp = 0, kappa = 1.5):
        '''
        # Kalman filter + Adam, with bias correction and variance estimation. 
        # Does not need to tune any parameter except the learning rate
        # before the first step of gradient accumulation:
        if t % acc_step == 0 and hasattr(optimizer, 'prestep'):
            optimizer.prestep()
        '''
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta parameter: {}".format(beta2))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr = lr, beta2=beta2, weight_decay = weight_decay, gamma=gamma, sigma_dp=sigma_dp, kappa = kappa)
        # if nesterov and (momentum <= 0 or dampening != 0):
            # raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(KFAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(KFAdam, self).__setstate__(state)

    def prestep(self, closure=required):
        loss = None
        for group in self.param_groups:
            kappa = group['kappa']
            gamma = group['gamma']
            if gamma>0: # only compute grad_plus
                for p in group['params']:
                    if not p.requires_grad:
                        continue
                    state = self.state[p]
                    if 'kf_beta_t' not in state:
                        continue
                    k_t = (state['kf_beta_t'] + 1.0)/(state['kf_beta_t'] + kappa**2)
                    break
                compute_grad = True
                break
            else:
                compute_grad = False
                break
        if compute_grad:
            with torch.enable_grad():
                loss = closure(scale = 1 - (1-k_t)/(gamma*k_t)) # compute grad
        for group in self.param_groups:
            kappa = group['kappa']
            gamma = group['gamma']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                if 'kf_beta_t' not in state:
                    continue
                k_t = (state['kf_beta_t'] + 1.0)/(state['kf_beta_t'] + kappa**2)
                if gamma<=0:
                    gamma = (1-k_t)/k_t
                # perturb
                p.data.add_(state['kf_d_t'], alpha = gamma)
        with torch.enable_grad():
            if compute_grad:
                closure(scale = (1-k_t)/(gamma*k_t))
            else:
                loss = closure(scale = (1-k_t)/(gamma*k_t))
        for group in self.param_groups:
            kappa = group['kappa']
            gamma = group['gamma']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                if 'kf_beta_t' not in state:
                    continue
                k_t = (state['kf_beta_t'] + 1.0)/(state['kf_beta_t'] + kappa**2)
                if gamma<=0:
                    gamma = (1-k_t)/k_t
                # perturb back
                p.data.add_(state['kf_d_t'], alpha = -gamma)
        return loss

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
            beta2 = group['beta2']
            lr = group['lr']
            weight_decay = group['weight_decay']
            kappa = group['kappa']
            gamma = group['gamma']
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
                    p_state['bias_correction_1'] = 1
                    p_state['kf_beta_t'] = 1
                    p_state['kf_m_t'] = torch.zeros_like(p.data).to(p.data)
                    p_state['kf_d_t'] = torch.zeros_like(p.data).to(p.data)
                p_state['step'] += 1              
                k_t = (p_state['kf_beta_t'] + 1.0)/(p_state['kf_beta_t'] + kappa**2)
                if gamma<=0:
                    gamma = (1-k_t)/k_t
                p_state['kf_beta_t'] = (1-k_t)*(p_state['kf_beta_t'] + 1.0)
                if isinstance(gamma, torch.Tensor):
                    gamma = gamma.item()

                bias_correction_2 = 1 - beta2**p_state['step']
                p_state['bias_correction_1'] =  (1-k_t)*p_state['bias_correction_1'] + k_t
                exp_avg_sq_hat = torch.divide(p_state['exp_avg_sq'], bias_correction_2).subtract(sigma_dp**2).clamp_min(1e-8)
                p_state['exp_avg_sq'].mul_(beta2).add_(torch.norm(grad).pow(2).div(torch.numel(grad)), alpha= 1-beta2)
                p_state['kf_m_t'].lerp_(grad, weight = k_t)
                p_state['kf_d_t'] = -p.data.clone().to(p.data)

                m_t_hat = p_state['kf_m_t']/p_state['bias_correction_1']

                # begin update
                if weight_decay != 0:
                    p.data.mul_(1 - group['lr'] * weight_decay)
                
                p.data.add_(m_t_hat, alpha = -lr/(exp_avg_sq_hat.sqrt().item()))

                p_state['kf_d_t'].add_(p.data, alpha = 1)
        return loss
    