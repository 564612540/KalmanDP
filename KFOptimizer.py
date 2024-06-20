import torch
from torch.optim.optimizer import Optimizer, required
import torch.autograd as ta

class KFOptimizer(Optimizer):
    def __init__(self, params, optimizer:Optimizer, sigma_H=3e-6, sigma_g=1e-5):
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
        super(KFOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(KFOptimizer, self).__setstate__(state)

    def hessian_d_product(self):
        """
        evaluate hessian vector product
        """
        loss = 0
        G = []
        X = []
        D = []
        for group in self.param_groups:
            for p in group['params']:
                # if p.grad is None:
                #     continue
                if p.requires_grad:
                    G.append(p.grad)
                    # print(p.grad)
                    X.append(p)
                    if 'd_t' not in self.state[p]:
                        self.state[p]['d_t'] = torch.zeros_like(p.grad).to(p.grad)
                    D.append(self.state[p]['d_t'])
        Hd = ta.grad(G, X, D, retain_graph=False)
        # print(Hd)
        Hd = list(Hd)
        # print("Hd: ", Hd)
        # blk_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if 'Hd_t' not in self.state[p] or self.state[p]['Hd_t'] is None:
                    self.state[p]['Hd_t'] = Hd.pop(0)
                else:
                    self.state[p]['Hd_t'].add_(Hd.pop(0))
        return Hd

    def prestep(self):
        for group in self.param_groups:
            sigma_g = group['sigma_g']
            sigma_H = group['sigma_H']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                if 'kf_beta_t' not in self.state[p]:
                    continue
                beta_t = self.state[p]['kf_beta_t'] + sigma_H**2
                k_t = beta_t/(beta_t + sigma_g**2 - sigma_H**2 )
                k_1 = (1-k_t)/k_t
                p.data.add_(self.state[p]['kf_d_t'], alpha = k_1)

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
                if hasattr(p, 'private_grad'):
                    grad = p.private_grad
                    has_private_grad = True
                elif p.grad is not None:
                    grad = p.grad
                else:
                    continue
                if 'kf_beta_t' not in self.state[p]:
                    self.state[p]['kf_beta_t'] = 1
                    self.state[p]['kf_d_t'] = torch.zeros_like(p.data).to(p.data)
                    self.state[p]['kf_m_t'] = grad.clone().to(p.data)
                beta_t = self.state[p]['kf_beta_t'] + sigma_H**2
                k_t = beta_t/(beta_t + sigma_g**2 - sigma_H**2)
                k_1 = (1-k_t)/k_t
                self.state[p]['kf_beta_t'] = (1-k_t)*beta_t
                p.data.add_(self.state[p]['kf_d_t'], alpha = -k_1)
                self.state[p]['kf_m_t'].lerp_(grad, weight = k_t)
                if has_private_grad:
                    p.private_grad = self.state[p]['kf_m_t'].clone().to(p.data)
                else:
                    p.grad = self.state[p]['kf_m_t'].clone().to(p.data)
                self.state[p]['kf_d_t'] = -p.data.clone().to(p.data)
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            sigma_g = group['sigma_g']
            sigma_H = group['sigma_H']
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['kf_d_t'].add_(p.data, alpha = 1)
                # beta_t = self.state[p]['kf_beta_t'] + sigma_H**2
                # k_t = beta_t/(beta_t + sigma_g**2 - sigma_H**2 )
                # k_1 = (1-k_t)/k_t
                # p.data.add_(self.state[p]['kf_d_t'], alpha = k_1)
        return loss
    

        # def finite_difference(self, closure = required):
    #     loss = closure()
    #     # loss.backward()
    #     first_loss = loss
    #     for group in self.param_groups:
    #         sigma_g = group['sigma_g']
    #         sigma_H = group['sigma_H']
    #         for p in group['params']:
    #             if p.grad is None:
    #                 continue
    #             if 'kf_beta_t' not in self.state[p]:
    #                 self.state[p]['kf_beta_t'] = 1
    #                 self.state[p]['kf_d_t'] = torch.zeros_like(p.data).to(p.data)
    #                 self.state[p]['kf_m_t'] = p.grad.clone().to(p.data)
    #             beta_t = self.state[p]['kf_beta_t'] + sigma_H**2
    #             k_t = beta_t/(beta_t + sigma_g**2 - sigma_H**2 )
    #             k_1 = (1-k_t)/k_t
    #             if 'kf_g_t' not in self.state[p] or self.state[p]['kf_g_t'] is None:
    #                 self.state[p]['kf_g_t'] = p.grad.clone().to(p.data)
    #                 self.state[p]['kf_step'] = 1
    #             else:
    #                 self.state[p]['kf_g_t'].add_(p.grad)
    #                 self.state[p]['kf_step'] += 1
    #             p.data.add_(self.state[p]['kf_d_t'], alpha = -k_1)
    #     # self.zero_grad()
    #     loss = closure()
    #     # loss.backward()
    #     for group in self.param_groups:
    #         sigma_g = group['sigma_g']
    #         sigma_H = group['sigma_H']
    #         for p in group['params']:
    #             if p.grad is None:
    #                 continue
    #             beta_t = self.state[p]['kf_beta_t'] + sigma_H**2
    #             k_t = beta_t/(beta_t + sigma_g**2 - sigma_H**2)
    #             k_1 = (1-k_t)/k_t
                
    #             p.data.add_(self.state[p]['kf_d_t'], alpha = k_1)
    #             self.state[p]['kf_m_t'].lerp_(p.grad, weight = self.state[p]['k_t'])
    #             self.state[p]['kf_g_t'] = None
    #     return first_loss
