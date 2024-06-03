import torch
from torch.optim.optimizer import Optimizer, required
import torch.autograd as ta

# def KF_prestep(model, closure):
#     loss = closure()
#     loss.backward()
#     for param in model.parameters():
#         # perturb
#         if hasattr(param,'grad'):
#             param.orig_grad=param.grad.clone()
#             param.orig_data=param.data.clone()
#             if hasattr(param, 'dt'):
#                 param.data = 



class KFOptimizer(Optimizer):
    def __init__(self, params, optimizer, sigma_q=0, sigma_p=0):
        defaults = dict(sigma_q=sigma_q, sigma_p=sigma_p)
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

    def finite_difference(self, closure):
        loss = closure()
        loss.backward()
        first_loss = loss.clone().detach()
        for group in self.param_groups:
            sigma_p = group['sigma_p']
            sigma_q = group['sigma_q']
            for p in group['params']:
                if p.grad is None:
                    continue
                if 'kf_beta_t' not in self.state[p]:
                    self.state[p]['kf_beta_t'] = 1
                    self.state[p]['kf_d_t'] = torch.zeros_like(p.data).to(p.data)
                beta_t = self.state[p]['kf_beta_t'] + sigma_q**2
                k_t = beta_t/(beta_t + sigma_p**2 - sigma_q**2 )
                k_1 = (1-k_t)/k_t
                self.state[p]['kf_g_t'] = p.grad.clone().to(p.data)
                p.data.add_(self.state[p]['d_t'], alpha = -k_1)
        self.zero_grad()
        loss = closure()
        loss.backward()
        for group in self.param_groups:
            sigma_p = group['sigma_p']
            sigma_q = group['sigma_q']
            for p in group['params']:
                if p.grad is None:
                    continue
                beta_t = self.state[p]['kf_beta_t'] + sigma_q**2
                k_t = beta_t/(beta_t + sigma_p**2 - sigma_q**2)
                k_1 = (1-k_t)/k_t
                
                p.data.add_(self.state[p]['d_t'], alpha = k_1)
                self.state[p]['kf_m_t'].lerp_(p.grad, weight = self.state[p]['k_t'])
                self.state[p]['kf_g_t'] = None
        return first_loss

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # loss = None
        # if closure is not None:
        #     loss = closure()

        for group in self.param_groups:
            sigma_p = group['sigma_p']
            sigma_q = group['sigma_q']
            for p in group['params']:
                if p.grad is None:
                    continue
                beta_t = self.state[p]['kf_beta_t'] + sigma_q**2
                k_t = beta_t/(beta_t + sigma_p**2 - sigma_q**2)
                self.state[p]['kf_beta_t'] = (1-k_t)*beta_t
                p.grad = self.state[p]['kf_m_t'].clone().to(p.data)
                self.state[p]['kf_d_t'] = -p.data.clone().to(p.data)
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['kf_d_t'].add_(p.data, alpha = 1)
        return loss