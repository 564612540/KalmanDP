import torch
from torch.optim.optimizer import Optimizer, required
import torch.autograd as ta

class KFSGD(Optimizer):
    def __init__(self, params, lr=required, weight_decay=0, sigma_q=0, sigma_p=0):
        defaults = dict(lr=lr,
                        weight_decay=weight_decay, sigma_q=sigma_q, sigma_p=sigma_p)
        # if nesterov and (momentum <= 0 or dampening != 0):
            # raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(KFSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(KFSGD, self).__setstate__(state)

    def hessian_d_product(self, closure=None):
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


    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        printed = False
        self.hessian_d_product()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            # H = group['H']
            sigma_p = group['sigma_p']
            sigma_q = group['sigma_q']
            g_list = []
            m_list = []
            Hd_list = []
            beta_t = None
            norm_fact = None
            # concat as a vector
            for p in group['params']:
                if p.grad is None:
                    continue
                if beta_t is None:
                    if 'beta_t' not in self.state[p]:
                        self.state[p]['beta_t'] = 1
                    if 'norm_fact' not in self.state[p]:
                        self.state[p]['norm_fact'] = 0
                    beta_t = self.state[p]['beta_t'] + sigma_q**2
                    norm_fact = self.state[p]['norm_fact']
                    # H = H.to(p)
                g_list.append(p.grad.data.view(-1))
                if 'm_t' not in self.state[p]:
                    self.state[p]['m_t'] = p.grad.clone().to(p.grad)# torch.zeros_like(p.grad).reshape(-1).to(p.grad) # 
                m_list.append(self.state[p]['m_t'].view(-1))
                # if 'd_t' not in self.state[p]:
                #     self.state[p]['d_t'] = torch.zeros_like(p.grad).to(p.grad)
                # print(self.state[p]['Hd_t'].size())
                # print("Manual Hd: ", torch.mm(H, self.state[p]['d_t'].view(-1,1)))
                Hd_list.append(self.state[p]['Hd_t'].reshape(-1))
            g_vector = torch.cat(g_list)
            m_vector = torch.cat(m_list)
            Hd_vector = torch.cat(Hd_list)

            # prediction
            m_vector.add_(Hd_vector, alpha=-1)
            # delta_g = g_vector - m_vector
            # print("delta_g:", torch.norm(delta_g).item())

            k_t = beta_t/(beta_t + sigma_p**2 - sigma_q**2)
            if not printed:
                print(k_t)
                printed = True
                
            # filter
            m_vector.lerp_(g_vector, k_t)
            norm_fact = 1 #(1-k_t)*norm_fact + k_t
            beta_t = (1-k_t)*beta_t

            offset = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                if 'beta_t' in self.state[p]:
                    self.state[p]['beta_t'] = beta_t
                    self.state[p]['norm_fact'] = norm_fact
                param_num = torch.numel(p)
                self.state[p]['m_t'] = m_vector[offset:offset+param_num].view_as(p)
                offset += param_num

                self.state[p]['d_t'] = p.data.clone().to(p.data)
                if weight_decay != 0:
                    p.data.mul_(1 - group['lr'] * weight_decay)
                p.data.add_(self.state[p]['m_t'], alpha = -group['lr']/norm_fact)
                self.state[p]['d_t'].add_(p.data, alpha=-1)
                self.state[p]['Hd_t'] = None
        return loss