import torch
from torch.optim.optimizer import Optimizer, required
import torch.autograd as ta

class KFOptimizer(Optimizer):
    '''
        optimizer = KFOptimizer(model.parameters(), optimizer, sigma_H, sigma_g)
        def train(model, train_dl, optimizer, criterion, device = 'cpu', acc_step = 1, lr_scheduler = None):
            model.to(device)
            model.train()
            for t, (input, label) in enumerate(train_dl):
                input = input.to(device)
                label = label.to(device)
                def closure():
                    optimizer.zero_grad()
                    predict = model(input)
                    if not isinstance(predict, torch.Tensor):
                        predict = predict.logits
                    loss = criterion(predict, label)
                    loss.backward()
                    return loss, predict
                loss, predict = optimizer.finite_difference(closure)

                if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
    '''
    def __init__(self, params, optimizer, sigma_H=0, sigma_g=0):
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

    def finite_difference(self, closure = required):
        loss = closure()
        # loss.backward()
        first_loss = loss.clone().detach()
        for group in self.param_groups:
            sigma_g = group['sigma_g']
            sigma_H = group['sigma_H']
            for p in group['params']:
                if p.grad is None:
                    continue
                if 'kf_beta_t' not in self.state[p]:
                    self.state[p]['kf_beta_t'] = 1
                    self.state[p]['kf_d_t'] = torch.zeros_like(p.data).to(p.data)
                beta_t = self.state[p]['kf_beta_t'] + sigma_H**2
                k_t = beta_t/(beta_t + sigma_g**2 - sigma_H**2 )
                k_1 = (1-k_t)/k_t
                self.state[p]['kf_g_t'] = p.grad.clone().to(p.data)
                p.data.add_(self.state[p]['d_t'], alpha = -k_1)
        # self.zero_grad()
        loss = closure()
        # loss.backward()
        for group in self.param_groups:
            sigma_g = group['sigma_g']
            sigma_H = group['sigma_H']
            for p in group['params']:
                if p.grad is None:
                    continue
                beta_t = self.state[p]['kf_beta_t'] + sigma_H**2
                k_t = beta_t/(beta_t + sigma_g**2 - sigma_H**2)
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
            sigma_g = group['sigma_g']
            sigma_H = group['sigma_H']
            for p in group['params']:
                if p.grad is None:
                    continue
                beta_t = self.state[p]['kf_beta_t'] + sigma_H**2
                k_t = beta_t/(beta_t + sigma_g**2 - sigma_H**2)
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