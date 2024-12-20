import torch
import numpy as np
from tqdm import tqdm
from tqdm._utils import _term_move_up
import torch.autograd as ta

def train(model, train_dl, optimizer, criterion, log_file, epoch = -1, log_frequency = -1, acc_step = 1, lr_scheduler = None):
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    # print(" ")
    for t, (input, label) in enumerate(train_dl):
        input = input.to(model.local_rank)
        label = label.to(model.local_rank)
        def closure():
            predict = model(input)
            if not isinstance(predict, torch.Tensor):
                predict = predict.logits
            loss = criterion(predict, label)
            model.backward(loss)
            return loss, predict
        
        if hasattr(optimizer, 'prestep'):
            loss, predict = optimizer.prestep(closure)
        else:
            loss, predict = closure()
        
        train_loss = loss.item()
        _, predicted = predict.max(1)
        total = total + label.size(0)
        correct = correct + predicted.eq(label).sum().item()

        del input
        del label
        del loss
        del predict

        if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.step()
            # optimizer.prestep()
            optimizer.zero_grad()

        if (t+1)%(acc_step)== 0 or ((t + 1) == len(train_dl)):
            print('Epoch: %d:%d Train Loss: %.3f | Acc: %.3f%% (%d/%d)'% (epoch, t+1, train_loss, 100.*correct/total, correct, total))
            if log_frequency>0 and ((t+1)%(acc_step*log_frequency) == 0 or t+1 == len(train_dl)):
                log_file.update([epoch, t],[100.*correct/total, train_loss])


@torch.no_grad()
def test(model, test_dl, criterion, log_file, epoch = -1, **kwargs):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.to(model.local_rank), targets.to(model.local_rank)
            outputs = model(inputs)
            if not isinstance(outputs, torch.Tensor):
                outputs = outputs.logits
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('Epoch: ', epoch, 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print(" ")
    log_file.update([epoch, -1],[100.*correct/total, test_loss/(batch_idx+1)])

import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr