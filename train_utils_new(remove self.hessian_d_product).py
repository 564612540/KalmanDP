import torch
import numpy as np
from tqdm import tqdm
from tqdm._utils import _term_move_up
import torch.autograd as ta

def noisy_train(model, train_dl, optimizer, criterion, log_file, device = 'cpu', epoch = -1, noise = 0, log_frequency = -1, acc_step = 1, lr_scheduler = None):
    model.to(device)
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    snr = 0
    for t, (input, label) in enumerate(train_dl):
        input = input.to(device)
        label = label.to(device)
        predict = model(input)
        if not isinstance(predict, torch.Tensor):
            predict = predict.logits
        loss = criterion(predict, label)
        loss.backward()

        for param in model.parameters():
          if hasattr(param,'grad'):
            param.orig_grad=param.grad.clone()
            param.orig_data=param.data.clone()
            if hasattr(param,'dt'):
              param.data=(param.data+param.dt).clone()
        optimizer.zero_grad()

        predict = model(input)
        if not isinstance(predict, torch.Tensor):
            predict = predict.logits
        loss = criterion(predict, label)
        loss.backward()
        for param in model.parameters():
          if hasattr(param,'grad'):
            optimizer.state[param]['Hd_t']=(param.grad-param.orig_grad).clone()
            param.data=param.orig_data.clone()
            param.grad=param.orig_grad.clone()

        # train_loss = train_loss*0.9 + loss.item()*0.1
        train_loss = loss.item()
        _, predicted = predict.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

        if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
            model, snr = add_noise(model, noise)
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

        if t==0 or (t+1)%(acc_step)== 0 or ((t + 1) == len(train_dl)):
            print('Epoch: %d:%d Train Loss: %.3f | Acc: %.3f%% (%d/%d) | SNR: %-.6f'% (epoch, t+1, train_loss, 100.*correct/total, correct, total, snr))
            if log_frequency>0 and ((t+1)%(acc_step*log_frequency) == 0 or t+1 == len(train_dl)):
                log_file.update([epoch, t],[100.*correct/total, train_loss, snr])

    return model
