This is an implementation of the Kalman Filter + DP optimizer.

Optimizer is in `KFSGD.py`

Code provides a optimizer wrapper:

`KFOptimizer(params, optimizer, sigma_H, sigma_g)`
* Initialization arguments:
    * params: trainable parameters, Dict()
    * optimizer: an optimizer
    * sigma_H: variance estimation of Hessian
    * sigma_g: variance estimation of gradient
* Additional attributes:
    * optimizer: original optimizer
    * prestep(): prediction step for Kalman Filter
    * hessian_d_product(): not used

How to use:

Step 1: define your own optimizer (and lr_scheduler)
Step 2: 
```python
from KFSGD import KFOptimizer
wrapped_optimizer = KFOptimizer(model.parameters(), optimizer, sigma_H, sigma_g)
```
Step 3: when training, call `wrapped_optimizer.prestep()` before the first step of gradient accumulation, i.e.
```python
for data in dataloader:
    if first_step_of_gradient_accumulation:
        wrapped_optimizer.prestep()
    # normal training steps
    if last_step_of_gradient_accumulation:
        wrapped_optimizer.step()
        wrapped_optimizer.zero_grad()
```

The optimizer is compatible with fastDP (works with .grad, and .private_grad)

Example code:
```sh
EPS=8
LR=0.003
TAG="Adam_CNN_NLR_${LR}_EPS_${EPS}"
python ./run_KFSGD.py \
    --tag ${TAG} --log_type file --log_freq 10 \
    --bs 5000 --mnbs 250 --data cifar10 --data_path ./data \
    --algo adam --lr ${LR} --epoch 80 --model cnn5 \
    --clipping --noise -1 --epsilon ${EPS}
```