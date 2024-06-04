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