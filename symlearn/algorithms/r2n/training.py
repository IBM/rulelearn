import torch

import symlearn.algorithms.r2n.utilities as util


def train(dataloader, model, loss_fn, optimizer, temp, min_temp, decay_rate, coef, mv_reg=1e-4, multivariate=False):
    thresholdnet = model[0].LearningLayer
    rulenet = model[1]
    succes_rate, loop_loss = 0, 0

    for batch, ((X_cat, X_num), y) in enumerate(dataloader):
        x = [X_cat.float(), X_num.float()]
        pred = model(x).T

        loss = loss_fn(pred, y.float()) + coef * rulenet.get_penalty()
        if multivariate:
            loss += mv_reg * thresholdnet.get_penalty()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # Calculating metric
        succes_rate += util.binary_acc(pred, y).item()
        loop_loss += loss.item()
    # Cooling schedule
    temp = temp * decay_rate if temp > min_temp else temp
    if multivariate:
        thresholdnet.update_temp(temp)
    rulenet.update_temp(temp)

    return temp, loop_loss, succes_rate / len(dataloader)
