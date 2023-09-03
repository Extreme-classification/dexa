import torch


def construct_optimizer(params, net):
    if params.optim == 'Adam':
        return torch.optim.SparseAdam(
            list(net.classifier.parameters()),
            lr=params.learning_rate)
    elif params.optim == 'AdamW':
        no_decay = ['bias', 'LayerNorm.weight']
        gp = [
            {'params': [p for n, p in net.named_parameters() if 'encoder' in n and not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in net.named_parameters() if 'encoder' in n and any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': net.transform_lbl.transform.weight, 'weight_decay': 0.0, 'lr': params.learning_rate}
            ]
        return torch.optim.AdamW(
            gp, 
            **{'lr': params.learning_rate, 'eps': 1e-06, 'amsgrad': True})
    else:
        raise NotImplementedError("")


def construct_schedular(params, optimizer):
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=params.warmup_steps,
        num_training_steps=params.num_epochs*(
            params.num_points/params.batch_size))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps,
                                    num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from
    the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        num = num_training_steps - current_step
        den = max(1, num_training_steps - num_warmup_steps)
        return max(0.0, float(num) / float(den))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
