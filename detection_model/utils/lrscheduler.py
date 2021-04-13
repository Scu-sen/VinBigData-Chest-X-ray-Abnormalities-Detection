from math import cos, pi
import numpy as np
from torch.optim.optimizer import Optimizer


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CyclicLR(_LRScheduler):
    """Cyclic LR Scheduler.

        Implement the cyclical learning rate policy (CLR) described in
        https://arxiv.org/pdf/1506.01186.pdf

        Different from the original paper, we use cosine anealing rather than
        triangular policy inside a cycle. This improves the performance in the
        3D detection area.

        Attributes:
            max_iters (int): maximum iteration of running
            target_ratio (tuple[float]): Relative ratio of the highest LR and the
                lowest LR to the initial LR.
            cyclic_times (int): Number of cycles during training
            step_ratio_up (float): The ratio of the increasing process of LR in
                the total cycle.
        """

    def __init__(self,
                 optimizer,
                 max_iters,
                 target_ratio=(10, 1e-4),
                 cyclic_times=1,
                 step_ratio_up=0.4,
                 last_epoch=-1):
        if isinstance(target_ratio, float):
            target_ratio = (target_ratio, target_ratio / 1e5)
        elif isinstance(target_ratio, tuple):
            target_ratio = (target_ratio[0], target_ratio[0] / 1e5) \
                if len(target_ratio) == 1 else target_ratio
        else:
            raise ValueError('target_ratio should be either float '
                             f'or tuple, got {type(target_ratio)}')

        assert len(target_ratio) == 2, \
            '"target_ratio" must be list or tuple of two floats'
        assert 0 <= step_ratio_up < 1.0, \
            '"step_ratio_up" must be in range [0,1)'

        self.target_ratio = target_ratio
        self.cyclic_times = cyclic_times
        self.step_ratio_up = step_ratio_up
        self.lr_phases = []  # init lr_phases
        self.optimizer = optimizer

        # initiate lr_phases
        # total lr_phases are separated as up and down
        max_iter_per_phase = max_iters // self.cyclic_times
        iter_up_phase = int(self.step_ratio_up * max_iter_per_phase)
        self.lr_phases.append(
            [0, iter_up_phase, max_iter_per_phase, 1, self.target_ratio[0]])
        self.lr_phases.append([
            iter_up_phase, max_iter_per_phase, max_iter_per_phase,
            self.target_ratio[0], self.target_ratio[1]
        ])

        super(CyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lr = []
        curr_iter = self.last_epoch
        # make sure that the length of base_lrs doesn't change. Dont care about the actual value
        for base_lr in self.base_lrs:
            for (start_iter, end_iter, max_iter_per_phase, start_ratio,
                 end_ratio) in self.lr_phases:  # up_phase or the cosine annealing phase
                curr_iter %= max_iter_per_phase
                if start_iter <= curr_iter < end_iter:
                    progress = curr_iter - start_iter
                    lr = annealing_cos(base_lr * start_ratio,
                                         base_lr * end_ratio,
                                         progress / (end_iter - start_iter))
                    new_lr.append(lr)
                    break
        return new_lr


def annealing_cos(start, end, factor, weight=1):
    """Calculate annealing cos learning rate.

    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out