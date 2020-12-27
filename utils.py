import os
import errno
import math
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.mkdir(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:1].view(-1).float()

    return correct_k


def get_prime(images, patch_size, interpolation='bicubic'):
    """Get down-sampled original image"""
    prime = F.interpolate(images, size=[patch_size, patch_size], mode=interpolation, align_corners=True)
    return prime


def get_patch(images, action_sequence, patch_size):
    """Get small patch of the original image"""
    batch_size = images.size(0)
    image_size = images.size(2)

    patch_coordinate = torch.floor(action_sequence * (image_size - patch_size)).int()
    patches = []
    for i in range(batch_size):
        per_patch = images[i, :,
                    (patch_coordinate[i, 0].item()): ((patch_coordinate[i, 0] + patch_size).item()),
                    (patch_coordinate[i, 1].item()): ((patch_coordinate[i, 1] + patch_size).item())]

        patches.append(per_patch.view(1, per_patch.size(0), per_patch.size(1), per_patch.size(2)))

    return torch.cat(patches, 0)


def adjust_learning_rate(optimizer, train_configuration, epoch, training_epoch_num, args):
    """Sets the learning rate"""

    backbone_lr = 0.5 * train_configuration['backbone_lr'] * \
                  (1 + math.cos(math.pi * epoch / training_epoch_num))
    if args.train_stage == 1:
        fc_lr = 0.5 * train_configuration['fc_stage_1_lr'] * \
                (1 + math.cos(math.pi * epoch / training_epoch_num))
    elif args.train_stage == 3:
        fc_lr = 0.5 * train_configuration['fc_stage_3_lr'] * \
                (1 + math.cos(math.pi * epoch / training_epoch_num))

    if train_configuration['train_model_prime']:
        optimizer.param_groups[0]['lr'] = backbone_lr
        optimizer.param_groups[1]['lr'] = backbone_lr
        optimizer.param_groups[2]['lr'] = fc_lr
    else:
        optimizer.param_groups[0]['lr'] = backbone_lr
        optimizer.param_groups[1]['lr'] = fc_lr

    for param_group in optimizer.param_groups:
        print(param_group['lr'])


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = checkpoint + '/' + filename
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, checkpoint + '/model_best.pth.tar')