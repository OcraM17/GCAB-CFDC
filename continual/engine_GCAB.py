# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import json
import os
import math
from typing import Iterable
import torch
from timm.utils import accuracy
from timm.loss import SoftTargetCrossEntropy
import numpy as np
import continual.utils as utils
from continual.losses import DistillationLoss

CE = SoftTargetCrossEntropy()


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, task_id: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True, debug=False, args=None,
                    teacher_model: torch.nn.Module = None,
                    model_without_ddp: torch.nn.Module = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Task: [{}] Epoch: [{}]'.format(task_id, epoch)
    print_freq = 10

    for batch_index, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if batch_index == 0:
            print(f'Image size is {samples.shape}.')

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        s = (model.smax - 1 / model.smax) * batch_index / len(data_loader) + 1 / model.smax
        with torch.cuda.amp.autocast(enabled=not args.no_amp):
            loss_tuple = forward(samples, targets, model, s, teacher_model, criterion)

        loss = sum(filter(lambda x: x is not None, loss_tuple))
        internal_losses = model_without_ddp.get_internal_losses(loss)
        for internal_loss_value in internal_losses.values():
            loss += internal_loss_value

        check_loss(loss)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        loss_scaler(loss, optimizer, s, task_id, model, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update_dict(internal_losses)
        metric_logger.update(loss=loss_tuple[0])
        metric_logger.update(gcab=loss_tuple[1])
        metric_logger.update(pfr=loss_tuple[2])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if debug:
            print('Debug, only doing one epoch!')
            break

    if hasattr(model_without_ddp, 'hook_after_epoch'):
        model_without_ddp.hook_after_epoch()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def check_loss(loss):
    if not math.isfinite(loss.item()):
        raise Exception('Loss is {}, stopping training'.format(loss.item()))


def forward(samples, targets, model, s, teacher_model, criterion):

    outputs, masks, feats = model(samples, s, None)
    main_output = outputs

    loss = criterion(main_output, targets)

    reg = 0
    count = 0
    if model.mask_pre is not None:
        for m, mp in zip(masks, model.mask_pre):
            aux = 1 - mp
            reg += (m * aux).sum()
            count += aux.sum()
    else:
        for m in masks:
            reg += m.sum()
            count += np.prod(m.size()).item()
    reg /= count
    reg *= model.lambda_gcab
    l2 = None
    if teacher_model is not None:
        with torch.no_grad():
            teacher_outputs = teacher_model.forward_SAB(samples)
        feats = model.projectors[-1](torch.permute(feats.unsqueeze(2), (0, 3, 2, 1)))
        feats = torch.permute(feats.squeeze(2), (0, 2, 1))
        l2 = torch.norm(teacher_outputs - feats, p=2) * model.lambda_pfr
    return loss, reg, l2


@torch.no_grad()
def evaluate(data_loader, model, device, logger, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    file_tag = open(args.report + '_tag.txt', 'a+')
    file_taw = open(args.report + '_taw.txt', 'a+')
    file_cumul = open(args.report + '_cumul.txt', 'a+')
    model.eval()
    task_vals = {i: [[], 0] for i in range(len(model.nb_classes_per_task))}
    task_ag = {i: [[], 0] for i in range(len(model.nb_classes_per_task))}
    task_cumul = {i: [[], 0] for i in range(len(model.nb_classes_per_task))}
    for images, target, task_ids in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            output, masks, feats = model(images, s=model.smax, inference=True)
            if isinstance(output, dict):
                output = output['logits']
            loss = criterion(output, target)
        for t_id in range(len(model.nb_classes_per_task)):
            subset = torch.logical_and(target >= (sum(model.nb_classes_per_task[:t_id])),
                                       target < (sum(model.nb_classes_per_task[:t_id + 1])))
            task_vals[t_id][0].append(((torch.argmax(
                output[subset][:, (sum(model.nb_classes_per_task[:t_id])):(sum(model.nb_classes_per_task[:t_id + 1]))],
                1) + (sum(model.nb_classes_per_task[:t_id])))
                                       == target[subset]).sum())
            task_ag[t_id][0].append((torch.argmax(output[subset], 1) == target[subset]).sum())

            task_vals[t_id][1] += target[subset].shape[0]
            task_ag[t_id][1] += target[subset].shape[0]

            subset_cumulative = target < sum(model.nb_classes_per_task[:t_id + 1])
            task_cumul[t_id][0].append((torch.argmax(output[subset_cumulative], 1) == target[subset_cumulative]).sum())
            task_cumul[t_id][1] += target[subset_cumulative].shape[0]

        acc1, acc5 = accuracy(output, target, topk=(1, min(5, output.shape[1])))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        logger.add([output.cpu().argmax(dim=1), target.cpu(), task_ids], subset='test')
    print('--TAW--')
    for key, val in task_vals.items():
        print('TASK:', key, 'ACC:', np.round((sum(val[0]) / val[1]).cpu().item() * 100, 2))
        file_taw.write('TASK:' + str(key) + ' ACC:' + str(np.round((sum(val[0]) / val[1]).cpu().item() * 100, 2)) + ' ')
    print('--TAG--')
    for key, val in task_ag.items():
        print('TASK:', key, 'ACC:', np.round((sum(val[0]) / val[1]).cpu().item() * 100, 2))
        file_tag.write('TASK:' + str(key) + ' ACC:' + str(np.round((sum(val[0]) / val[1]).cpu().item() * 100, 2)) + ' ')
    print('--CUMULATIVE--')
    for key, val in task_cumul.items():
        print('TASK:', key, 'ACC:', np.round((sum(val[0]) / val[1]).cpu().item() * 100, 2))
        file_tag.write('TASK:' + str(key) + ' ACC:' + str(np.round((sum(val[0]) / val[1]).cpu().item() * 100, 2)) + ' ')
    file_taw.write('\n')
    file_tag.write('\n')
    file_cumul.write('\n')
    file_taw.close()
    file_tag.close()
    file_cumul.close()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_and_log(args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                 epoch, task_id, loss_scaler, max_accuracy, accuracy_list,
                 n_parameters, device, data_loader_val, train_stats, log_store, log_path, logger,
                 model_log, skipped_task=False):
    if args.output_dir:
        if os.path.isdir(args.resume):
            checkpoint_paths = [os.path.join(args.resume, f'checkpoint_{task_id}.pth')]
        else:
            checkpoint_paths = [output_dir / f'checkpoint_{task_id}.pth']
        for checkpoint_path in checkpoint_paths:
            if skipped_task:
                continue

            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'task_id': task_id,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)

    test_stats = evaluate(data_loader_val, model, device, logger, args)
    print(f"Accuracy of the network on the {len(data_loader_val.dataset)} test images: {test_stats['acc1']:.1f}%")
    max_accuracy = max(max_accuracy, test_stats["acc1"])
    print(f'Max accuracy: {max_accuracy:.2f}%')
    accuracy_list.append(test_stats['acc1'])

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                 **{f'test_{k}': v for k, v in test_stats.items()},
                 'epoch': epoch,
                 'n_parameters': n_parameters}

    mean_acc5 = -1.0
    if log_store is not None:
        log_store['results'][task_id] = log_stats
        all_acc5 = [task_log['test_acc5'] for task_log in log_store['results'].values()]
        mean_acc5 = sum(all_acc5) / len(all_acc5)

    if log_path is not None and utils.is_main_process():
        with open(log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'epoch': epoch,
                'acc': round(100 * logger.accuracy, 2),
                'avg_acc': round(100 * logger.average_incremental_accuracy, 2),
                'forgetting': round(100 * logger.forgetting, 6),
                'acc_per_task': [round(100 * acc_t, 2) for acc_t in logger.accuracy_per_task],
                'train_lr': log_stats.get('train_lr', 0.),
                'bwt': round(100 * logger.backward_transfer, 2),
                'fwt': round(100 * logger.forward_transfer, 2),
                'test_acc1': round(log_stats['test_acc1'], 2),
                'test_acc5': round(log_stats['test_acc5'], 2),
                'mean_acc5': round(mean_acc5, 2),
                'train_loss': round(log_stats.get('train_loss', 0.), 5),
                'test_loss': round(log_stats['test_loss'], 5),
                **model_log
            }) + '\n')
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

    return max_accuracy


def indexes_task_outputs(logits, targets, increment_per_task):
    if increment_per_task[0] != increment_per_task[1]:
        raise NotImplementedError(f'Not supported yet for non equal task size')

    inc = increment_per_task[0]
    indexes = torch.zeros(len(logits), inc).long()

    for r in range(indexes.shape[0]):
        for c in range(indexes.shape[1]):
            indexes[r, c] = (targets[r] // inc) * inc + r * logits.shape[1] + c

    indexed_logits = logits.view(-1)[indexes.view(-1)].view(len(logits), inc)
    indexed_targets = targets % inc

    return indexed_logits, indexed_targets
