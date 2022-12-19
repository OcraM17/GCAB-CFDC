# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import argparse
import copy
import datetime
import json
import os
import statistics
import time
import warnings
from pathlib import Path
import yaml

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from continuum.metrics import Logger
from continuum.tasks import split_train_val
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import continual.utils as utils
from continual import factory, scaler
from continual.classifier import Classifier
from continual.datasets import build_dataset
from continual.engine_GCAB import eval_and_log, train_one_epoch
from continual.losses import bce_with_logits
from torch import nn

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)


def get_args_parser():
    parser = argparse.ArgumentParser('GCAB-FDC training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--incremental-batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--base-epochs', default=500, type=int,
                        help='Number of epochs for base task')
    parser.add_argument('--no-amp', default=False, action='store_true',
                        help='Disable mixed precision')

    # Model parameters
    parser.add_argument('--model', default='')
    parser.add_argument('--input-size', default=32, type=int, help='images input size')
    parser.add_argument('--patch-size', default=16, type=int)
    parser.add_argument('--embed-dim', default=768, type=int)
    parser.add_argument('--depth', default=12, type=int)
    parser.add_argument('--num-heads', default=12, type=int)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--norm', default='layer', choices=['layer', 'scale'],
                        help='Normalization layer type')

    parser.add_argument('--thres_cosh', type=float, default=0.1)
    parser.add_argument('--thres_emb', type=float, default=0.1)
    parser.add_argument('--smax', type=float, default=0.1)
    parser.add_argument('--lambda_gcab', type=float, default=0.1)
    parser.add_argument('--lambda_pfr', type=float, default=0.1)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--incremental-lr", default=None, type=float,
                        help="LR to use for incremental task (t > 0)")
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--incremental-warmup-lr', type=float, default=None, metavar='LR',
                        help='warmup learning rate (default: 1e-6) for task T > 0')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    parser.add_argument('--data-path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--output-dir', default='',
                        help='Dont use that')
    parser.add_argument('--output-basedir', default='./checkponts/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # Continual Learning parameters
    parser.add_argument("--initial-increment", default=50, type=int,
                        help="Base number of classes")
    parser.add_argument("--increment", default=10, type=int,
                        help="Number of new classes per incremental task")
    parser.add_argument('--class-order', default=None, type=int, nargs='+',
                        help='Class ordering, a list of class ids.')

    parser.add_argument("--eval-every", default=50, type=int,
                        help="Eval model every X epochs, if None only eval at the task end")
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Only do one batch per epoch')
    parser.add_argument('--max-task', default=None, type=int,
                        help='Max task id to train on')
    parser.add_argument('--name', default='', help='Name to display for screen')
    parser.add_argument('--options', default=[], nargs='*')

    parser.add_argument('--GCAB', action='store_true', default=False,
                        help='Enable super DyTox god mode.')
    parser.add_argument('--ind-clf', default='', choices=['1-1', '1-n', 'n-n', 'n-1'],
                        help='Independent classifier per task but predicting all seen classes')
    parser.add_argument('--joint-tokens', default=False, action='store_true',
                        help='Forward w/ all task tokens alltogether [Faster but not working as well, not sure why')

    parser.add_argument('--freeze-task', default=[], nargs="*", type=str,
                        help='What to freeze before every incremental task (t > 0).')
    parser.add_argument('--freeze-ft', default=[], nargs="*", type=str,
                        help='What to freeze before every finetuning (t > 0).')
    parser.add_argument('--freeze-eval', default=False, action='store_true',
                        help='Frozen layers are put in eval. Important for stoch depth')

    # Convit - CaiT
    parser.add_argument('--local-up-to-layer', default=10, type=int,
                        help='number of GPSA layers')
    parser.add_argument('--locality-strength', default=1., type=float,
                        help='Determines how focused each head is around its attention center')
    parser.add_argument('--class-attention', default=False, action='store_true',
                        help='Freeeze and Process the class token as done in CaiT')

    # Logs
    parser.add_argument('--log-path', default="logs")
    parser.add_argument('--log-category', default="misc")
    parser.add_argument('--compress', default=None)
    parser.add_argument('--report', default=None)

    # Classification
    parser.add_argument('--bce-loss', default=False, action='store_true')

    # distributed training parameters
    parser.add_argument('--local_rank', default=None, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Resuming
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-task', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--save-every-epoch', default=None, type=int)

    parser.add_argument('--validation', default=0.0, type=float,
                        help='Use % of the training set as val, replacing the test.')

    return parser


def main(args):
    args.num_workers = 0
    logger = Logger(list_subsets=['train', 'test'])

    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    scenario_train, args.nb_classes = build_dataset(is_train=True, args=args)
    scenario_val, _ = build_dataset(is_train=False, args=args)

    model = factory.get_backbone(args)
    model.head = Classifier(
        model.embed_dim, args.nb_classes, args.initial_increment,
        args.increment, len(scenario_train)
    )
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)

    if args.name:
        log_path = os.path.join(args.log_dir, f"logs_{args.trial_id}.json")
        long_log_path = os.path.join(args.log_dir, f"long_logs_{args.trial_id}.json")

        if utils.is_main_process():
            os.system("echo '\ek{}\e\\'".format(args.name))
            os.makedirs(args.log_dir, exist_ok=True)
            with open(os.path.join(args.log_dir, f"config_{args.trial_id}.json"), 'w+') as f:
                config = vars(args)
                config["nb_parameters"] = n_parameters
                json.dump(config, f, indent=2)
            with open(log_path, 'w+') as f:
                pass
            with open(long_log_path, 'w+') as f:
                pass
        log_store = {'results': {}}

        args.output_dir = os.path.join(args.output_basedir,
                                       f"{datetime.datetime.now().strftime('%y-%m-%d')}_{args.name}_{args.trial_id}")
    else:
        log_store = None
        log_path = long_log_path = None
    if args.output_dir and utils.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print('number of params:', n_parameters)

    loss_scaler = scaler.ContinualScaler(args.no_amp)

    criterion = bce_with_logits
    teacher_model = None

    output_dir = Path(args.output_dir)

    nb_classes = args.initial_increment
    base_lr = args.lr
    accuracy_list = []
    start_time = time.time()

    args.increment_per_task = [args.initial_increment] + [args.increment for _ in range(len(scenario_train) - 1)]

    dataset_true_val = None
    file_rep = open(args.compress, 'a+')
    for task_id, dataset_train in enumerate(scenario_train):
        if args.max_task == task_id:
            print(f"Stop training because of max task")
            break
        print(f"Starting task id {task_id}/{len(scenario_train) - 1}")

        dataset_val = scenario_val[:task_id + 1]
        if args.validation > 0.:
            if task_id == 0:
                dataset_train, dataset_val = split_train_val(dataset_train, args.validation)
                dataset_true_val = dataset_val
            else:
                dataset_train, dataset_val = split_train_val(dataset_train, args.validation)
                dataset_true_val.concat(dataset_val)
            dataset_val = dataset_true_val

        for i in range(3):
            assert abs(dataset_train.trsf.transforms[-1].mean[i] - dataset_val.trsf.transforms[-1].mean[i]) < 0.0001
            assert abs(dataset_train.trsf.transforms[-1].std[i] - dataset_val.trsf.transforms[-1].std[i]) < 0.0001

        if task_id > 0:
            teacher_model = copy.deepcopy(model_without_ddp)
            teacher_model.freeze(['all'])
            teacher_model.eval()
            teacher_model = teacher_model.to('cuda')

        if args.GCAB:
            model_without_ddp = factory.update_gcab(model_without_ddp, task_id, len(scenario_train), args)

        print("Adding new parameters")
        if task_id > 0 and not args.GCAB:
            model_without_ddp.head.add_classes()

        if task_id > 0:
            model_without_ddp.freeze(args.freeze_task)
            model_without_ddp.tabs[0].eval()
            for n, p in model_without_ddp.tabs[0].named_parameters():
                if n.startswith('norm'):
                    p.requires_grad = False
                    p.grad = None

            for i in range(len(model_without_ddp.sabs)):
                model_without_ddp.sabs[i].eval()
                for n, p in model_without_ddp.sabs[i].named_parameters():
                    if n.startswith('norm'):
                        p.requires_grad = False
                        p.grad = None

            model_without_ddp.tabs[0].attn.proj.weight.requires_grad = True
            model_without_ddp.tabs[0].attn.proj.bias.requires_grad = True
            model_without_ddp.tabs[0].attn.k.weight.requires_grad = True
            model_without_ddp.tabs[0].attn.v.weight.requires_grad = True
            model_without_ddp.tabs[0].attn.q.weight.requires_grad = True
            model_without_ddp.tabs[0].mlp.fc1.weight.requires_grad = True
            model_without_ddp.tabs[0].mlp.fc1.bias.requires_grad = True
            model_without_ddp.tabs[0].mlp.fc2.weight.requires_grad = True
            model_without_ddp.tabs[0].mlp.fc2.bias.requires_grad = True
            model_without_ddp.embs_2.requires_grad = True
            model_without_ddp.embs_0.requires_grad = True
            model_without_ddp.head[-1].norm.requires_grad = False
            model_without_ddp.head[-1].norm.eval()
            model_without_ddp.projectors.append(
                nn.Sequential(nn.Conv2d(384, 384, (1, 1)), nn.LayerNorm((384, 1, 64)), nn.ReLU(),
                              nn.Conv2d(384, 384, (1, 1))).to('cuda'))

        loader_train, loader_val = factory.get_loaders(dataset_train, dataset_val, args)
        if task_id > 0 and args.incremental_batch_size:
            args.batch_size = args.incremental_batch_size

        if args.incremental_lr is not None and task_id > 0:
            linear_scaled_lr = args.incremental_lr * args.batch_size * utils.get_world_size() / 512.0
        else:
            linear_scaled_lr = base_lr * args.batch_size * utils.get_world_size() / 512.0

        args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, model_without_ddp)
        lr_scheduler, _ = create_scheduler(args, optimizer)

        skipped_task = False
        initial_epoch = epoch = 0
        if args.resume and args.start_task > task_id:
            utils.load_first_task_model(model_without_ddp, loss_scaler, task_id, args)
            print("Skipping first task")
            epochs = 0
            train_stats = {"task_skipped": str(task_id)}
            skipped_task = True
        elif args.base_epochs is not None and task_id == 0:
            epochs = args.base_epochs
        else:
            epochs = args.epochs

        if args.distributed:
            del model
            model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu],
                                                              find_unused_parameters=True)
        else:
            model = model_without_ddp

        model_without_ddp.nb_epochs = epochs
        model_without_ddp.nb_batch_per_epoch = len(loader_train)

        print(f"Start training for {epochs - initial_epoch} epochs")
        max_accuracy = 0.0
        for epoch in range(initial_epoch, epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, criterion, loader_train,
                optimizer, device, epoch, task_id, loss_scaler,
                args.clip_grad,
                debug=args.debug,
                args=args,
                teacher_model=teacher_model,
                model_without_ddp=model_without_ddp,
            )
            lr_scheduler.step(epoch)

            if args.save_every_epoch is not None and epoch % args.save_every_epoch == 0:
                if os.path.isdir(args.resume):
                    with open(os.path.join(args.resume, 'save_log.txt'), 'w+') as f:
                        f.write(f'task={task_id}, epoch={epoch}\n')

                    checkpoint_paths = [os.path.join(args.resume, f'checkpoint_{task_id}.pth')]
                    for checkpoint_path in checkpoint_paths:
                        if (task_id < args.start_task and args.start_task > 0) and os.path.isdir(
                                args.resume) and os.path.exists(checkpoint_path):
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

            if args.eval_every and epoch % args.eval_every == 0:
                eval_and_log(
                    args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                    epoch, task_id, loss_scaler, max_accuracy,
                    [], n_parameters, device, loader_val, train_stats, None, long_log_path,
                    logger, model_without_ddp.epoch_log()
                )
                logger.end_epoch()

        eval_and_log(
            args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
            epoch, task_id, loss_scaler, max_accuracy,
            accuracy_list, n_parameters, device, loader_val, train_stats, log_store, log_path,
            logger, model_without_ddp.epoch_log(), skipped_task
        )
        logger.end_task()

        nb_classes += args.increment

        task = torch.autograd.Variable(torch.LongTensor([task_id]).cuda(), volatile=False)
        masks = []
        with torch.no_grad():
            masks.append(
                torch.where(model_without_ddp.gate(model_without_ddp.smax * model_without_ddp.embs_0(task)) > 0.5, 1.0,
                            0.0))
            masks[-1].requires_grad = False
            masks[-1].grad = None
            masks.append(
                torch.where(model_without_ddp.gate(model_without_ddp.smax * model_without_ddp.embs_2(task)) > 0.5, 1.0,
                            0.0))
            masks[-1].requires_grad = False
            masks[-1].grad = None
        if task_id > 0:
            model_without_ddp.projectors[-1].eval()
            for n, p in model_without_ddp.projectors[-1].named_parameters():
                p.requires_grad = False
                p.grad = None

        if task == 0:
            model_without_ddp.mask_pre = masks
        else:
            for i in range(len(model_without_ddp.mask_pre)):
                model_without_ddp.mask_pre[i] = torch.max(model_without_ddp.mask_pre[i], masks[i])
        val_compr = []
        for i in range(len(model_without_ddp.mask_pre)):
            val_compr.append(model_without_ddp.mask_pre[i].sum() / model_without_ddp.mask_pre[i].numel())
        file_rep.write('TASK ' + str(task_id) + ': ' + (str(sum(val_compr).item() / len(val_compr))) + '\n')

        model_without_ddp.mask_back = {}
        for n, _ in model_without_ddp.named_parameters():
            vals = model_without_ddp.get_view_for(n, model_without_ddp.mask_pre)
            if vals is not None:
                model_without_ddp.mask_back[n] = 1 - vals

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print(f'Setting {args.data_set} with {args.initial_increment}-{args.increment}')
    print(f"All accuracies: {accuracy_list}")
    print(f"Average Incremental Accuracy: {statistics.mean(accuracy_list)}")
    if args.name:
        print(f"Experiment name: {args.name}")
        log_store['summary'] = {"avg": statistics.mean(accuracy_list)}
        if log_path is not None and utils.is_main_process():
            with open(log_path, 'a+') as f:
                f.write(json.dumps(log_store['summary']) + '\n')


def load_options(args, options):
    varargs = vars(args)

    name = []
    for o in options:
        with open(o) as f:
            new_opts = yaml.safe_load(f)

        for k, v in new_opts.items():
            if k not in varargs:
                raise ValueError(f'Option {k}={v} doesnt exist!')
        varargs.update(new_opts)
        name.append(o.split("/")[-1].replace('.yaml', ''))

    return '_'.join(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DyTox training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)

    if args.options:
        name = load_options(args, args.options)
        if not args.name:
            args.name = name

    args.log_dir = os.path.join(
        args.log_path, args.data_set.lower(), args.log_category,
        datetime.datetime.now().strftime('%y-%m'),
        f"week-{int(datetime.datetime.now().strftime('%d')) // 7 + 1}",
        f"{int(datetime.datetime.now().strftime('%d'))}_{args.name}"
    )

    if isinstance(args.class_order, list) and isinstance(args.class_order[0], list):
        print(f'Running {len(args.class_order)} different class orders.')
        class_orders = copy.deepcopy(args.class_order)

        for i, order in enumerate(class_orders, start=1):
            print(f'Running class ordering {i}/{len(class_orders)}.')
            args.trial_id = i
            args.class_order = order
            main(args)
    else:
        args.trial_id = 1
        main(args)
