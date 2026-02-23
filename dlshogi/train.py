import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, update_bn

from dlshogi.common import *
from dlshogi.network.policy_value_network import policy_value_network
from dlshogi import serializers
from dlshogi.data_loader import Hcpe3DataLoader
from dlshogi.data_loader import DataLoader


import argparse
import sys
import re
import importlib
from pathlib import Path

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple


LossFn = Callable[[Any, Any, Any, Any, Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def setup_logging(log_path: str | None) -> None:
    if log_path:
        logging.basicConfig(
            format='%(asctime)s\t%(levelname)s\t%(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            filename=log_path,
            level=logging.DEBUG,
        )
    else:
        logging.basicConfig(
            format='%(asctime)s\t%(levelname)s\t%(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            stream=sys.stdout,
            level=logging.DEBUG,
        )


def resolve_device(gpu_id: int) -> torch.device:
    if gpu_id >= 0:
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def _parse_constructor_expr(expr: str) -> Tuple[str, Dict[str, Any]]:
    name, args_expr = expr.split('(', 1)
    kwargs = eval(f'dict({args_expr.rstrip(")")})')
    return name, kwargs


def create_optimizer(
    optimizer_expr: str,
    model_params,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    optimizer_name, optimizer_args = _parse_constructor_expr(optimizer_expr)
    if '.' in optimizer_name:
        module_name, class_name = optimizer_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        optimizer_class = getattr(module, class_name)
    else:
        optimizer_class = getattr(optim, optimizer_name)

    if weight_decay >= 0:
        optimizer_args['weight_decay'] = weight_decay

    optimizer = optimizer_class(model_params, lr=lr, **optimizer_args)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(
            f'Invalid optimizer type: {type(optimizer)}. '
            'Must be a subclass of torch.optim.Optimizer'
        )
    return optimizer


def create_scheduler(
    scheduler_expr: str,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler_name, scheduler_args = _parse_constructor_expr(scheduler_expr)
    if '.' in scheduler_name:
        module_name, class_name = scheduler_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        scheduler_class = getattr(module, class_name)
    else:
        scheduler_class = getattr(optim.lr_scheduler, scheduler_name)

    scheduler = scheduler_class(optimizer, **scheduler_args)
    if not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
        raise TypeError(
            f'Invalid scheduler type: {type(scheduler)}. '
            'Must be a subclass of torch.optim.lr_scheduler.LRScheduler'
        )
    return scheduler


def log_training_config(args: argparse.Namespace) -> None:
    logging.info('network {}'.format(args.network))
    logging.info('batchsize={}'.format(args.batchsize))
    logging.info('lr={}'.format(args.lr))
    logging.info('weight_decay={}'.format(args.weight_decay))
    if args.lr_scheduler:
        logging.info('lr_scheduler {}'.format(args.lr_scheduler))
    if args.use_critic:
        logging.info('use critic')
    if args.beta:
        logging.info('entropy regularization coeff={}'.format(args.beta))
    logging.info('val_lambda={}'.format(args.val_lambda))


def _restore_optimizer_hparams(
    optimizer: torch.optim.Optimizer,
    lr: float,
    weight_decay: float,
) -> None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        if weight_decay >= 0:
            param_group['weight_decay'] = weight_decay


def init_or_resume_state(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    swa_model: AveragedModel | None,
) -> Tuple[int, int]:
    if args.initmodel:
        # for compatibility
        logging.info('Loading the model from {}'.format(args.initmodel))
        serializers.load_npz(args.initmodel, model)

    if not args.resume:
        return 0, 0

    checkpoint = torch.load(args.resume, map_location=device)
    epoch = checkpoint['epoch']
    t = checkpoint['t']

    if 'model' in checkpoint:
        logging.info('Loading the checkpoint from {}'.format(args.resume))
        model.load_state_dict(checkpoint['model'])
        if args.use_swa and swa_model is not None and 'swa_model' in checkpoint:
            swa_model.load_state_dict(checkpoint['swa_model'])
        if not args.reset_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not args.lr_scheduler:
                _restore_optimizer_hparams(optimizer, args.lr, args.weight_decay)
        if args.use_amp and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        if (
            args.lr_scheduler
            and not args.reset_scheduler
            and scheduler is not None
            and 'scheduler' in checkpoint
        ):
            scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        # for compatibility
        logging.info('Loading the optimizer state from {}'.format(args.resume))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.use_amp and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return epoch, t


def build_checkpoint(
    epoch: int,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    swa_model: AveragedModel | None,
    use_swa: bool,
    swa_start_epoch: int,
) -> Dict[str, Any]:
    checkpoint: Dict[str, Any] = {
        'epoch': epoch,
        't': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
    }
    if use_swa and swa_model is not None and epoch >= swa_start_epoch:
        checkpoint['swa_model'] = swa_model.state_dict()
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    return checkpoint


@dataclass
class LossAccumulator:
    steps: int = 0
    loss1: float = 0.0
    loss2: float = 0.0
    loss3: float = 0.0
    loss: float = 0.0

    def add(self, loss1: float, loss2: float, loss3: float, loss: float) -> None:
        self.steps += 1
        self.loss1 += loss1
        self.loss2 += loss2
        self.loss3 += loss3
        self.loss += loss

    def merge(self, other: 'LossAccumulator') -> None:
        self.steps += other.steps
        self.loss1 += other.loss1
        self.loss2 += other.loss2
        self.loss3 += other.loss3
        self.loss += other.loss

    def averages(self) -> Tuple[float, float, float, float]:
        return (
            self.loss1 / self.steps,
            self.loss2 / self.steps,
            self.loss3 / self.steps,
            self.loss / self.steps,
        )

    def clear(self) -> None:
        self.steps = 0
        self.loss1 = 0.0
        self.loss2 = 0.0
        self.loss3 = 0.0
        self.loss = 0.0


@dataclass
class EvalAccumulator(LossAccumulator):
    accuracy1: float = 0.0
    accuracy2: float = 0.0
    entropy1: float = 0.0
    entropy2: float = 0.0

    def add_eval(
        self,
        loss1: float,
        loss2: float,
        loss3: float,
        loss: float,
        accuracy1: float,
        accuracy2: float,
        entropy1: float,
        entropy2: float,
    ) -> None:
        self.add(loss1, loss2, loss3, loss)
        self.accuracy1 += accuracy1
        self.accuracy2 += accuracy2
        self.entropy1 += entropy1
        self.entropy2 += entropy2

    def averages_with_metrics(self) -> Tuple[float, float, float, float, float, float, float, float]:
        loss1, loss2, loss3, loss = self.averages()
        return (
            loss1,
            loss2,
            loss3,
            loss,
            self.accuracy1 / self.steps,
            self.accuracy2 / self.steps,
            self.entropy1 / self.steps,
            self.entropy2 / self.steps,
        )


def accuracy(y, t):
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)


def binary_accuracy(y, t):
    pred = y >= 0
    truth = t >= 0.5
    return pred.eq(truth).sum().item() / len(t)


def evaluate_model(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    compute_losses_fn: LossFn,
) -> Tuple[float, float, float, float, float, float, float, float]:
    accum = EvalAccumulator()
    model.eval()
    with torch.no_grad():
        for x1, x2, t1, t2, value in test_dataloader:
            y1, y2 = model(x1, x2)

            loss1, loss2, loss3, loss = compute_losses_fn(y1, y2, t1, t2, value)
            entropy1 = (-F.softmax(y1, dim=1) * F.log_softmax(y1, dim=1)).sum(dim=1)
            p2 = y2.sigmoid()
            # entropy2 = -(p2 * F.log(p2) + (1 - p2) * F.log(1 - p2))
            log1p_ey2 = F.softplus(y2)
            entropy2 = -(p2 * (y2 - log1p_ey2) + (1 - p2) * -log1p_ey2)
            accum.add_eval(
                loss1.item(),
                loss2.item(),
                loss3.item(),
                loss.item(),
                accuracy(y1, t1),
                binary_accuracy(y2, t2),
                entropy1.mean().item(),
                entropy2.mean().item(),
            )

    return accum.averages_with_metrics()


def save_checkpoint_file(
    checkpoint_pattern: str,
    epoch: int,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    swa_model: AveragedModel | None,
    use_swa: bool,
    swa_start_epoch: int,
) -> None:
    path = checkpoint_pattern.format(**{'epoch': epoch, 'step': step})
    logging.info('Saving the checkpoint to {}'.format(path))
    checkpoint = build_checkpoint(
        epoch,
        step,
        model,
        optimizer,
        scaler,
        scheduler,
        swa_model,
        use_swa,
        swa_start_epoch,
    )
    torch.save(checkpoint, path)


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint.format(epoch=0, step=0)).parent
    if args.model:
        return Path(args.model.format(epoch=0, step=0)).parent
    return Path(".")


def save_hparams_yaml(args: argparse.Namespace) -> Path:
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PyYAML is required to save hparams.yaml. Please install pyyaml."
        ) from e

    run_dir = _resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "hparams.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, allow_unicode=True, sort_keys=True)
    return out_path


def _dict_to_expr(class_path: str, init_args: dict[str, Any] | None) -> str:
    if not init_args:
        return f"{class_path}()"
    args = ",".join(f"{k}={repr(v)}" for k, v in init_args.items())
    return f"{class_path}({args})"


def load_train_config(config_path: str | None) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        import yaml
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PyYAML is required for --config support. Please install pyyaml."
        ) from e

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format: {config_path}")

    defaults: Dict[str, Any] = {}

    # Flat schema: direct argparse key -> value
    known_keys = {
        "train_data", "test_data", "batchsize", "testbatchsize", "epoch", "network",
        "checkpoint", "resume", "reset_optimizer", "model", "initmodel", "log",
        "optimizer", "lr", "weight_decay", "lr_scheduler", "scheduler_step_mode",
        "reset_scheduler", "clip_grad_max_norm", "use_critic", "beta", "val_lambda",
        "val_lambda_decay_epoch", "gpu", "eval_interval", "use_swa", "swa_start_epoch",
        "swa_freq", "swa_n_avr", "use_amp", "amp_dtype", "use_average", "use_evalfix",
        "temperature", "patch", "cache",
    }
    for key in known_keys:
        if key in cfg:
            defaults[key] = cfg[key]

    # Lightning-style schema support (config.yaml compatibility)
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    trainer_cfg = cfg.get("trainer", {}) if isinstance(cfg.get("trainer", {}), dict) else {}
    opt_cfg = cfg.get("optimizer", {}) if isinstance(cfg.get("optimizer", {}), dict) else {}
    sched_cfg = cfg.get("lr_scheduler", {}) if isinstance(cfg.get("lr_scheduler", {}), dict) else {}

    if "train_data" not in defaults and "train_files" in data_cfg:
        defaults["train_data"] = data_cfg["train_files"]
    if "test_data" not in defaults and "val_files" in data_cfg and data_cfg["val_files"]:
        defaults["test_data"] = data_cfg["val_files"][0]
    if "batchsize" not in defaults and "batch_size" in data_cfg:
        defaults["batchsize"] = data_cfg["batch_size"]
    if "testbatchsize" not in defaults and "val_batch_size" in data_cfg:
        defaults["testbatchsize"] = data_cfg["val_batch_size"]
    for src, dst in [
        ("use_average", "use_average"),
        ("use_evalfix", "use_evalfix"),
        ("temperature", "temperature"),
        ("patch", "patch"),
        ("cache", "cache"),
    ]:
        if dst not in defaults and src in data_cfg:
            defaults[dst] = data_cfg[src]

    if "network" not in defaults and "network" in model_cfg:
        defaults["network"] = model_cfg["network"]
    if "val_lambda" not in defaults and "val_lambda" in model_cfg:
        defaults["val_lambda"] = model_cfg["val_lambda"]
    if "val_lambda_decay_epoch" not in defaults and "val_lambda_decay_epoch" in model_cfg:
        defaults["val_lambda_decay_epoch"] = model_cfg["val_lambda_decay_epoch"]
    if "model" not in defaults and "model_filename" in model_cfg:
        defaults["model"] = model_cfg["model_filename"]

    if "epoch" not in defaults and "max_epochs" in trainer_cfg:
        defaults["epoch"] = trainer_cfg["max_epochs"]
    if "clip_grad_max_norm" not in defaults and "gradient_clip_val" in trainer_cfg:
        defaults["clip_grad_max_norm"] = trainer_cfg["gradient_clip_val"]

    if "optimizer" not in defaults and "class_path" in opt_cfg:
        defaults["optimizer"] = _dict_to_expr(opt_cfg["class_path"], opt_cfg.get("init_args", {}))
    if "lr" not in defaults and isinstance(opt_cfg.get("init_args"), dict) and "lr" in opt_cfg["init_args"]:
        defaults["lr"] = opt_cfg["init_args"]["lr"]
    if (
        "weight_decay" not in defaults
        and isinstance(opt_cfg.get("init_args"), dict)
        and "weight_decay" in opt_cfg["init_args"]
    ):
        defaults["weight_decay"] = opt_cfg["init_args"]["weight_decay"]

    if "lr_scheduler" not in defaults and "class_path" in sched_cfg:
        defaults["lr_scheduler"] = _dict_to_expr(
            sched_cfg["class_path"], sched_cfg.get("init_args", {})
        )
    if "scheduler_step_mode" not in defaults and "lr_scheduler_interval" in model_cfg:
        defaults["scheduler_step_mode"] = model_cfg["lr_scheduler_interval"]

    return defaults


def build_parser(config_defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train policy value network')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH), help='config yaml file')
    parser.add_argument('train_data', type=str, nargs='*', help='training data file')
    parser.add_argument('test_data', type=str, nargs='?', help='test data file')
    parser.add_argument('--batchsize', '-b', type=int, default=1024, help='Number of positions in each mini-batch')
    parser.add_argument('--testbatchsize', type=int, default=1024, help='Number of positions in each test mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
    parser.add_argument('--network', default='resnet10_swish', help='network type')
    parser.add_argument('--checkpoint', default='checkpoint-{epoch:03}.pth', help='checkpoint file name')
    parser.add_argument('--resume', '-r', default='', help='Resume from snapshot')
    parser.add_argument('--reset_optimizer', action='store_true')
    parser.add_argument('--model', type=str, help='model file name')
    parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file (for compatibility)')
    parser.add_argument('--log', help='log file path')
    parser.add_argument('--optimizer', default='SGD(momentum=0.9,nesterov=True)', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--lr_scheduler', help='learning rate scheduler')
    parser.add_argument('--scheduler_step_mode', type=str, default='epoch', choices=['epoch', 'step'], help='Scheduler step mode: epoch or step')
    parser.add_argument('--reset_scheduler', action='store_true')
    parser.add_argument('--clip_grad_max_norm', type=float, default=10.0, help='max norm of the gradients')
    parser.add_argument('--use_critic', action='store_true')
    parser.add_argument('--beta', type=float, help='entropy regularization coeff')
    parser.add_argument('--val_lambda', type=float, default=0.333, help='regularization factor')
    parser.add_argument('--val_lambda_decay_epoch', type=int, help='Number of total epochs to decay val_lambda to 0')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
    parser.add_argument('--eval_interval', type=int, default=1000, help='evaluation interval')
    parser.add_argument('--use_swa', action='store_true')
    parser.add_argument('--swa_start_epoch', type=int, default=1)
    parser.add_argument('--swa_freq', type=int, default=250)
    parser.add_argument('--swa_n_avr', type=int, default=10)
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--amp_dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='Data type for automatic mixed precision')
    parser.add_argument('--use_average', action='store_true')
    parser.add_argument('--use_evalfix', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--patch', type=str, help='Overwrite with the hcpe')
    parser.add_argument('--cache', type=str, help='training data cache file')
    parser.set_defaults(**config_defaults)
    return parser


def setup_optimizer_scheduler_swa(
    args: argparse.Namespace,
    model: torch.nn.Module,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None, AveragedModel | None]:
    if args.optimizer[-1] != ')':
        args.optimizer += '()'
    optimizer = create_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)

    scheduler = None
    if args.lr_scheduler:
        scheduler = create_scheduler(args.lr_scheduler, optimizer)

    swa_model = None
    if args.use_swa:
        logging.info(
            f'use swa(swa_start_epoch={args.swa_start_epoch}, '
            f'swa_freq={args.swa_freq}, swa_n_avr={args.swa_n_avr})'
        )
        ema_a = args.swa_n_avr / (args.swa_n_avr + 1)
        ema_b = 1 / (args.swa_n_avr + 1)
        ema_avg = (
            lambda averaged_model_parameter, model_parameter, num_averaged:
            ema_a * averaged_model_parameter + ema_b * model_parameter
        )
        swa_model = AveragedModel(model, avg_fn=ema_avg)

    return optimizer, scheduler, swa_model


def setup_amp(args: argparse.Namespace) -> Tuple[torch.dtype, torch.cuda.amp.GradScaler]:
    if args.use_amp:
        logging.info(f'use amp dtype={args.amp_dtype}')
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    return amp_dtype, scaler


def compute_train_policy_loss(
    y1,
    t1,
    t2,
    value,
    use_critic: bool,
    beta: float | None,
    cross_entropy_loss_with_soft_target_fn,
) -> torch.Tensor:
    loss1 = cross_entropy_loss_with_soft_target_fn(y1, t1)
    if use_critic:
        z = t2.view(-1) - value.view(-1) + 0.5
        loss1 = (loss1 * z).mean()
    else:
        loss1 = loss1.mean()
    if beta:
        loss1 += beta * (
            F.softmax(y1, dim=1) * F.log_softmax(y1, dim=1)
        ).sum(dim=1).mean()
    return loss1


def compute_losses(
    y1,
    y2,
    t1,
    t2,
    value,
    current_val_lambda: float,
    for_train: bool,
    use_critic: bool,
    beta: float | None,
    cross_entropy_loss_with_soft_target_fn,
    cross_entropy_loss_fn,
    bce_with_logits_loss_fn,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if for_train:
        loss1 = compute_train_policy_loss(
            y1,
            t1,
            t2,
            value,
            use_critic,
            beta,
            cross_entropy_loss_with_soft_target_fn,
        )
    else:
        loss1 = cross_entropy_loss_fn(y1, t1).mean()
    loss2 = bce_with_logits_loss_fn(y2, t2)
    loss3 = bce_with_logits_loss_fn(y2, value)
    loss = loss1 + (1 - current_val_lambda) * loss2 + current_val_lambda * loss3
    return loss1, loss2, loss3, loss


def log_eval_interval_train_stats(
    current_epoch,
    current_step,
    interval_steps,
    sum_train_loss1,
    sum_train_loss2,
    sum_train_loss3,
    sum_train_loss,
    test_loss1,
    test_loss2,
    test_loss3,
    test_loss,
    test_accuracy1,
    test_accuracy2,
):
    logging.info(
        'epoch = {}, steps = {}, train loss = {:.07f}, {:.07f}, {:.07f}, {:.07f}, test loss = {:.07f}, {:.07f}, {:.07f}, {:.07f}, test accuracy = {:.07f}, {:.07f}'.format(
            current_epoch,
            current_step,
            sum_train_loss1 / interval_steps,
            sum_train_loss2 / interval_steps,
            sum_train_loss3 / interval_steps,
            sum_train_loss / interval_steps,
            test_loss1,
            test_loss2,
            test_loss3,
            test_loss,
            test_accuracy1,
            test_accuracy2,
        )
    )


def prepare_datasets_and_loaders(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, Hcpe3DataLoader, DataLoader]:
    logging.info('Reading training data')
    train_len, actual_len = Hcpe3DataLoader.load_files(
        args.train_data,
        args.use_average,
        args.use_evalfix,
        args.temperature,
        args.patch,
        args.cache,
    )
    train_data = np.arange(train_len, dtype=np.uint64)

    logging.info('Reading test data')
    test_data = np.fromfile(args.test_data, dtype=HuffmanCodedPosAndEval)

    if args.use_average:
        logging.info('train position num before preprocessing = {}'.format(actual_len))
    logging.info('train position num = {}'.format(len(train_data)))
    logging.info('test position num = {}'.format(len(test_data)))

    train_dataloader = Hcpe3DataLoader(train_data, args.batchsize, device, shuffle=True)
    test_dataloader = DataLoader(test_data, args.testbatchsize, device)
    return train_data, train_dataloader, test_dataloader


def hcpe_loader(data: np.ndarray, batchsize: int, device: torch.device):
    for x1, x2, t1, t2, value in Hcpe3DataLoader(data, batchsize, device):
        yield {'x1': x1, 'x2': x2}


def run_training_epoch(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    swa_model: AveragedModel | None,
    train_dataloader: Hcpe3DataLoader,
    test_dataloader: DataLoader,
    val_lambda: float,
    epoch: int,
    step: int,
    amp_dtype: torch.dtype,
    cross_entropy_loss_with_soft_target,
    cross_entropy_loss,
    bce_with_logits_loss,
) -> Tuple[LossAccumulator, int]:
    interval_accum = LossAccumulator()
    epoch_accum = LossAccumulator()
    eval_interval = args.eval_interval

    for x1, x2, t1, t2, value in train_dataloader:
        step += 1
        with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=amp_dtype):
            model.train()
            y1, y2 = model(x1, x2)

            model.zero_grad()
            loss1, loss2, loss3, loss = compute_losses(
                y1,
                y2,
                t1,
                t2,
                value,
                val_lambda,
                for_train=True,
                use_critic=args.use_critic,
                beta=args.beta,
                cross_entropy_loss_with_soft_target_fn=cross_entropy_loss_with_soft_target,
                cross_entropy_loss_fn=cross_entropy_loss,
                bce_with_logits_loss_fn=bce_with_logits_loss,
            )

        scaler.scale(loss).backward()
        if args.clip_grad_max_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_max_norm)
        scaler.step(optimizer)
        scaler.update()

        if args.use_swa and epoch >= args.swa_start_epoch and step % args.swa_freq == 0:
            swa_model.update_parameters(model)

        interval_accum.add(
            loss1.item(),
            loss2.item(),
            loss3.item(),
            loss.item(),
        )

        if step % eval_interval == 0:
            model.eval()

            x1, x2, t1, t2, value = test_dataloader.sample()
            with torch.no_grad():
                y1, y2 = model(x1, x2)

                loss1, loss2, loss3, loss = compute_losses(
                    y1,
                    y2,
                    t1,
                    t2,
                    value,
                    val_lambda,
                    for_train=False,
                    use_critic=args.use_critic,
                    beta=args.beta,
                    cross_entropy_loss_with_soft_target_fn=cross_entropy_loss_with_soft_target,
                    cross_entropy_loss_fn=cross_entropy_loss,
                    bce_with_logits_loss_fn=bce_with_logits_loss,
                )
                log_eval_interval_train_stats(
                    epoch,
                    step,
                    interval_accum.steps,
                    interval_accum.loss1,
                    interval_accum.loss2,
                    interval_accum.loss3,
                    interval_accum.loss,
                    loss1.item(),
                    loss2.item(),
                    loss3.item(),
                    loss.item(),
                    accuracy(y1, t1),
                    binary_accuracy(y2, t2),
                )

            epoch_accum.merge(interval_accum)
            interval_accum.clear()

        if args.lr_scheduler and args.scheduler_step_mode == 'step':
            scheduler.step()

    epoch_accum.merge(interval_accum)
    return epoch_accum, step


def make_eval_loss_fn(
    val_lambda: float,
    args: argparse.Namespace,
    cross_entropy_loss_with_soft_target,
    cross_entropy_loss,
    bce_with_logits_loss,
) -> LossFn:
    return lambda y1, y2, t1, t2, value: compute_losses(
        y1,
        y2,
        t1,
        t2,
        value,
        val_lambda,
        for_train=False,
        use_critic=args.use_critic,
        beta=args.beta,
        cross_entropy_loss_with_soft_target_fn=cross_entropy_loss_with_soft_target,
        cross_entropy_loss_fn=cross_entropy_loss,
        bce_with_logits_loss_fn=bce_with_logits_loss,
    )


def save_final_model(
    args: argparse.Namespace,
    epoch: int,
    step: int,
    model: torch.nn.Module,
    swa_model: AveragedModel | None,
    train_data: np.ndarray,
    test_dataloader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    eval_loss_fn: LossFn,
) -> None:
    if not args.model:
        return

    if args.use_swa and epoch >= args.swa_start_epoch:
        logging.info('Updating batch normalization')
        forward_ = swa_model.forward
        swa_model.forward = lambda x: forward_(**x)
        with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=amp_dtype):
            update_bn(hcpe_loader(train_data, args.batchsize, device), swa_model)
        del swa_model.forward

        # print test loss with swa model
        test_loss1, test_loss2, test_loss3, test_loss, test_accuracy1, test_accuracy2, test_entropy1, test_entropy2 = evaluate_model(
            swa_model,
            test_dataloader,
            eval_loss_fn,
        )

        logging.info(
            'epoch = {}, steps = {}, swa test loss = {:.07f}, {:.07f}, {:.07f}, {:.07f}, swa test accuracy = {:.07f}, {:.07f}, swa test entropy = {:.07f}, {:.07f}'.format(
                epoch,
                step,
                test_loss1,
                test_loss2,
                test_loss3,
                test_loss,
                test_accuracy1,
                test_accuracy2,
                test_entropy1,
                test_entropy2,
            )
        )

    model_path = args.model.format(**{'epoch': epoch, 'step': step})
    logging.info('Saving the model to {}'.format(model_path))
    serializers.save_npz(model_path, swa_model.module if args.use_swa else model)


def main(*argv):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args(argv)

    config_defaults = load_train_config(pre_args.config)
    parser = build_parser(config_defaults)
    args = parser.parse_args(argv)

    if not args.train_data:
        parser.error("train_data is required (CLI or config.yaml)")
    if not args.test_data:
        parser.error("test_data is required (CLI or config.yaml)")

    setup_logging(args.log)
    hparams_path = save_hparams_yaml(args)
    logging.info("saved hparams.yaml to {}".format(hparams_path))
    log_training_config(args)
    val_lambda = args.val_lambda

    device = resolve_device(args.gpu)

    model = policy_value_network(args.network)
    model.to(device)

    optimizer, scheduler, swa_model = setup_optimizer_scheduler_swa(args, model)
    def cross_entropy_loss_with_soft_target(pred, soft_targets):
        return torch.sum(-soft_targets * F.log_softmax(pred, dim=1), 1)
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()
    amp_dtype, scaler = setup_amp(args)

    if args.use_evalfix:
        logging.info('use evalfix')
    logging.info('temperature={}'.format(args.temperature))

    epoch, t = init_or_resume_state(
        args,
        model,
        optimizer,
        scaler,
        device,
        scheduler,
        swa_model,
    )

    logging.info('optimizer {}'.format(re.sub(' +', ' ', str(optimizer).replace('\n', ''))))

    train_data, train_dataloader, test_dataloader = prepare_datasets_and_loaders(
        args, device
    )

    # train
    for _ in range(args.epoch):
        if args.lr_scheduler:
            logging.info('lr_scheduler lr={}'.format(scheduler.get_last_lr()[0]))
        if args.val_lambda_decay_epoch:
            # update val_lambda
            val_lambda = max(
                0,
                args.val_lambda * (1 - epoch / args.val_lambda_decay_epoch)
            )
            logging.info('update val_lambda={}'.format(val_lambda))
        epoch += 1
        epoch_accum, t = run_training_epoch(
            args,
            model,
            optimizer,
            scaler,
            scheduler,
            swa_model,
            train_dataloader,
            test_dataloader,
            val_lambda,
            epoch,
            t,
            amp_dtype,
            cross_entropy_loss_with_soft_target,
            cross_entropy_loss,
            bce_with_logits_loss,
        )
        eval_loss_fn = make_eval_loss_fn(
            val_lambda,
            args,
            cross_entropy_loss_with_soft_target,
            cross_entropy_loss,
            bce_with_logits_loss,
        )

        # print train loss and test loss for each epoch
        test_loss1, test_loss2, test_loss3, test_loss, test_accuracy1, test_accuracy2, test_entropy1, test_entropy2 = evaluate_model(
            model,
            test_dataloader,
            eval_loss_fn,
        )
        train_loss1, train_loss2, train_loss3, train_loss = epoch_accum.averages()

        logging.info('epoch = {}, steps = {}, train loss avr = {:.07f}, {:.07f}, {:.07f}, {:.07f}, test loss = {:.07f}, {:.07f}, {:.07f}, {:.07f}, test accuracy = {:.07f}, {:.07f}, test entropy = {:.07f}, {:.07f}'.format(
            epoch, t,
            train_loss1, train_loss2, train_loss3, train_loss,
            test_loss1, test_loss2, test_loss3, test_loss,
            test_accuracy1, test_accuracy2,
            test_entropy1, test_entropy2))

        if args.lr_scheduler and args.scheduler_step_mode == 'epoch':
            scheduler.step()

        # save checkpoint
        if args.checkpoint:
            save_checkpoint_file(
                args.checkpoint,
                epoch,
                t,
                model,
                optimizer,
                scaler,
                scheduler,
                swa_model,
                args.use_swa,
                args.swa_start_epoch,
            )

    save_final_model(
        args,
        epoch,
        t,
        model,
        swa_model,
        train_data,
        test_dataloader,
        device,
        amp_dtype,
        eval_loss_fn,
    )

if __name__ == '__main__':
    main(*sys.argv[1:])
