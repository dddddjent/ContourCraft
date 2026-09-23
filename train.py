import os
from pathlib import Path
import random
import signal
from types import FrameType

import numpy as np
from material.utils import init_matstack
import torch

from utils.arguments import load_from_checkpoint, load_params, create_modules
from utils.writer import WandbWriter
from utils.defaults import DEFAULTS
from utils.common import restore_rng_state, save_checkpoint


def main() -> None:
    interrupted = False

    def request_stop(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        if not interrupted:
            print(f'Received signal {signum}; finishing the current sequence batch before saving.', flush=True)
        interrupted = True

    def stop_requested() -> bool:
        return interrupted

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    modules, config = load_params()
    dataloader_modules, runner_module, runner, aux_modules = create_modules(modules, config)

    if 'material_stack' in aux_modules:
        aux_modules = init_matstack(config, modules, aux_modules, dataloader_modules)
        DEFAULTS.project_name = 'gaugar_finetune'

    runner, aux_modules = load_from_checkpoint(config, runner, aux_modules)

    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        
    if config.experiment.use_writer:
        writer = WandbWriter(config)
    else:
        writer = None

    global_step = config.step_start

    torch.manual_seed(57)
    np.random.seed(57)
    random.seed(57)
    if '_resume_rng_state' in aux_modules:
        restore_rng_state(aux_modules.pop('_resume_rng_state'))
    if config.restart.resume:
        print('Resumed training state; data iterators restart with fresh sampling (no exact data-order replay).', flush=True)
    for i in range(config.experiment.n_epochs):
        if interrupted or (config.experiment.max_iter is not None and global_step >= config.experiment.max_iter):
            break
        dataloaders_dict = dict()

        for dataloader_name, dataloader in dataloader_modules.items():
            dataloaders_dict[dataloader_name] = dataloader.create_dataloader()

        if 'material_stack' in aux_modules:
            global_step = runner_module.run_epoch(runner, aux_modules, dataloaders_dict, config, writer,
                                                 global_step=global_step, stop_requested=stop_requested)
        else:
            global_step = runner_module.run_epoch(runner, aux_modules, dataloaders_dict, config, writer,
                                                 global_step=global_step)

        if config.experiment.max_iter is not None and global_step >= config.experiment.max_iter:
            break

    if 'material_stack' in aux_modules:
        config.step_start = global_step
        checkpoint_dir = Path(DEFAULTS.data_root) / config.checkpoints_dir
        save_checkpoint(runner, aux_modules, config,
                        checkpoint_dir / f'step_{global_step:010d}.pth', global_step=global_step)
        status = 'interrupted' if interrupted else 'completed'
        print(f'Training {status}: global_step={global_step}, target={config.experiment.max_iter}', flush=True)


if __name__ == '__main__':
    main()
    
