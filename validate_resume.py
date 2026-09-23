"""Compare a bounded raw-body target update before and after checkpoint resume."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

# Template command (ccraft environment, from ContourCraft):
# python validate_resume.py --output ../data/outputs/contourcraft_raw_validation/fitting/sim_00000_raw --checkpoint ../data/outputs/contourcraft_raw_validation/runtime/smoke_checkpoint.pth --result ../data/outputs/contourcraft_resume_validation --rtol 0.002 --atol 0.00000001 --parameter-rtol 0.000001 --parameter-atol 0.00000001


def create_state(modules: dict[str, Any], config: Any, sample: Any) -> tuple[Any, dict[str, Any]]:
    from utils.arguments import create_runner

    _, runner, auxiliary = create_runner(modules, config)
    runner = runner.to(config.device)
    material = auxiliary['material_stack']
    # Match native init_matstack: material parameters are first created from a
    # CPU dataloader batch and then moved to the configured device together.
    material.initialize([sample.clone().cpu()])
    auxiliary['material_stack'] = material.to(config.device)
    auxiliary['optimizer_material'] = modules['material_stack'].create_optimizer(
        material, config.material_stack.mstack.optimizer)
    return runner, auxiliary


def verify_parameter_references(auxiliary: dict[str, Any]) -> None:
    material_ids = {id(parameter) for parameter in auxiliary['material_stack'].parameters()}
    optimizer_ids = {id(parameter) for group in auxiliary['optimizer_material'].param_groups
                     for parameter in group['params']}
    assert material_ids == optimizer_ids, 'Material optimizer references obsolete Parameter objects.'


def random_probe(device: str) -> dict[str, float]:
    import torch

    return dict(python=random.random(), numpy=float(np.random.random()),
                torch=float(torch.rand(())), cuda=float(torch.rand((), device=device)))


def target_step(runner: Any, auxiliary: dict[str, Any], sample: Any, step: int) -> dict[str, float]:
    from utils.common import add_field_to_pyg_batch, copy_pyg_batch
    from validate_raw_simulation import parameter_changes, parameter_snapshot

    sample = copy_pyg_batch(sample)
    sample = add_field_to_pyg_batch(sample, 'iter', [step], 'cloth', reference_key=None)
    verify_parameter_references(auxiliary)
    before = parameter_snapshot(auxiliary['material_stack'])
    metrics = runner.forward_ft(sample, auxiliary['material_stack'],
                                optimizer_list=[auxiliary['optimizer'], auxiliary['optimizer_material']],
                                scheduler_list=[auxiliary['scheduler'], None])
    assert metrics and all(np.isfinite(value) for value in metrics.values()), metrics
    parameter_changes(auxiliary['material_stack'], before)
    return {name: float(value) for name, value in metrics.items()}


def compare_state(expected: Any, actual: Any, rtol: float, atol: float,
                  path: str, statistics: dict[str, float | int]) -> None:
    """Compare all nested optimizer/module values with a per-tensor scale tolerance."""
    import torch

    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and expected.shape == actual.shape, path
        left, right = expected.detach().cpu(), actual.detach().cpu()
        assert torch.isfinite(left).all() and torch.isfinite(right).all(), path
        if left.is_floating_point():
            error = float((left - right).abs().max())
            scale = float(left.abs().max())
            assert error <= atol + rtol * scale, (path, error, scale)
            statistics['maximum_absolute_difference'] = max(statistics['maximum_absolute_difference'], error)
            statistics['maximum_scaled_difference'] = max(statistics['maximum_scaled_difference'],
                                                           error / (atol + rtol * scale) if error else 0.0)
        else:
            assert torch.equal(left, right), path
        statistics['tensor_count'] += 1
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        for name in expected:
            compare_state(expected[name], actual[name], rtol, atol, f'{path}.{name}', statistics)
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual), path
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare_state(left, right, rtol, atol, f'{path}[{index}]', statistics)
    else:
        assert expected == actual, (path, expected, actual)


def validate(output: Path, checkpoint: Path, result: Path, rtol: float, atol: float,
             parameter_rtol: float, parameter_atol: float) -> None:
    from fit_gaussian_garments import read_json

    assert checkpoint.is_file(), checkpoint
    assert not result.exists(), result
    assert rtol > 0 and atol > 0 and parameter_rtol > 0 and parameter_atol > 0
    preparation = read_json(output / 'stage4/preparation.json')
    assert preparation['body_model'] == 'raw'
    os.environ.update(CCRAFT_DATA_ROOT=preparation['ccraft_data'],
                      CCRAFT_PROJECT_DIR=str(Path(__file__).resolve().parent))
    sys.argv = [sys.argv[0]]

    import torch
    from datasets.finetune import create
    from torch_geometric.data import Batch
    from utils.arguments import load_from_checkpoint, load_params
    from utils.common import restore_rng_state, save_checkpoint

    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    modules, config = load_params(str(output / 'stage4/finetune'))
    dataset = create(config.dataloaders.finetune.dataset.finetune)
    sample = Batch.from_data_list([dataset[0]]).to(config.device)
    sample['cloth'].lookup = sample['cloth'].lookup[:, :1].clone()
    sample['obstacle'].lookup = sample['obstacle'].lookup[:, :1].clone()
    runner, auxiliary = create_state(modules, config, sample)
    # Explicitly bootstrap the older smoke fixture; production resume uses only
    # the current checkpoint format written below.
    smoke = torch.load(checkpoint, weights_only=False)
    runner.load_state_dict(smoke['training_module'])
    for name in ('material_stack', 'optimizer', 'scheduler', 'optimizer_material'):
        auxiliary[name].load_state_dict(smoke[name])
    del smoke
    verify_parameter_references(auxiliary)
    assert auxiliary['optimizer_material'].state, 'Expected an already trained smoke checkpoint.'
    config.step_start = int(config.restart.step_start)
    config.restart.training_origin = config.step_start
    print('Running a target step from the existing trained smoke checkpoint.', flush=True)
    target_step(runner, auxiliary, sample, config.step_start + 1)
    config.step_start += 1
    result.mkdir(parents=True)
    resume_checkpoint = result / 'resume_checkpoint.pth'
    save_checkpoint(runner, auxiliary, config, resume_checkpoint, global_step=config.step_start)
    saved = torch.load(resume_checkpoint, weights_only=False, map_location='cpu')
    expected_rng = random_probe(config.device)

    print('Running uninterrupted next target step.', flush=True)
    uninterrupted_metrics = target_step(runner, auxiliary, sample, config.step_start + 1)
    config.step_start += 1
    expected_checkpoint = result / 'uninterrupted.pth'
    save_checkpoint(runner, auxiliary, config, expected_checkpoint, global_step=config.step_start)
    expected = torch.load(expected_checkpoint, weights_only=False, map_location='cpu')
    del runner, auxiliary
    torch.cuda.empty_cache()

    runner, auxiliary = create_state(modules, config, sample)
    original_material_ids = {id(parameter) for parameter in auxiliary['material_stack'].parameters()}
    config.restart.checkpoint_path = str(resume_checkpoint)
    config.restart.resume = True
    config.restart.step_start = saved['global_step']
    runner, auxiliary = load_from_checkpoint(config, runner, auxiliary)
    assert config.step_start == saved['global_step']
    restore_rng_state(auxiliary.pop('_resume_rng_state'))
    assert random_probe(config.device) == expected_rng, 'Checkpoint RNG state did not restore exactly.'
    assert {id(parameter) for parameter in auxiliary['material_stack'].parameters()} == original_material_ids
    verify_parameter_references(auxiliary)
    exact = dict(tensor_count=0, maximum_absolute_difference=0.0, maximum_scaled_difference=0.0)
    compare_state(saved['training_module'], runner.state_dict(), 0.0, 0.0, 'loaded_model', exact)
    for name in ('material_stack', 'optimizer', 'scheduler', 'optimizer_material'):
        compare_state(saved[name], auxiliary[name].state_dict(), 0.0, 0.0, f'loaded_{name}', exact)
    print('Running the same next target step after production checkpoint resume.', flush=True)
    resumed_metrics = target_step(runner, auxiliary, sample, config.step_start + 1)
    config.step_start += 1
    resumed_checkpoint = result / 'resumed.pth'
    save_checkpoint(runner, auxiliary, config, resumed_checkpoint, global_step=config.step_start)
    actual = torch.load(resumed_checkpoint, weights_only=False, map_location='cpu')
    assert actual['global_step'] == expected['global_step']
    comparisons = {}
    for name in ('training_module', 'material_stack', 'optimizer', 'scheduler', 'optimizer_material'):
        stats = dict(tensor_count=0, maximum_absolute_difference=0.0, maximum_scaled_difference=0.0)
        is_parameter = name in ('training_module', 'material_stack')
        compare_state(expected[name], actual[name], parameter_rtol if is_parameter else rtol,
                      parameter_atol if is_parameter else atol, name, stats)
        comparisons[name] = stats
    report = dict(status='passed', source_checkpoint=str(checkpoint), output=str(output),
                  gpu=torch.cuda.get_device_name(), target_optimizer_steps_executed=3,
                  material_parameter_identity_preserved=True, optimizer_references_current_material=True,
                  random_state_restored=True, resumed_global_step=actual['global_step'],
                  checkpoint_restore_exact=exact, continuation_comparisons=comparisons,
                  uninterrupted_metrics=uninterrupted_metrics, resumed_metrics=resumed_metrics,
                  optimizer_rtol=rtol, optimizer_atol=atol,
                  parameter_rtol=parameter_rtol, parameter_atol=parameter_atol,
                  scope='Same real raw-body target sample update before and after production checkpoint resume; '
                        'checks network/material/optimizer/scheduler state and live material updates. '
                        'Continuation tolerances allow measured numerical differences; their cause is not established. '
                        'No AMASS alternating-training, dataloader-position, or convergence validation.')
    (result / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Passed: {result / "report.json"}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'checkpoint', 'result'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--rtol', type=float, default=2e-3, help='Optimizer continuation relative tolerance.')
    parser.add_argument('--atol', type=float, default=1e-8, help='Optimizer continuation absolute tolerance.')
    parser.add_argument('--parameter-rtol', type=float, default=1e-6)
    parser.add_argument('--parameter-atol', type=float, default=1e-8)
    args = parser.parse_args()
    validate(args.output.resolve(), args.checkpoint.resolve(), args.result.resolve(), args.rtol, args.atol,
             args.parameter_rtol, args.parameter_atol)


if __name__ == '__main__':
    main()
