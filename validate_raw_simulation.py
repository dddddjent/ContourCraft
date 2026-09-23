"""Bounded real-registration raw-body fitting and checkpoint-rollout smoke test."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Template command (ccraft environment, from ContourCraft):
# python validate_raw_simulation.py --data ../data/GaussianGarments/ClothTransformer/sim_00000_raw --output ../data/outputs/contourcraft_raw_validation/fitting/sim_00000_raw --checkpoint ../data/ContourCraft/trained_models/contourcraft.pth --result ../data/outputs/contourcraft_raw_validation/runtime --fit-steps 1 --rollout-steps 4


def parameter_snapshot(module: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in module.named_parameters()}


def parameter_changes(module: Any, before: dict[str, Any]) -> dict[str, float | int]:
    import torch

    changed, largest = 0, 0.0
    for name, value in module.named_parameters():
        assert torch.isfinite(value).all(), name
        delta = (value.detach().cpu() - before[name]).abs().max().item()
        changed += int(delta > 0)
        largest = max(largest, delta)
        if value.grad is not None:
            assert torch.isfinite(value.grad).all(), name
    assert changed > 0, 'Optimizer did not update any parameters.'
    return {'changed_parameter_tensors': changed, 'maximum_absolute_update': largest}


def validate(data: Path, output: Path, checkpoint: Path, result: Path,
             fit_steps: int, rollout_steps: int) -> None:
    from fit_gaussian_garments import read_json

    assert 1 <= fit_steps <= 4 and 1 <= rollout_steps <= 10, 'Keep this an explicitly bounded smoke test.'
    assert checkpoint.is_file(), checkpoint
    assert not result.exists(), result
    preparation = read_json(output / 'stage4/preparation.json')
    manifest = read_json(data / 'manifest.json')
    assert preparation['body_model'] == manifest['body_model'] == 'raw'
    assert preparation['data'] == str(data)
    os.environ.update(CCRAFT_DATA_ROOT=preparation['ccraft_data'],
                      CCRAFT_PROJECT_DIR=str(Path(__file__).resolve().parent))
    sys.argv = [sys.argv[0]]

    import torch
    from torch_geometric.data import Batch
    from datasets.finetune import create
    from evaluate_gaussian_garments import build_sample, full_body_motion
    from utils.arguments import create_runner, load_params
    from utils.common import add_field_to_pyg_batch, save_checkpoint
    from utils.validation import load_runner_and_material_from_checkpoint

    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    modules, config = load_params(str(output / 'stage4/finetune'))
    dataset = create(config.dataloaders.finetune.dataset.finetune)
    batch = Batch.from_data_list([dataset[0]])
    assert batch['cloth'].lookup.shape[1] >= fit_steps
    batch['cloth'].lookup = batch['cloth'].lookup[:, :fit_steps].clone()
    batch['obstacle'].lookup = batch['obstacle'].lookup[:, :fit_steps].clone()
    batch = add_field_to_pyg_batch(batch, 'iter', [config.restart.step_start], 'cloth', reference_key=None)
    _, runner, auxiliary = create_runner(modules, config)
    runner.load_state_dict(torch.load(checkpoint, weights_only=False)['training_module'])
    runner = runner.to(config.device)
    material = auxiliary['material_stack']
    # Match material.utils.init_matstack: create Parameters from CPU samples,
    # then transfer the module, so all fitted fields stay registered.
    material.initialize([batch])
    material = material.to(config.device)
    batch = batch.to(config.device)
    auxiliary['material_stack'] = material
    auxiliary['optimizer_material'] = modules['material_stack'].create_optimizer(
        material, config.material_stack.mstack.optimizer)
    before_model = parameter_snapshot(runner.model)
    before_material = parameter_snapshot(material)
    for field in ('lame_mu_input', 'lame_lambda_input', 'bending_coeff_input', 'v_mass'):
        assert f'materials.{output.name}.{field}' in before_material, field
    print(f'Running {fit_steps} real raw-body target-fitting optimizer step(s).', flush=True)
    metrics = runner.forward_ft(batch, material, optimizer_list=[auxiliary['optimizer'],
                               auxiliary['optimizer_material']], scheduler_list=None)
    assert metrics and all(np.isfinite(value) for value in metrics.values()), metrics
    model_changes = parameter_changes(runner.model, before_model)
    material_changes = parameter_changes(material, before_material)
    result.mkdir(parents=True)
    fitted_checkpoint = result / 'smoke_checkpoint.pth'
    save_checkpoint(runner, auxiliary, config, fitted_checkpoint)
    del runner, material, auxiliary, batch, before_model, before_material
    torch.cuda.empty_cache()

    _, runner, material = load_runner_and_material_from_checkpoint(str(fitted_checkpoint), modules, config)
    runner = runner.to(config.device).eval().requires_grad_(False)
    material = material.to(config.device).eval().requires_grad_(False)
    batch = build_sample(data, output, manifest, config, preparation['gender'])
    print(f'Running {rollout_steps} prediction step(s) from the reloaded checkpoint.', flush=True)
    trajectory = runner.valid_rollout(batch, material, n_steps=rollout_steps, bare=True)
    prediction = trajectory['pred']
    assert len(prediction) == rollout_steps + 2 and np.isfinite(prediction).all()
    body, _ = full_body_motion(data, manifest)
    np.testing.assert_allclose(trajectory['obstacle'], body['verts'][:rollout_steps + 2], rtol=0, atol=1e-6)
    np.savez_compressed(result / 'predictions.npz', frame_ids=np.arange(len(prediction)),
                        vertices=prediction, faces=trajectory['cloth_faces'])
    report = dict(status='passed', data=str(data), output=str(output), checkpoint=str(checkpoint),
                  smoke_checkpoint=str(fitted_checkpoint), body_model='raw',
                  gpu=torch.cuda.get_device_name(), registered_training_frames=len(preparation['frame_ids']),
                  target_optimizer_steps=fit_steps, rollout_prediction_steps=rollout_steps,
                  rollout_frame_ids=list(range(len(prediction))),
                  fitted_model_updates=model_changes, fitted_material_updates=material_changes,
                  fitting_metrics={key: float(value) for key, value in metrics.items()},
                  heldout_cloth_used=False, synthetic_cloth_used=False,
                  scope='Native target-fitting losses/backward/optimizer and reloaded-checkpoint short rollout. '
                        'Real raw bodies and real stage2 training registrations. '
                        'No AMASS regularization, convergence claim, or held-out accuracy evaluation.')
    (result / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Passed: {result / "report.json"}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'output', 'checkpoint', 'result'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--fit-steps', type=int, default=1)
    parser.add_argument('--rollout-steps', type=int, default=4)
    args = parser.parse_args()
    validate(args.data.resolve(), args.output.resolve(), args.checkpoint.resolve(),
             args.result.resolve(), args.fit_steps, args.rollout_steps)


if __name__ == '__main__':
    main()
