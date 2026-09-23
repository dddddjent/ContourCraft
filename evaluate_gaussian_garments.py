"""Roll out fitted Gaussian-Garments behavior with known future body motion."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from fit_gaussian_garments import convert_body, convert_raw_body, read_json

# Template command (ccraft environment, from ContourCraft):
# python evaluate_gaussian_garments.py --data ../data/GaussianGarments/ClothTransformer/sim_00000 --output ../Gaussian-Garments/output/ClothTransformer/sim_00000 --evaluation ../Gaussian-Garments/output/ClothTransformer/sim_00000/evaluation/full_sequence --checkpoint ../Gaussian-Garments/output/ClothTransformer/sim_00000/stage4/checkpoints/step_0000046000.pth --evaluate-from-start


def full_body_motion(data: Path, manifest: dict[str, Any]) -> tuple[dict[str, np.ndarray], str]:
    """Join only body motion, checking topology and physical time across the split."""
    body_model = manifest.get('body_model', 'smplx')
    assert body_model in ('smplx', 'raw'), body_model
    bodies, genders, frame_ids, timestamps = [], [], [], []
    for name in ('train', 'heldout'):
        split = manifest['splits'][name]
        ids = split['local_frame_ids']
        assert ids == list(range(len(ids))) and ids, f'Invalid {name} local frame order.'
        assert len(ids) == len(split['global_frame_ids']) == len(split['global_timestamps'])
        if body_model == 'raw':
            body = convert_raw_body(data / split['directory'] / 'body_mesh', ids, manifest['fps'])
            gender = 'none'
        else:
            body, gender = convert_body(data / split['directory'] / 'smplx', ids, manifest['fps'])
        bodies.append(body)
        genders.append(gender)
        frame_ids.extend(split['global_frame_ids'])
        timestamps.extend(split['global_timestamps'])
    assert frame_ids == manifest['frame_ids'] == list(range(len(frame_ids)))
    assert manifest['evaluation_frame_ids'] == manifest['splits']['heldout']['global_frame_ids']
    assert float(manifest['fps']) > 0
    np.testing.assert_allclose(np.diff(timestamps), 1 / manifest['fps'], rtol=0, atol=1e-7,
                               err_msg='Body motion must remain continuous across the split.')
    np.testing.assert_allclose(timestamps, manifest['timestamps'], rtol=0, atol=1e-7)
    assert genders[0] == genders[1]
    if body_model == 'raw':
        np.testing.assert_array_equal(bodies[0]['faces'], bodies[1]['faces'],
                                      err_msg='Raw body topology changes at the split boundary.')
        assert bodies[0]['verts'].shape[1:] == bodies[1]['verts'].shape[1:], (
            'Raw body vertex count changes at the split boundary.')
        joined = {'verts': np.concatenate([body['verts'] for body in bodies]),
                  'faces': bodies[0]['faces']}
    else:
        joined = {key: np.concatenate([body[key] for body in bodies])
                  for key in bodies[0] if key != 'mocap_frame_rate'}
    joined['mocap_frame_rate'] = np.asarray(manifest['fps'])
    return joined, genders[0]


def build_sample(data: Path, output: Path, manifest: dict[str, Any],
                 config: Any, gender: str) -> Any:
    """Construct native simulator input without loading future garment observations."""
    from torch_geometric.data import Batch, HeteroData
    from datasets.finetune import create_loader
    from utils.common import NodeType

    dataset_config = config.dataloaders.finetune.dataset.finetune
    body_model = manifest.get('body_model', 'smplx')
    assert dataset_config.body_model == body_model, 'Fitting and evaluation body models differ.'
    assert dataset_config.chronological_registration and dataset_config.fps == manifest['fps']
    assert np.isclose(config.runner.finetune.finetune_ts, 1 / manifest['fps'])
    loader = create_loader(dataset_config)
    body, actual_gender = full_body_motion(data, manifest)
    assert actual_gender == gender
    if body_model == 'raw':
        sequence = body
    else:
        sequence = loader.sequence_loader.convert_seq_to_hood_format(body)
        sequence = loader.sequence_loader.process_sequence(sequence)
    sample = loader.build_body_sample(HeteroData(), sequence, gender)
    sample = loader.garment_builder.build(sample, output / 'stage4/registrations/train.pkl',
                                          output.name, sequence)
    assert not (sample['cloth'].vertex_type == NodeType.HANDLE).any(), (
        'This evaluation requires unpinned garments; future attachment targets are unavailable.')
    # Native inference requires target tensors even for unpinned vertices. They
    # are masked out of integration and bare=True disables target-based losses.
    # No future garment trajectory is loaded or passed to the simulator.
    sample['cloth'].target_pos = sample['cloth'].pos.clone()
    sample['cloth'].lookup = sample['cloth'].pos[:, None].expand(-1, len(manifest['frame_ids']) - 2, -1).clone()
    sample['sequence_name'], sample['garment_name'] = 'train', output.name
    return Batch.from_data_list([sample]).to(config.device)


def evaluate(data: Path, output: Path, evaluation: Path, checkpoint: Path,
             from_start: bool) -> None:
    manifest = read_json(data / 'manifest.json')
    preparation = read_json(output / 'stage4/preparation.json')
    assert preparation['data'] == str(data)
    body_model = manifest.get('body_model', 'smplx')
    assert body_model in ('smplx', 'raw'), body_model
    assert preparation.get('body_model', 'smplx') == body_model
    assert checkpoint.is_file(), checkpoint
    assert not evaluation.exists(), evaluation
    os.environ.update(CCRAFT_DATA_ROOT=preparation['ccraft_data'],
                      CCRAFT_PROJECT_DIR=str(Path(__file__).resolve().parent))

    import torch
    from utils.arguments import load_params
    from utils.validation import load_runner_and_material_from_checkpoint

    # OmegaConf must not parse this entry point's argparse flags.
    sys.argv = [sys.argv[0]]
    torch.manual_seed(0)
    np.random.seed(0)
    modules, config = load_params(str(output / 'stage4/finetune'))
    _, runner, material = load_runner_and_material_from_checkpoint(str(checkpoint), modules, config)
    assert output.name in material.materials, 'Checkpoint does not contain this fitted garment.'
    runner = runner.to(config.device).eval()
    material = material.to(config.device).eval()
    runner.requires_grad_(False)
    material.requires_grad_(False)
    batch = build_sample(data, output, manifest, config, preparation['gender'])
    # ContourCraft differentiates collision contours with respect to geometry
    # during inference. Freeze parameters above, but retain geometry autograd.
    trajectory = runner.valid_rollout(batch, material, bare=True)
    prediction = trajectory['pred']
    assert len(prediction) == len(manifest['frame_ids']) and np.isfinite(prediction).all()
    frames = manifest['frame_ids'] if from_start else manifest['evaluation_frame_ids']
    evaluation.mkdir(parents=True)
    np.savez_compressed(evaluation / 'predictions.npz', frame_ids=frames,
                        vertices=prediction[frames], faces=trajectory['cloth_faces'])
    (evaluation / 'rollout.json').write_text(json.dumps({
        'status': 'complete', 'data': str(data), 'checkpoint': str(checkpoint),
        'body_model': body_model,
        'frame_ids': frames, 'rollout_frame_ids': manifest['frame_ids'], 'fps': manifest['fps'],
        'evaluation_scope': 'full_sequence' if from_start else 'held_out',
        'initialization': 'First two stage2 training registrations; native ContourCraft time integration',
        'heldout_cloth_used_by_simulator': False,
        'parameters': 'Frozen fitted network, material parameters and rest-edge multipliers',
    }, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'output', 'evaluation', 'checkpoint'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--evaluate-from-start', action='store_true')
    args = parser.parse_args()
    evaluate(args.data.resolve(), args.output.resolve(), args.evaluation.resolve(),
             args.checkpoint.resolve(), args.evaluate_from_start)


if __name__ == '__main__':
    main()
