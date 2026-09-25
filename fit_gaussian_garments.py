"""Prepare and fit Gaussian Garments exports with the authors' ContourCraft trainer."""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
from types import FrameType
from typing import Any

import numpy as np
import yaml

# Template command (ccraft environment, from ContourCraft):
# python fit_gaussian_garments.py --data ../data/GaussianGarments/ClothTransformer/sim_00000 --output ../Gaussian-Garments/output/ClothTransformer/sim_00000 --template-frame 0 --ccraft-data ../data/ContourCraft --body-model-root ../data/ContourCraft/aux_data/body_models --checkpoint ../data/ContourCraft/trained_models/contourcraft.pth --cmu-root /path/to/AMASS/smpl/CMU --steps 1000
# Add --prepare-only to export arrays/config without GPU import or training.
# Raw exports use stage1/template_uv.obj as rest_pos, following the author importer.
# Resume: python fit_gaussian_garments.py --data ../data/GaussianGarments/ClothTransformer/sim_00000_raw --output ../Gaussian-Garments/output/ClothTransformer/sim_00000_raw_gt_init --ccraft-data ../data/ContourCraft --cmu-root ../../datasets/AMASS/CMU --resume --save-every 10


def read_json(path: Path) -> dict[str, Any]:
    assert path.is_file(), path
    return json.loads(path.read_text())


def saved_checkpoints(output: Path) -> list[Path]:
    """List completed numeric checkpoints, excluding atomic-write temporary files."""
    return sorted((path for path in (output / 'stage4/checkpoints').glob('step_*.pth')
                   if path.stem.removeprefix('step_').isdigit()),
                  key=lambda path: int(path.stem.removeprefix('step_')))


def select_resume_checkpoint(output: Path, choice: str) -> Path:
    checkpoints = saved_checkpoints(output)
    if choice == 'latest':
        assert checkpoints, f'No saved fitting checkpoint to resume in {output / "stage4/checkpoints"}'
        return checkpoints[-1]
    checkpoint = Path(choice).expanduser().resolve()
    assert checkpoint.is_file(), checkpoint
    assert checkpoint.stem.startswith('step_') and checkpoint.stem[5:].isdigit(), (
        'Resume requires a numbered fitting checkpoint: step_XXXXXXXXXX.pth')
    later = [path for path in checkpoints if int(path.stem[5:]) > int(checkpoint.stem[5:])]
    assert not later, f'Newer checkpoints already exist; use --resume to continue from {later[-1]}'
    return checkpoint


def resume_training_command(output: Path, choice: str, steps: int | None,
                            save_every: int) -> tuple[list[str], int, Path]:
    """Restore fitted state while retaining the prepared input/configuration files."""
    import torch

    stage4 = output / 'stage4'
    preparation = read_json(stage4 / 'preparation.json')
    assert read_json(stage4 / 'garment_import.json') == preparation, 'Garment import is incomplete or differs.'
    config = yaml.safe_load((stage4 / 'finetune.yaml').read_text())
    checkpoint = select_resume_checkpoint(output, choice)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    required = {'training_module', 'material_stack', 'optimizer', 'optimizer_material', 'scheduler', 'config'}
    assert required.issubset(state), f'Checkpoint lacks resume state: {sorted(required - state.keys())}'
    assert any(key.startswith(f'materials.{output.name}.') for key in state['material_stack']), (
        'Checkpoint does not contain this fitted garment.')
    saved_config = state['config']
    assert Path(saved_config['checkpoints_dir']).resolve() == (stage4 / 'checkpoints').resolve(), (
        'Resume checkpoint belongs to a different experiment.')
    step = int(checkpoint.stem[5:])
    assert state.get('global_step', step) == step, 'Checkpoint step differs from its filename.'
    origin = config['restart']['step_start']
    target = saved_config['experiment']['max_iter'] if steps is None else origin + steps
    assert target is not None and target > step >= origin, (
        f'Checkpoint is at {step}; requested target is {target}. Use --steps for a larger TOTAL budget if needed.')
    command = [sys.executable, 'train.py', f'config={stage4 / "finetune"}',
               'restart.resume=True', f'restart.checkpoint_path={checkpoint}',
               f'restart.step_start={step}', f'restart.training_origin={origin}',
               f'experiment.max_iter={target}', f'experiment.save_checkpoint_every={save_every}',
               f'experiment.save_checkpoint_every_wlong={save_every}']
    return command, target, checkpoint


def run_training(command: list[str], env: dict[str, str]) -> None:
    """Forward stop requests to the trainer and wait for its completed-batch save."""
    process = subprocess.Popen(command, cwd=Path(__file__).resolve().parent, env=env,
                               start_new_session=True)

    def request_stop(signum: int, frame: FrameType | None) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    previous = {signum: signal.signal(signum, request_stop) for signum in (signal.SIGINT, signal.SIGTERM)}
    # Restore caller signal handling even if launching/waiting raises an error.
    try:
        returncode = process.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def inspect_inputs(data: Path, output: Path, template_frame: int) -> dict[str, Any]:
    """Require the complete training prefix, without reading held-out observations."""
    manifest = read_json(data / 'manifest.json')
    assert manifest['status'] == 'complete'
    assert output.name and '.' not in output.name, 'Experiment name must be a valid material ModuleDict key (no dots).'
    assert manifest['component'] in ('clothtransformer_gaussian_garments_export',
                                     'dgarments_gaussian_garments_export')
    body_model = manifest.get('body_model', 'smplx')
    assert body_model in ('smplx', 'raw'), body_model
    train = manifest['splits']['train']
    ids = train['local_frame_ids']
    assert ids == list(range(len(ids))) and len(ids) >= 3, 'Need at least three contiguous training frames.'
    assert 0 <= template_frame < len(ids)
    fps = float(manifest['fps'])
    assert fps > 0 and fps.is_integer()
    np.testing.assert_allclose(np.diff(train['global_timestamps']), 1 / fps, rtol=0, atol=1e-7)
    assert train['global_frame_ids'] == list(range(len(ids))), 'Expected chronological training prefix.'
    heldout = manifest['splits']['heldout']['global_frame_ids']
    assert set(train['global_frame_ids']).isdisjoint(heldout)
    source = data / train['directory']
    assert source == data / 'input' / manifest['subject'] / 'train'
    experiment = read_json(output / 'experiment.json')
    assert experiment['data'] == str(data) and experiment['subject'] == manifest['subject']
    assert experiment['template_frame'] == template_frame, 'Use the reconstruction template frame.'
    for path in (output / 'stage1/template_uv.obj', output / 'stage2/Template/template.obj'):
        assert path.is_file(), path
    meshes = output / 'stage2/train/meshes'
    expected = [meshes / f'frame_{frame:05d}.obj' for frame in ids]
    missing = [path.name for path in expected if not path.is_file()]
    assert not missing, f'Incomplete training registration: {len(missing)} missing meshes; first: {missing[:3]}'
    assert sorted(meshes.glob('*.obj')) == expected, 'Registration frames differ from the training split.'
    body_directory, extension = ('body_mesh', 'ply') if body_model == 'raw' else ('smplx', 'pkl')
    for frame in ids:
        path = source / body_directory / f'{frame:05d}.{extension}'
        assert path.is_file(), path
    return manifest


def convert_body(source: Path, ids: list[int], fps: float) -> tuple[dict[str, np.ndarray], str]:
    """Author CMU SMPL-X schema, preserving exported full axis-angle hands."""
    fields = {'betas': 10, 'expression': 10, 'global_orient': 3, 'body_pose': 63,
              'jaw_pose': 3, 'leye_pose': 3, 'reye_pose': 3, 'left_hand_pose': 45,
              'right_hand_pose': 45, 'transl': 3}
    records = []
    for frame in ids:
        with (source / f'{frame:05d}.pkl').open('rb') as stream:
            record = pickle.load(stream)
        assert record['use_pca'] is False and record['flat_hand_mean'] is True
        assert record['gender'] in ('female', 'male', 'neutral')
        for name, size in fields.items():
            value = np.asarray(record[name])
            assert value.shape == (size,) and np.isfinite(value).all(), (frame, name, value.shape)
        records.append(record)
    gender = records[0]['gender']
    assert all(record['gender'] == gender for record in records)
    values = {key: np.stack([record[key] for record in records]).astype(np.float32) for key in fields}
    result = {key: values[key] for key in ('betas', 'expression')}
    result.update(trans=values['transl'], root_orient=values['global_orient'],
                  pose_body=values['body_pose'], pose_jaw=values['jaw_pose'],
                  pose_hand=np.concatenate([values['left_hand_pose'], values['right_hand_pose']], axis=-1),
                  pose_eye=np.concatenate([values['leye_pose'], values['reye_pose']], axis=-1),
                  mocap_frame_rate=np.asarray(fps))
    return result, gender


def convert_raw_body(source: Path, ids: list[int], fps: float) -> dict[str, np.ndarray]:
    """Preserve raw PLY vertex correspondence, faces and capture coordinates."""
    import trimesh

    assert ids and fps > 0
    vertices = []
    reference_faces = None
    for index, frame in enumerate(ids):
        path = source / f'{frame:05d}.ply'
        assert path.is_file(), path
        mesh = trimesh.load_mesh(path, process=False)
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        assert verts.ndim == 2 and verts.shape[1] == 3 and np.isfinite(verts).all(), path
        assert faces.ndim == 2 and faces.shape[1] == 3 and faces.size > 0, path
        assert faces.min() >= 0 and faces.max() < len(verts), path
        if index == 0:
            reference_faces = faces
        if index:
            assert verts.shape == vertices[0].shape, f'Body vertex count changed: {path}'
        np.testing.assert_array_equal(faces, reference_faces, err_msg=f'Body topology changed: {path}')
        vertices.append(verts)
    return dict(verts=np.stack(vertices), faces=reference_faces, mocap_frame_rate=np.asarray(fps))


def convert_registration(output: Path, ids: list[int]) -> dict[str, np.ndarray]:
    """Use the author's OBJ reader and sequence schema, checking shared topology."""
    from utils.io import load_obj

    template_vertices, template_faces, _, _ = load_obj(output / 'stage1/template_uv.obj', tex_coords=True)
    vertices, uv_coords = [], []
    reference_uv_faces = None
    for frame in ids:
        verts, faces, uvs, uv_faces = load_obj(output / f'stage2/train/meshes/frame_{frame:05d}.obj', tex_coords=True)
        assert verts.shape == template_vertices.shape and np.isfinite(verts).all()
        np.testing.assert_array_equal(faces, template_faces)
        assert uvs.ndim == 2 and uvs.shape[1] == 2 and np.isfinite(uvs).all()
        assert uv_faces.shape == faces.shape and uv_faces.min() >= 0 and uv_faces.max() < len(uvs)
        if frame == 0:
            reference_uv_faces = uv_faces
        np.testing.assert_array_equal(uv_faces, reference_uv_faces)
        vertices.append(verts)
        uv_coords.append(uvs)
    trajectory = np.stack(vertices)
    return dict(vertices=trajectory, faces=template_faces, uv_coords=np.stack(uv_coords),
                uv_faces=reference_uv_faces, pred=trajectory, cloth_faces=template_faces)


def make_config(stage4: Path, ccraft_data: Path, body_models: Path,
                checkpoint: Path, fps: float, steps: int, body_model: str = 'smplx') -> dict[str, Any]:
    """Change only paths and target export conventions in the released base config."""
    config = yaml.safe_load((Path(__file__).parent / 'configs/finetune/base.yaml').read_text())
    target = config['dataloaders']['finetune']['dataset']['finetune']
    target.update(train_split_path=str(stage4 / 'train.csv'), valid_split_path=str(stage4 / 'valid.csv'),
                  registration_root=str(stage4 / 'registrations'), body_sequence_root=str(stage4 / 'smplx'),
                  body_model_root=str(body_models), garment_dicts_dir=str(stage4 / 'garment_dicts'),
                  fps=int(fps), flat_hand_mean=True, preserve_wrist_pose=True, chronological_registration=True)
    if body_model == 'raw':
        target.update(body_model='raw', body_sequence_root=str(stage4 / 'body_mesh'), sequence_loader='mesh',
                      pinned_verts=False, omit_hands=False, use_betas_for_restpos=False,
                      obstacle_dict_file=None)
    config['dataloaders']['long']['dataset']['ccraft']['body_model_root'] = str(body_models)
    config['runner']['finetune']['finetune_ts'] = 1 / fps
    config['restart']['checkpoint_path'] = str(checkpoint)
    config['experiment']['max_iter'] = config['restart']['step_start'] + steps
    config['checkpoints_dir'] = str(stage4 / 'checkpoints')
    # Keep metrics enabled, but never use the upstream author's W&B account.
    return config


def prepare(data: Path, output: Path, manifest: dict[str, Any], template_frame: int,
            ccraft_data: Path, body_models: Path, checkpoint: Path, steps: int) -> dict[str, Any]:
    stage4 = output / 'stage4'
    train = manifest['splits']['train']
    ids = train['local_frame_ids']
    body_model = manifest.get('body_model', 'smplx')
    body_directory = 'body_mesh' if body_model == 'raw' else 'smplx'
    if body_model == 'raw':
        body = convert_raw_body(data / train['directory'] / body_directory, ids, manifest['fps'])
        gender = 'none'
    else:
        body, gender = convert_body(data / train['directory'] / body_directory, ids, manifest['fps'])
    config = make_config(stage4, ccraft_data, body_models, checkpoint, manifest['fps'], steps, body_model)
    record = dict(status='prepared', data=str(data), output=str(output), template_frame=template_frame,
                  subject=output.name, gender=gender, fps=manifest['fps'], frame_ids=ids,
                  global_frame_ids=train['global_frame_ids'], ccraft_data=str(ccraft_data),
                  body_model_root=str(body_models), checkpoint=str(checkpoint), steps=steps,
                  supervision='stage2 training registrations only; held-out observations excluded',
                  validation='empty: the exported held-out suffix is reserved for subsequent evaluation')
    if body_model == 'raw':
        record['body_model'] = 'raw'
        record['rest_mesh'] = str(output / 'stage1/template_uv.obj')
        record['rest_shape'] = 'Original author convention: posed stage1/template_uv.obj vertices stored as rest_pos'
    if stage4.exists():
        assert read_json(stage4 / 'preparation.json') == record, 'Existing preparation differs; use its original arguments.'
        assert yaml.safe_load((stage4 / 'finetune.yaml').read_text()) == config
        for path in (f'{body_directory}/train.npz', 'registrations/train.pkl', 'train.csv', 'valid.csv'):
            assert (stage4 / path).is_file(), stage4 / path
        return record
    registration = convert_registration(output, ids)
    (stage4 / body_directory).mkdir(parents=True)
    (stage4 / 'registrations').mkdir()
    (stage4 / 'garment_dicts').mkdir()
    np.savez_compressed(stage4 / body_directory / 'train.npz', **body)
    with (stage4 / 'registrations/train.pkl').open('wb') as stream:
        pickle.dump(registration, stream, protocol=pickle.HIGHEST_PROTOCOL)
    for name in ('train', 'valid'):
        with (stage4 / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('id', 'length', 'garment', 'gender'))
            if name == 'train':
                writer.writerow(('train', len(ids), output.name, gender))
    (stage4 / 'finetune.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (stage4 / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def check_training_assets(ccraft_data: Path, body_models: Path, checkpoint: Path,
                          cmu_root: Path, gender: str, body_model: str = 'smplx') -> None:
    """Fail before garment import if the author's alternating training data is absent."""
    assert checkpoint.is_file(), f'Missing author checkpoint: {checkpoint}'
    if body_model == 'smplx':
        assert (body_models / 'smplx' / f'SMPLX_{gender.upper()}.npz').is_file(), body_models
    split = ccraft_data / 'aux_data/datasplits/train_ccraft.csv'
    assert split.is_file(), f'Missing author auxiliary data: {split}'
    assert (ccraft_data / 'aux_data/smpl_aux.pkl').is_file()
    with split.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows, 'The original ContourCraft training split is empty.'
    for row in rows:
        sequence = cmu_root / row['id']
        if sequence.suffix != '.npz':
            sequence = Path(str(sequence) + '.npz')
        assert sequence.is_file(), f'Missing AMASS CMU regularization sequence: {sequence}'
        for garment_name in row['garment'].split(','):
            garment = ccraft_data / 'aux_data/garment_dicts/smpl' / garment_name.strip()
            if garment.suffix != '.pkl':
                garment = Path(str(garment) + '.pkl')
            assert garment.is_file(), garment
        assert (body_models / 'smpl' / f"SMPL_{row['gender'].upper()}.pkl").is_file(), body_models


def import_garment(output: Path, record: dict[str, Any], body_models: Path, checkpoint: Path) -> None:
    """Build raw rest-shape graphs, or run the author's parametric garment import."""
    from utils.mesh_creation import GarmentCreator

    stage4 = output / 'stage4'
    ready = stage4 / 'garment_import.json'
    garment = stage4 / 'garment_dicts' / f'{output.name}.pkl'
    if ready.exists():
        assert read_json(ready) == record and garment.is_file()
        return
    assert not garment.exists(), f'Incomplete garment import; inspect and remove {garment} before retrying.'
    if record.get('body_model', 'smplx') == 'raw':
        template = output / 'stage1/template_uv.obj'
        assert record['rest_mesh'] == str(template) and template.is_file()
        creator = GarmentCreator(stage4 / 'garment_dicts', None, None, None,
                                 collect_lbs=False, coarse=True, approximate_center=True,
                                 verbose=True, swap_axes=False)
        garment_dict = creator.make_garment_dict(template)
        with garment.open('wb') as stream:
            pickle.dump(garment_dict, stream, protocol=pickle.HIGHEST_PROTOCOL)
        ready.write_text(json.dumps(record, indent=2) + '\n')
        return

    import smplx

    creator = GarmentCreator(stage4 / 'garment_dicts', body_models, 'smplx', record['gender'],
                             n_samples_lbs=0, verbose=True, coarse=True,
                             approximate_center=True, swap_axes=False)
    creator.body_model = smplx.create(str(body_models), 'smplx', gender=record['gender'],
                                     use_pca=False, flat_hand_mean=True)
    obj = creator._load_from_obj(output / 'stage1/template_uv.obj')
    with np.load(stage4 / 'smplx/train.npz') as archive:
        index = record['template_frame']
        sequence = {key: archive[key][index:index + 1] for key in archive.files if key != 'mocap_frame_rate'}
    body = {key: sequence[key] for key in ('betas', 'expression')}
    body.update(transl=sequence['trans'], global_orient=sequence['root_orient'],
                body_pose=sequence['pose_body'], jaw_pose=sequence['pose_jaw'],
                left_hand_pose=sequence['pose_hand'][:, :45], right_hand_pose=sequence['pose_hand'][:, 45:],
                leye_pose=sequence['pose_eye'][:, :3], reye_pose=sequence['pose_eye'][:, 3:])
    # The native Simulator parses OmegaConf CLI; our argparse flags are not its config.
    sys.argv = [sys.argv[0]]
    creator._add_posed_garment_raw(obj, output.name, body, str(checkpoint),
                                   n_relaxation_steps=30, pinned_indices=None, gender=record['gender'])
    ready.write_text(json.dumps(record, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--template-frame', type=int, default=0)
    parser.add_argument('--ccraft-data', type=Path, required=True)
    parser.add_argument('--body-model-root', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='Author pretrained checkpoint, not a resume checkpoint.')
    parser.add_argument('--cmu-root', type=Path, help='Original AMASS CMU SMPL sequences for alternating training.')
    parser.add_argument('--steps', type=int, help='TOTAL combined batches since pretrained initialization; defaults to 1000, or the saved target on resume.')
    parser.add_argument('--resume', nargs='?', const='latest',
                        help='Resume latest fitted checkpoint, or specify its path.')
    parser.add_argument('--save-every', type=int, default=10, help='Save every N completed sequence batches (default: 10).')
    parser.add_argument('--prepare-only', action='store_true', help='Export arrays and config; no GPU import or optimizer steps.')
    args = parser.parse_args()
    assert args.steps is None or (args.steps >= 2 and args.steps % 2 == 0), '--steps must be even and >=2.'
    assert args.save_every > 0, '--save-every must be positive.'
    assert not args.resume or not (args.prepare_only or args.checkpoint), (
        '--resume cannot be combined with --prepare-only or --checkpoint.')
    data, output, ccraft_data = args.data.resolve(), args.output.resolve(), args.ccraft_data.resolve()
    body_models = (args.body_model_root or ccraft_data / 'aux_data/body_models').resolve()
    checkpoint = (args.checkpoint or ccraft_data / 'trained_models/contourcraft.pth').resolve()
    manifest = inspect_inputs(data, output, args.template_frame)
    if args.resume:
        record = read_json(output / 'stage4/preparation.json')
        assert record['data'] == str(data) and record['output'] == str(output)
        assert record['ccraft_data'] == str(ccraft_data) and record['body_model_root'] == str(body_models)
        assert record.get('body_model', 'smplx') == manifest.get('body_model', 'smplx')
        assert record['frame_ids'] == manifest['splits']['train']['local_frame_ids']
        command, target_step, checkpoint = resume_training_command(output, args.resume, args.steps, args.save_every)
    else:
        steps = 1000 if args.steps is None else args.steps
    if not args.prepare_only:
        assert args.cmu_root is not None, '--cmu-root is required for the author training procedure.'
        body_model = manifest.get('body_model', 'smplx')
        gender = record['gender'] if args.resume else 'none'
        if body_model == 'smplx' and not args.resume:
            _, gender = convert_body(data / manifest['splits']['train']['directory'] / 'smplx',
                                     manifest['splits']['train']['local_frame_ids'], manifest['fps'])
        check_training_assets(ccraft_data, body_models, checkpoint, args.cmu_root.resolve(), gender, body_model)
        if not args.resume:
            assert not saved_checkpoints(output), 'Fitting checkpoints already exist; continue with --resume.'
    if not args.resume:
        record = prepare(data, output, manifest, args.template_frame, ccraft_data, body_models, checkpoint, steps)
    print(f"Prepared {len(record['frame_ids'])} training frames at {record['fps']} FPS in {output / 'stage4'}", flush=True)
    if args.prepare_only:
        return
    env = dict(os.environ, CCRAFT_DATA_ROOT=str(ccraft_data), CCRAFT_CMU_ROOT=str(args.cmu_root.resolve()),
               CCRAFT_PROJECT_DIR=str(Path(__file__).resolve().parent),
               CCRAFT_EXPERIMENT_ROOT=str(output / 'stage4/logs'), WANDB_MODE='offline')
    os.environ.update(env)
    if not args.resume:
        # End the import process before training so its CUDA context and caches are released.
        importer = multiprocessing.get_context('spawn').Process(
            target=import_garment, args=(output, record, body_models, checkpoint))
        importer.start()
        importer.join()
        assert importer.exitcode == 0, 'Author garment import failed; see its traceback above.'
        config = yaml.safe_load((output / 'stage4/finetune.yaml').read_text())
        target_step = config['experiment']['max_iter']
        command = [sys.executable, 'train.py', f'config={output / "stage4/finetune"}',
                   f'restart.training_origin={config["restart"]["step_start"]}',
                   f'experiment.save_checkpoint_every={args.save_every}',
                   f'experiment.save_checkpoint_every_wlong={args.save_every}']
    else:
        print(f'Resuming {checkpoint}; total target step {target_step}.', flush=True)
    print('Starting author train.py with local offline metric logging.', flush=True)
    run_training(command, env)
    checkpoints = saved_checkpoints(output)
    assert checkpoints, 'Author training returned without a fitting checkpoint.'
    final_step = int(checkpoints[-1].stem[5:])
    status = 'complete' if final_step >= target_step else 'interrupted'
    (output / 'stage4/fitting.json').write_text(json.dumps(dict(status=status, latest_checkpoint=str(checkpoints[-1]),
                                                              completed_step=final_step, target_step=target_step,
                                                              resumed=bool(args.resume), save_every=args.save_every,
                                                              cmu_root=str(args.cmu_root.resolve())), indent=2) + '\n')
    print(f'{status.capitalize()}; checkpoint: {checkpoints[-1]}', flush=True)


if __name__ == '__main__':
    main()
