"""CPU checks for the real Gaussian Garments conversion and author configuration."""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np
import yaml

import fit_gaussian_garments as fitting

# Template command (activate gaugar; from ContourCraft; CPU only):
# CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_gaussian_garments.py -v


def write_mesh(path: Path, frame: int, faces: str = 'f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n') -> None:
    """Write a moving two-triangle garment with non-degenerate UVs."""
    path.write_text(f'v 0 0 {frame}\nv 1 0 {frame}\nv 1 1 {frame}\nv 0 1 {frame}\n'
                    'vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n' + faces)


class GaussianGarmentsConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.data, self.output = self.base / 'data', self.base / 'experiment'
        self.body = self.data / 'input/subject/train/smplx'
        self.body.mkdir(parents=True)
        self.meshes = self.output / 'stage2/train/meshes'
        self.meshes.mkdir(parents=True)
        (self.output / 'stage1').mkdir()
        (self.output / 'stage2/Template').mkdir()
        write_mesh(self.output / 'stage1/template_uv.obj', 0)
        write_mesh(self.output / 'stage2/Template/template.obj', 0)
        self.manifest: dict[str, Any] = {
            'status': 'complete', 'component': 'clothtransformer_gaussian_garments_export',
            'body_model': 'smplx', 'subject': 'subject', 'fps': 25,
            'splits': {
                'train': {'directory': 'input/subject/train', 'local_frame_ids': [0, 1, 2],
                          'global_frame_ids': [0, 1, 2], 'global_timestamps': [0.0, 0.04, 0.08]},
                'heldout': {'directory': 'never_read_heldout', 'global_frame_ids': [3, 4]},
            },
        }
        self.write_manifest()
        (self.output / 'experiment.json').write_text(json.dumps({
            'data': str(self.data), 'subject': 'subject', 'template_frame': 0,
        }))
        sizes = {'betas': 10, 'expression': 10, 'global_orient': 3, 'body_pose': 63,
                 'jaw_pose': 3, 'leye_pose': 3, 'reye_pose': 3, 'left_hand_pose': 45,
                 'right_hand_pose': 45, 'transl': 3}
        self.records: list[dict[str, Any]] = []
        for frame in range(3):
            record: dict[str, Any] = {
                name: (np.arange(size, dtype=np.float32) + offset * 100 + frame * 1000) / 10000
                for offset, (name, size) in enumerate(sizes.items())
            }
            record.update(gender='female', use_pca=False, flat_hand_mean=True)
            self.records.append(record)
            with (self.body / f'{frame:05d}.pkl').open('wb') as stream:
                pickle.dump(record, stream)
            write_mesh(self.meshes / f'frame_{frame:05d}.obj', frame)

    def write_manifest(self) -> None:
        (self.data / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_body_conversion_preserves_full_pose_expression_translation_and_timing(self) -> None:
        converted, gender = fitting.convert_body(self.body, [0, 1, 2], 25)
        self.assertEqual(gender, 'female')
        self.assertEqual(float(converted['mocap_frame_rate']), 25)
        names = {'betas': 'betas', 'expression': 'expression', 'trans': 'transl',
                 'root_orient': 'global_orient', 'pose_body': 'body_pose', 'pose_jaw': 'jaw_pose'}
        for destination, source in names.items():
            with self.subTest(field=destination):
                np.testing.assert_array_equal(converted[destination], np.stack([r[source] for r in self.records]))
                self.assertEqual(converted[destination].dtype, np.float32)
        self.assertEqual(converted['pose_hand'].shape, (3, 90))
        for frame, record in enumerate(self.records):
            np.testing.assert_array_equal(converted['pose_hand'][frame, :45], record['left_hand_pose'])
            np.testing.assert_array_equal(converted['pose_hand'][frame, 45:], record['right_hand_pose'])
            np.testing.assert_array_equal(converted['pose_eye'][frame, :3], record['leye_pose'])
            np.testing.assert_array_equal(converted['pose_eye'][frame, 3:], record['reye_pose'])
        self.records[1]['use_pca'] = True
        with (self.body / '00001.pkl').open('wb') as stream:
            pickle.dump(self.records[1], stream)
        with self.assertRaises(AssertionError):
            fitting.convert_body(self.body, [0, 1, 2], 25)

    def test_input_validation_rejects_missing_raw_reordered_and_leaking_frames(self) -> None:
        self.assertEqual(fitting.inspect_inputs(self.data, self.output, 0), self.manifest)
        self.assertFalse((self.data / 'never_read_heldout').exists())
        original = copy.deepcopy(self.manifest)
        for name in ('raw', 'frame_order', 'time', 'heldout_overlap'):
            self.manifest = copy.deepcopy(original)
            if name == 'raw':
                self.manifest['body_model'] = 'raw'
            elif name == 'frame_order':
                self.manifest['splits']['train']['local_frame_ids'] = [1, 0, 2]
            elif name == 'time':
                self.manifest['splits']['train']['global_timestamps'][2] = 0.12
            else:
                self.manifest['splits']['heldout']['global_frame_ids'] = [2, 3]
            self.write_manifest()
            with self.subTest(invalid=name), self.assertRaises(AssertionError):
                fitting.inspect_inputs(self.data, self.output, 0)
        self.manifest = original
        self.write_manifest()
        with self.assertRaisesRegex(AssertionError, 'template frame'):
            fitting.inspect_inputs(self.data, self.output, 1)
        write_mesh(self.meshes / 'frame_00003.obj', 3)
        with self.assertRaisesRegex(AssertionError, 'Registration frames differ'):
            fitting.inspect_inputs(self.data, self.output, 0)
        (self.meshes / 'frame_00003.obj').unlink()
        (self.meshes / 'frame_00002.obj').unlink()
        with self.assertRaisesRegex(AssertionError, 'Incomplete training registration'):
            fitting.inspect_inputs(self.data, self.output, 0)
        self.assertFalse((self.output / 'stage4').exists())

    def write_raw_body(self) -> Path:
        """Write indexed moving PLYs without creating parametric body inputs."""
        source = self.body.parent / 'body_mesh'
        source.mkdir()
        for frame in range(3):
            (source / f'{frame:05d}.ply').write_text(
                'ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\n'
                'property float z\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n'
                f'0 0 {frame}\n1 0 {frame}\n1 1 {frame}\n0 1 {frame}\n3 0 1 2\n3 0 2 3\n')
        self.manifest['body_model'] = 'raw'
        self.write_manifest()
        return source

    def test_raw_conversion_and_preparation_preserve_geometry_and_exclude_heldout(self) -> None:
        source = self.write_raw_body()
        self.assertEqual(fitting.inspect_inputs(self.data, self.output, 0), self.manifest)
        body = fitting.convert_raw_body(source, [0, 1, 2], 25)
        np.testing.assert_array_equal(body['verts'][:, :, 2], [[0] * 4, [1] * 4, [2] * 4])
        np.testing.assert_array_equal(body['faces'], [[0, 1, 2], [0, 2, 3]])
        record = fitting.prepare(self.data, self.output, self.manifest, 0,
                                 self.base / 'assets', self.base / 'models', self.base / 'checkpoint.pth', 2)
        self.assertEqual(record['body_model'], 'raw')
        self.assertEqual(record['gender'], 'none')
        stage4 = self.output / 'stage4'
        self.assertFalse((stage4 / 'smplx').exists())
        with np.load(stage4 / 'body_mesh/train.npz') as archive:
            np.testing.assert_array_equal(archive['verts'], body['verts'])
            np.testing.assert_array_equal(archive['faces'], body['faces'])
            self.assertEqual(float(archive['mocap_frame_rate']), 25)
        config = yaml.safe_load((stage4 / 'finetune.yaml').read_text())
        target = config['dataloaders']['finetune']['dataset']['finetune']
        self.assertEqual(target['body_model'], 'raw')
        self.assertFalse(target['omit_hands'])
        self.assertFalse(target['pinned_verts'])
        author = yaml.safe_load((Path(fitting.__file__).parent / 'configs/finetune/base.yaml').read_text())
        for section in ('criterions', 'material_stack', 'model'):
            self.assertEqual(config[section], author[section])
        self.assertEqual(config['dataloaders']['long']['dataset']['ccraft']['sequence_loader'], 'cmu_npz_smpl')
        self.assertFalse((self.data / 'never_read_heldout').exists())
        path = source / '00001.ply'
        path.write_text(path.read_text().replace('3 0 1 2', '3 0 2 1'))
        with self.assertRaisesRegex(AssertionError, 'Body topology changed'):
            fitting.convert_raw_body(source, [0, 1, 2], 25)

    def test_raw_import_uses_posed_stage1_template_as_author_rest_reference(self) -> None:
        self.write_raw_body()
        arguments = (self.data, self.output, self.manifest, 0, self.base / 'assets',
                     self.base / 'models', self.base / 'checkpoint.pth', 2)
        # Distinct stage2 geometry must not replace the author's stage1 reference.
        write_mesh(self.output / 'stage2/Template/template.obj', 10)
        record = fitting.prepare(*arguments)
        self.assertEqual(record['rest_mesh'], str(self.output / 'stage1/template_uv.obj'))
        fitting.import_garment(self.output, record, self.base / 'absent_models', self.base / 'absent_checkpoint')
        with (self.output / 'stage4/garment_dicts/experiment.pkl').open('rb') as stream:
            garment = pickle.load(stream)
        np.testing.assert_array_equal(garment['rest_pos'], [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        self.assertNotIn('lbs', garment)
        self.assertIn('coarse_edges', garment)
        self.assertTrue((garment['node_type'] == 0).all())

    def test_real_obj_conversion_rejects_changed_face_and_uv_order(self) -> None:
        converted = fitting.convert_registration(self.output, [0, 1, 2])
        self.assertEqual(converted['vertices'].shape, (3, 4, 3))
        np.testing.assert_array_equal(converted['vertices'][:, :, 2], [[0] * 4, [1] * 4, [2] * 4])
        np.testing.assert_array_equal(converted['faces'], [[0, 1, 2], [0, 2, 3]])
        np.testing.assert_array_equal(converted['pred'], converted['vertices'])
        np.testing.assert_array_equal(converted['cloth_faces'], converted['faces'])
        self.assertEqual(converted['uv_coords'].shape, (3, 4, 2))
        for faces in ('f 1/1 3/3 4/4\nf 1/1 2/2 3/3\n',
                      'f 1/2 2/1 3/3\nf 1/2 3/3 4/4\n'):
            write_mesh(self.meshes / 'frame_00001.obj', 1, faces)
            with self.subTest(faces=faces), self.assertRaises(AssertionError):
                fitting.convert_registration(self.output, [0, 1, 2])

    def test_config_preserves_author_losses_optimizer_and_regularization(self) -> None:
        source = Path(fitting.__file__).parent / 'configs/finetune/base.yaml'
        author = yaml.safe_load(source.read_text())
        config = fitting.make_config(self.output / 'stage4', self.base / 'ccraft-data',
                                     self.base / 'models', self.base / 'checkpoint.pth', 25, 2)
        for section in ('criterions', 'material_stack', 'model'):
            self.assertEqual(config[section], author[section])
        runner = config['runner']['finetune']
        self.assertEqual(runner['optimizer'], author['runner']['finetune']['optimizer'])
        self.assertEqual(runner['regular_ts'], author['runner']['finetune']['regular_ts'])
        self.assertEqual(runner['initial_ts'], author['runner']['finetune']['initial_ts'])
        self.assertEqual(runner['finetune_ts'], 1 / 25)
        self.assertNotEqual(runner['finetune_ts'], runner['regular_ts'])
        original_long = copy.deepcopy(author['dataloaders']['long'])
        original_long['dataset']['ccraft']['body_model_root'] = str(self.base / 'models')
        self.assertEqual(config['dataloaders']['long'], original_long)
        self.assertEqual(config['restart']['step_start'], author['restart']['step_start'])
        self.assertEqual(config['experiment']['max_iter'], author['restart']['step_start'] + 2)
        self.assertTrue(config['experiment']['use_writer'])
        target = config['dataloaders']['finetune']['dataset']['finetune']
        self.assertEqual(target['fps'], 25)
        self.assertTrue(target['preserve_wrist_pose'])
        self.assertTrue(target['flat_hand_mean'])
        self.assertTrue(target['chronological_registration'])
        self.assertEqual(target['sequence_loader'], author['dataloaders']['finetune']['dataset']['finetune']['sequence_loader'])

    def test_training_assets_support_author_comma_separated_garments(self) -> None:
        ccraft_data, models, cmu = self.base / 'ccraft-data', self.base / 'models', self.base / 'CMU'
        checkpoint = ccraft_data / 'trained_models/contourcraft.pth'
        sequence = cmu / '01/01_01_poses.npz'
        garment_root = ccraft_data / 'aux_data/garment_dicts/smpl'
        for path in (checkpoint, sequence, models / 'smplx/SMPLX_FEMALE.npz', models / 'smpl/SMPL_MALE.pkl',
                     ccraft_data / 'aux_data/smpl_aux.pkl', garment_root / 'shirt.pkl', garment_root / 'pants.pkl'):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        split = ccraft_data / 'aux_data/datasplits/train_ccraft.csv'
        split.parent.mkdir()
        with split.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('id', 'length', 'garment', 'gender'))
            writer.writerow(('01/01_01_poses', 100, 'shirt,pants', 'male'))
            writer.writerow(('01/01_01_poses.npz', 100, 'shirt.pkl, pants.pkl', 'male'))
        fitting.check_training_assets(ccraft_data, models, checkpoint, cmu, 'female')
        sequence.unlink()
        with self.assertRaisesRegex(AssertionError, 'Missing AMASS CMU regularization sequence'):
            fitting.check_training_assets(ccraft_data, models, checkpoint, cmu, 'female')
        sequence.touch()
        (garment_root / 'pants.pkl').unlink()
        with self.assertRaisesRegex(AssertionError, 'pants.pkl'):
            fitting.check_training_assets(ccraft_data, models, checkpoint, cmu, 'female')

    def test_prepare_only_cli_writes_training_artifacts_without_assets_or_heldout(self) -> None:
        script = Path(fitting.__file__).resolve()
        command = [sys.executable, str(script), '--data', str(self.data), '--output', str(self.output),
                   '--ccraft-data', str(self.base / 'absent_ccraft_assets'), '--steps', '2', '--prepare-only']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        result = subprocess.run(command, cwd=script.parent, env=env, check=True, capture_output=True, text=True)
        self.assertIn('Prepared 3 training frames at 25 FPS', result.stdout)
        stage4 = self.output / 'stage4'
        with np.load(stage4 / 'smplx/train.npz') as archive:
            self.assertEqual(archive['pose_hand'].shape, (3, 90))
            np.testing.assert_array_equal(archive['expression'], np.stack([r['expression'] for r in self.records]))
            np.testing.assert_array_equal(archive['trans'], np.stack([r['transl'] for r in self.records]))
            self.assertEqual(float(archive['mocap_frame_rate']), 25)
        with (stage4 / 'registrations/train.pkl').open('rb') as stream:
            registration = pickle.load(stream)
        np.testing.assert_array_equal(registration['vertices'][:, :, 2], [[0] * 4, [1] * 4, [2] * 4])
        with (stage4 / 'train.csv').open(newline='') as stream:
            self.assertEqual(list(csv.DictReader(stream)), [
                {'id': 'train', 'length': '3', 'garment': 'experiment', 'gender': 'female'}])
        with (stage4 / 'valid.csv').open(newline='') as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])
        self.assertEqual(yaml.safe_load((stage4 / 'finetune.yaml').read_text())['runner']['finetune']['finetune_ts'], 0.04)
        self.assertEqual(json.loads((stage4 / 'preparation.json').read_text())['frame_ids'], [0, 1, 2])
        self.assertFalse((stage4 / 'checkpoints').exists())
        self.assertFalse((stage4 / 'garment_import.json').exists())
        self.assertFalse((self.data / 'never_read_heldout').exists())

    def make_resume_fixture(self, step: int, target: int = 46000) -> Path:
        """Create a serialized launcher fixture; actual state restore is tested separately."""
        import torch

        stage4 = self.output / 'stage4'
        if not stage4.exists():
            record = fitting.prepare(self.data, self.output, self.manifest, 0,
                                     self.base / 'assets', self.base / 'models', self.base / 'checkpoint.pth', 1000)
            (stage4 / 'garment_import.json').write_text(json.dumps(record))
        config = yaml.safe_load((stage4 / 'finetune.yaml').read_text())
        config['experiment']['max_iter'] = target
        checkpoint = stage4 / 'checkpoints' / f'step_{step:010d}.pth'
        checkpoint.parent.mkdir(exist_ok=True)
        torch.save(dict(config=config, training_module={}, optimizer={}, optimizer_material={}, scheduler={},
                        material_stack={f'materials.{self.output.name}.v_mass': torch.ones(4)}), checkpoint)
        return checkpoint

    def test_resume_uses_numeric_latest_and_saved_target_without_repreparing(self) -> None:
        self.make_resume_fixture(45010)
        checkpoint = self.make_resume_fixture(45021, target=47000)
        (checkpoint.parent / 'step_9999999999.pth.tmp').touch()
        before = (self.output / 'stage4/finetune.yaml').read_text()
        command, target, selected = fitting.resume_training_command(self.output, 'latest', None, 3)
        self.assertEqual(selected, checkpoint)
        self.assertEqual(target, 47000)
        for argument in ('restart.resume=True', 'restart.step_start=45021',
                         'restart.training_origin=45000', 'experiment.max_iter=47000',
                         'experiment.save_checkpoint_every=3'):
            self.assertIn(argument, command)
        self.assertEqual((self.output / 'stage4/finetune.yaml').read_text(), before)
        command, target, selected = fitting.resume_training_command(self.output, str(checkpoint), 200, 10)
        self.assertEqual(target, 45200)
        self.assertIn('experiment.max_iter=45200', command)

    def test_resume_rejects_missing_incomplete_foreign_or_outdated_checkpoints(self) -> None:
        import torch

        with self.assertRaisesRegex(AssertionError, 'No saved fitting checkpoint'):
            fitting.select_resume_checkpoint(self.output, 'latest')
        first = self.make_resume_fixture(45010)
        latest = self.make_resume_fixture(45020)
        with self.assertRaisesRegex(AssertionError, 'Newer checkpoints'):
            fitting.select_resume_checkpoint(self.output, str(first))
        with self.assertRaisesRegex(AssertionError, 'larger TOTAL budget'):
            fitting.resume_training_command(self.output, 'latest', 20, 10)
        saved = torch.load(latest, weights_only=False)
        for invalid in ('optimizer_material', 'experiment', 'global_step'):
            state = copy.deepcopy(saved)
            if invalid == 'optimizer_material':
                state.pop(invalid)
            elif invalid == 'experiment':
                state['config']['checkpoints_dir'] = str(self.base / 'different/checkpoints')
            else:
                state['global_step'] = 45019
            torch.save(state, latest)
            with self.subTest(invalid=invalid), self.assertRaises(AssertionError):
                fitting.resume_training_command(self.output, 'latest', None, 10)

    def test_training_wrapper_forwards_stop_and_waits(self) -> None:
        import signal

        handlers = {}

        def install(signum: int, handler: Any) -> Any:
            previous = handlers.get(signum, signal.SIG_DFL)
            handlers[signum] = handler
            return previous

        with patch('fit_gaussian_garments.subprocess.Popen') as popen, \
                patch('fit_gaussian_garments.signal.signal', side_effect=install):
            process = popen.return_value
            process.poll.return_value = None

            def wait() -> int:
                handlers[signal.SIGINT](signal.SIGINT, None)
                return 0

            process.wait.side_effect = wait
            fitting.run_training(['python', 'train.py'], {})
            process.send_signal.assert_called_once_with(signal.SIGINT)
            process.kill.assert_not_called()
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            self.assertEqual(handlers[signal.SIGINT], signal.SIG_DFL)


if __name__ == '__main__':
    unittest.main()
