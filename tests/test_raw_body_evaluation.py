"""Raw collider evaluation preserves motion and topology across held-out boundaries."""
from __future__ import annotations

import copy
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest

import numpy as np

from evaluate_gaussian_garments import build_sample, full_body_motion

# Template command (ccraft environment, from ContourCraft):
# CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_raw_body_evaluation.py -v


def write_body(path: Path, frame: int, reverse: bool = False) -> None:
    """Write a translated tetrahedron without requiring a parametric body model."""
    face = '3 0 2 1' if reverse else '3 0 1 2'
    path.write_text('ply\nformat ascii 1.0\nelement vertex 4\n'
                    'property float x\nproperty float y\nproperty float z\n'
                    'element face 4\nproperty list uchar int vertex_indices\nend_header\n'
                    f'0 0 {frame}\n1 0 {frame}\n0 1 {frame}\n0 0 {frame + 1}\n'
                    f'{face}\n3 0 3 1\n3 0 2 3\n3 1 3 2\n')


class RawBodyEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data = Path(temporary.name)
        self.manifest: dict[str, Any] = {
            'body_model': 'raw', 'fps': 25, 'frame_ids': list(range(5)),
            'evaluation_frame_ids': [3, 4], 'timestamps': [frame / 25 for frame in range(5)],
            'splits': {},
        }
        for name, ids in (('train', [0, 1, 2]), ('heldout', [3, 4])):
            directory = f'{name}/body_mesh'
            (self.data / directory).mkdir(parents=True)
            self.manifest['splits'][name] = {
                'directory': name, 'local_frame_ids': list(range(len(ids))),
                'global_frame_ids': ids, 'global_timestamps': [frame / 25 for frame in ids],
            }
            for local, frame in enumerate(ids):
                write_body(self.data / directory / f'{local:05d}.ply', frame)

    def test_complete_body_motion_uses_no_smpl_or_garment_observations(self) -> None:
        body, gender = full_body_motion(self.data, self.manifest)
        self.assertEqual(gender, 'none')
        self.assertEqual(body['verts'].shape, (5, 4, 3))
        self.assertEqual(body['faces'].shape, (4, 3))
        np.testing.assert_array_equal(body['verts'][:, 0, 2], np.arange(5))
        self.assertEqual(float(body['mocap_frame_rate']), 25)

    def test_topology_changes_at_boundary_are_rejected(self) -> None:
        for local, frame in enumerate((3, 4)):
            write_body(self.data / 'heldout/body_mesh' / f'{local:05d}.ply', frame, reverse=True)
        with self.assertRaisesRegex(AssertionError, 'topology changes'):
            full_body_motion(self.data, self.manifest)

    def test_native_evaluation_sample_has_only_initial_cloth_and_full_raw_motion(self) -> None:
        from datasets.finetune import Config

        output = self.data / 'garment'
        stage4 = output / 'stage4'
        (stage4 / 'registrations').mkdir(parents=True)
        (stage4 / 'garments').mkdir()
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        rest = np.asarray([[0, 0, -1], [1, 0, -1], [1, 1, -1], [0, 1, -1]], dtype=np.float32)
        cloth = rest[None] + np.arange(3, dtype=np.float32)[:, None, None]
        with (stage4 / 'registrations/train.pkl').open('wb') as stream:
            pickle.dump({'vertices': cloth, 'faces': faces}, stream)
        with (stage4 / 'garments/garment.pkl').open('wb') as stream:
            pickle.dump({'rest_pos': rest, 'faces': faces,
                         'node_type': np.zeros((4, 1), dtype=np.int64), 'center': [0],
                         'coarse_edges': {0: {0: np.asarray([[0, 2], [1, 3]])}}}, stream)
        dataset = Config(train_split_path='', valid_split_path='', smpl_dir='', garment_dict_dir='',
                         body_model='raw', body_sequence_root=str(stage4 / 'body_mesh'),
                         chronological_registration=True, n_coarse_levels=1, fps=25)
        dataset.garment_dicts_dir = str(stage4 / 'garments')
        config = SimpleNamespace(
            dataloaders=SimpleNamespace(finetune=SimpleNamespace(dataset=SimpleNamespace(finetune=dataset))),
            runner=SimpleNamespace(finetune=SimpleNamespace(finetune_ts=1 / 25)), device='cpu')
        sample = build_sample(self.data, output, self.manifest, config, 'none')
        body, _ = full_body_motion(self.data, self.manifest)
        np.testing.assert_array_equal(sample['obstacle'].prev_pos.numpy(), body['verts'][0])
        np.testing.assert_array_equal(sample['obstacle'].pos.numpy(), body['verts'][1])
        np.testing.assert_array_equal(sample['obstacle'].lookup.numpy(), body['verts'][2:].transpose(1, 0, 2))
        np.testing.assert_array_equal(sample['cloth'].prev_pos.numpy(), cloth[0])
        np.testing.assert_array_equal(sample['cloth'].pos.numpy(), cloth[1])
        np.testing.assert_array_equal(sample['cloth'].target_pos.numpy(), cloth[1])
        np.testing.assert_array_equal(sample['cloth'].lookup.numpy(), np.repeat(cloth[1, :, None], 3, axis=1))
        np.testing.assert_array_equal(sample['cloth'].rest_pos.numpy(), rest)

    def test_gaps_reordering_and_time_discontinuity_are_rejected(self) -> None:
        for invalid in ('time', 'global_order', 'local_order', 'evaluation_range'):
            manifest = copy.deepcopy(self.manifest)
            if invalid == 'time':
                manifest['splits']['heldout']['global_timestamps'] = [0.16, 0.20]
            elif invalid == 'global_order':
                manifest['splits']['heldout']['global_frame_ids'] = [4, 3]
            elif invalid == 'local_order':
                manifest['splits']['heldout']['local_frame_ids'] = [1, 0]
            else:
                manifest['evaluation_frame_ids'] = [2, 3, 4]
            with self.subTest(invalid=invalid), self.assertRaises(AssertionError):
                full_body_motion(self.data, manifest)


if __name__ == '__main__':
    unittest.main()
