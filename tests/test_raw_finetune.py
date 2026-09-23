"""Exercise raw target meshes through the real ContourCraft fine-tuning loader."""
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from datasets.finetune import Config, RawBodySequenceLoader, create_loader

# Template command (activate ccraft; from ContourCraft):
# CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_raw_finetune.py -v


def create_fixture(root: Path) -> tuple[Config, np.ndarray, np.ndarray]:
    body_root, registrations, garments = (root / name for name in ('body', 'registrations', 'garments'))
    for directory in (body_root, registrations, garments):
        directory.mkdir()
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    plane = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    movement = np.arange(5, dtype=np.float32)[:, None, None] * np.asarray([0, 0, 0.04])
    body = np.asarray(plane[None] + movement, dtype=np.float32)
    cloth = body + np.asarray([0, 0, 0.1], dtype=np.float32)
    np.savez(body_root / 'train.npz', verts=body, faces=faces, mocap_frame_rate=25)
    with (registrations / 'train.pkl').open('wb') as stream:
        pickle.dump({'vertices': cloth, 'faces': faces}, stream)
    with (garments / 'garment.pkl').open('wb') as stream:
        pickle.dump({'rest_pos': cloth[0] * 2, 'faces': faces,
                     'node_type': np.zeros((4, 1), dtype=np.int64),
                     'center': [0], 'coarse_edges': {0: {0: np.asarray([[0, 2], [1, 3]])}}}, stream)
    config = Config(train_split_path='', valid_split_path='', smpl_dir='', garment_dict_dir='',
                    body_model='raw', body_sequence_root=str(body_root),
                    registration_root=str(registrations), chronological_registration=True,
                    n_coarse_levels=1, fps=25)
    config.garment_dicts_dir = str(garments)
    return config, body, cloth


class RawFinetuneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config, self.body, self.cloth = create_fixture(Path(self.temporary.name))

    def test_raw_sample_preserves_chronology_topology_and_explicit_rest_geometry(self) -> None:
        with patch('datasets.finetune.build_smpl_bygender', side_effect=AssertionError('SMPL model loaded')), \
             patch('datasets.finetune.make_obstacle_dict', side_effect=AssertionError('SMPL labels loaded')):
            loader = create_loader(self.config)
            sample = loader.load_sample('garment', 'train', 'none')
        for node, vertices in (('obstacle', self.body), ('cloth', self.cloth)):
            for key, frame in (('prev_pos', 0), ('pos', 1), ('target_pos', 2)):
                np.testing.assert_array_equal(sample[node][key].numpy(), vertices[frame])
            np.testing.assert_array_equal(sample[node].lookup.numpy(), vertices[2:].transpose(1, 0, 2))
            np.testing.assert_array_equal(sample[node].faces_batch.numpy().T, [[0, 1, 2], [0, 2, 3]])
        np.testing.assert_array_equal(sample['cloth'].rest_pos.numpy(), self.cloth[0] * 2)
        np.testing.assert_array_equal(sample['obstacle'].vertex_type.numpy(), np.ones((4, 1)))
        np.testing.assert_array_equal(sample['obstacle'].vertex_level.numpy(), np.zeros((4, 1)))
        np.testing.assert_array_equal(sample['cloth'].vertex_type.numpy(), np.zeros((4, 1)))
        self.assertEqual(tuple(sample['cloth', 'coarse_edge0', 'cloth'].edge_index.shape), (2, 4))
        self.assertEqual(loader.garment_builder.garment_smpl_model_dict, {})

    def test_raw_sequence_rejects_resampling_short_sequences_and_invalid_topology(self) -> None:
        path = Path(self.config.body_sequence_root) / 'invalid.npz'
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        for name, changes in (
            ('fps', {'mocap_frame_rate': 30}),
            ('frames', {'verts': self.body[:2]}),
            ('topology', {'faces': faces[None]}),
            ('index', {'faces': faces + 4}),
            ('finite', {'verts': self.body * np.nan}),
        ):
            record = {'verts': self.body, 'faces': faces, 'mocap_frame_rate': 25}
            record.update(changes)
            np.savez(path, **record)
            with self.subTest(invalid=name), self.assertRaises(AssertionError):
                RawBodySequenceLoader(self.config).load_sequence(path)

    def test_raw_sample_rejects_mismatched_body_and_cloth_lengths(self) -> None:
        path = Path(self.config.registration_root) / 'train.pkl'
        with path.open('rb') as stream:
            registration = pickle.load(stream)
        registration['vertices'] = registration['vertices'][:-1]
        with path.open('wb') as stream:
            pickle.dump(registration, stream)
        with self.assertRaisesRegex(AssertionError, 'frame counts differ'):
            create_loader(self.config).load_sample('garment', 'train', 'none')


if __name__ == '__main__':
    unittest.main()
