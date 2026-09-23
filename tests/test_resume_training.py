"""State restoration and completed-step boundaries for interrupted fine-tuning."""
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch
from torch_geometric.data import Batch, HeteroData

from material.mstack import MaterialStack
from runners.finetune import run_epoch
from utils.arguments import load_from_checkpoint
from utils.common import capture_rng_state, restore_rng_state, save_checkpoint

# Template command (activate ccraft; from ContourCraft):
# CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_resume_training.py -v


def make_modules() -> tuple[torch.nn.Module, dict]:
    runner = torch.nn.Linear(2, 1)
    material = MaterialStack.__new__(MaterialStack)
    torch.nn.Module.__init__(material)
    material.materials = torch.nn.ModuleDict({'garment': torch.nn.Linear(2, 1)})
    optimizer = torch.optim.Adam(runner.parameters(), lr=0.01)
    material_optimizer = torch.optim.Adam(material.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    return runner, {'optimizer': optimizer, 'scheduler': scheduler,
                    'material_stack': material, 'optimizer_material': material_optimizer}


def update(runner: torch.nn.Module, aux: dict) -> None:
    aux['optimizer'].zero_grad()
    aux['optimizer_material'].zero_grad()
    x = torch.tensor([[0.2, 0.4]])
    loss = (runner(x) + aux['material_stack'].materials['garment'](x)).square().sum()
    loss.backward()
    aux['optimizer'].step()
    aux['optimizer_material'].step()
    aux['scheduler'].step()


def config_for(path: Path) -> object:
    return OmegaConf.create({
        'step_start': 0, 'restart': {'checkpoint_path': str(path), 'step_start': 45001,
                                    'training_origin': 45000, 'resume': True, 'load_optimizer': True},
        'material_stack': {'mstack': {'material': 'fixture'}},
        'dataloaders': {'finetune': {'dataset': {'finetune': {
            'garment_dicts_dir': '/fixture/garments', 'body_sequence_root': '/fixture/body',
            'registration_root': '/fixture/registrations', 'body_model': 'raw'}}}},
    })


class ResumeTrainingTests(unittest.TestCase):
    def test_resume_matches_next_update_and_retains_material_optimizer_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'step_0000045001.pth'
            config = config_for(path)
            original, original_aux = make_modules()
            update(original, original_aux)
            save_checkpoint(original, original_aux, config, path, global_step=45001)
            self.assertFalse(path.with_name(path.name + '.tmp').exists())
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved['global_step'], 45001)
            self.assertEqual(saved['config']['step_start'], 45001)
            restored, restored_aux = make_modules()
            parameters_before = list(restored_aux['material_stack'].parameters())
            restored, restored_aux = load_from_checkpoint(config, restored, restored_aux)
            self.assertEqual(config.step_start, 45001)
            self.assertTrue(all(a is b for a, b in zip(parameters_before, restored_aux['material_stack'].parameters())))
            optimizer_parameters = restored_aux['optimizer_material'].param_groups[0]['params']
            self.assertTrue(all(a is b for a, b in zip(parameters_before, optimizer_parameters)))
            update(original, original_aux)
            update(restored, restored_aux)
            for left, right in zip(original.parameters(), restored.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            for left, right in zip(original_aux['material_stack'].parameters(), restored_aux['material_stack'].parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertEqual(original_aux['scheduler'].state_dict(), restored_aux['scheduler'].state_dict())

    def test_resume_rejects_missing_training_state_and_wrong_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'step_0000045001.pth'
            config = config_for(path)
            runner, aux = make_modules()
            save_checkpoint(runner, aux, config, path, global_step=45001)
            config.dataloaders.finetune.dataset.finetune.registration_root = '/other'
            with self.assertRaisesRegex(AssertionError, 'experiment mismatch'):
                load_from_checkpoint(config, runner, aux)
            state = torch.load(path, weights_only=False)
            del state['optimizer_material']
            torch.save(state, path)
            with self.assertRaisesRegex(AssertionError, 'Incomplete resume checkpoint'):
                load_from_checkpoint(config, runner, aux)

    def test_rng_state_restores_python_numpy_and_torch(self) -> None:
        state = capture_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng_state(state)
        actual = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

    def test_existing_numeric_checkpoint_uses_explicit_completed_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'step_0000045001.pth'
            config = config_for(path)
            runner, aux = make_modules()
            save_checkpoint(runner, aux, config, path, global_step=45001)
            state = torch.load(path, weights_only=False)
            state.pop('global_step')
            state.pop('rng_state')
            state['config']['step_start'] = 45000
            torch.save(state, path)
            load_from_checkpoint(config, runner, aux)
            self.assertEqual(config.step_start, 45001)
            self.assertNotIn('_resume_rng_state', aux)

    def test_odd_resume_starts_long_and_stop_finishes_only_current_batch(self) -> None:
        sample = HeteroData()
        sample['cloth'].pos = torch.zeros(3, 3)
        batch = Batch.from_data_list([sample])
        config = OmegaConf.create({'config': 'fixture', 'device': 'cpu',
                                  'restart': {'training_origin': 45000, 'step_start': 45001},
                                  'experiment': {'max_iter': 45004}})
        runner, aux = make_modules()
        runner = SimpleNamespace(model=runner)
        dataloaders = {'long': [batch] * 3, 'finetune': [batch] * 3}
        calls = []

        def long_step(*args: object) -> dict:
            calls.append('long')
            return {}

        def ft_step(*args: object) -> dict:
            calls.append('ft')
            return {}

        with patch('runners.finetune.step_long', side_effect=long_step), \
             patch('runners.finetune.step_ft', side_effect=ft_step), \
             patch('runners.finetune.make_checkpoint') as save:
            completed = run_epoch(runner, aux, dataloaders, config, global_step=45001)
            self.assertEqual(completed, 45004)
            self.assertEqual(calls, ['long', 'ft', 'long'])
            self.assertEqual([entry.args[-1] for entry in save.call_args_list], [45002, 45003, 45004])
            calls.clear()
            completed = run_epoch(runner, aux, dataloaders, config, global_step=45001,
                                  stop_requested=lambda: len(calls) > 0)
            self.assertEqual(completed, 45002)
            self.assertEqual(calls, ['long'])


if __name__ == '__main__':
    unittest.main()
