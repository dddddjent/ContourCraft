import importlib
import os
from dataclasses import dataclass, MISSING
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from utils.datasets import build_smpl_bygender, make_obstacle_dict
from utils.io import pickle_load
from utils.defaults import DEFAULTS
from datasets.ccraft import GarmentBuilder as GarmentBuilderBase 
from datasets.ccraft import BodyBuilder as BodyBuilder


@dataclass
class Config:
    train_split_path: str = MISSING
    valid_split_path: str = MISSING
    smpl_dir: str = MISSING
    garment_dict_dir: str = MISSING
    registration_root: Optional[str] = None
    body_sequence_root: Optional[str] = None
    smplx_segmentation_file: Optional[str] = None
    body_model: str = 'smplx'  # 'smplx' for parameter sequences, 'raw' for mesh sequences.

    # data_root: Optional[str] = None # do not set
    # temp_data_root: Optional[str] = None # do not set
    obstacle_dict_file: Optional[str] = None

    body_model_root: str = 'body_models'  # Path to the directory containg body model files, should contain `smpl` and/or `smplx` sub-directories. Relative to $DEFAULTS.data_root/aux_data/
    model_type: str = 'smpl'  # Type of the body model ('smpl' or 'smplx')
    sequence_loader: str = 'cmu_npz_smpl'  # Name of the sequence loader to use 

    swap_axes: bool = False
    pinned_verts: bool = True
    use_betas_for_restpos: bool = False
    n_coarse_levels: int = 4
    separate_arms: bool = True
    # Exported fitted colliders use full axis-angle hands and unchanged wrists.
    flat_hand_mean: bool = False
    preserve_wrist_pose: bool = False
    chronological_registration: bool = False
    button_edges: bool = True
    omit_hands: bool = True
    nobody_freq: float = 0.
    fps: int = 30
    wholeseq: bool = True

    repeat_datasplit: int = 100
    
    noise_scale: float = 0.0
    restpos_scale_max: float = 1.0
    restpos_scale_min: float = 1.0
    use_betas_for_restpos: bool = False

    single_sequence_file: Optional[str] = None

def create_loader(mcfg: Config) -> 'Loader':
    garment_dict_dir = Path(DEFAULTS.aux_data) / mcfg.garment_dicts_dir
    assert mcfg.body_model in ('smplx', 'raw'), f'Unknown body model: {mcfg.body_model}'
    if mcfg.body_model == 'raw':
        assert not mcfg.use_betas_for_restpos, 'Raw garments require explicit rest_pos geometry'
        assert mcfg.chronological_registration, 'Raw body and cloth must use chronological registration'
        assert mcfg.obstacle_dict_file is None, 'Raw body topology cannot use SMPL obstacle labels'
        return Loader(mcfg, {}, garment_dict_dir, obstacle_dict={})

    body_model_root = Path(DEFAULTS.aux_data) / mcfg.body_model_root

    if mcfg.sequence_loader == 'hood_pkl':
        mcfg.model_type = 'smpl'
    elif 'smplx' in  mcfg.sequence_loader:
        mcfg.model_type = 'smplx'
    elif 'smpl' in mcfg.sequence_loader:
        mcfg.model_type = 'smpl'

    body_models_dict = build_smpl_bygender(
        body_model_root, mcfg.model_type, flat_hand_mean=mcfg.flat_hand_mean)
    obstacle_dict = make_obstacle_dict(mcfg)

    loader = Loader(mcfg, 
                    body_models_dict, garment_dict_dir, obstacle_dict=obstacle_dict)
    return loader



def create(mcfg: Config, **kwargs):
    loader = create_loader(mcfg)

    if 'valid' in kwargs and kwargs['valid']:
        split_file = mcfg.valid_split_path
    else:
        split_file = mcfg.train_split_path

    split_path = os.path.join(DEFAULTS.aux_data, split_file)
    datasplit = pd.read_csv(split_path, dtype='str')

    if mcfg.registration_root is None and 'registration_root' not in datasplit.columns:
        raise ValueError('No registration_root provided in the configuration file and no registration_root column in the datasplit')
    
    if mcfg.body_sequence_root is None and 'body_sequence_root' not in datasplit.columns:
        raise ValueError('No body_sequence_root provided in the configuration file and no body_sequence_root column in the datasplit')

    if 'valid' not in kwargs or not kwargs['valid']:
        datasplit = pd.concat([datasplit] * mcfg.repeat_datasplit, ignore_index=True)

    dataset = Dataset(loader, datasplit)
    return dataset

    


class GarmentBuilder(GarmentBuilderBase):
    def __init__(self, mcfg: Config, body_models_dict, garment_dicts_dir: str):
        super().__init__(mcfg, body_models_dict, garment_dicts_dir)

    def load_garment_dict(self, garment_name: str) -> None:
        if self.mcfg.body_model != 'raw':
            super().load_garment_dict(garment_name)
            return
        if garment_name not in self.garments_dict:
            garment_path = Path(self.garment_dicts_dir) / (garment_name + '.pkl')
            garment_dict = pickle_load(garment_path)
            assert 'rest_pos' in garment_dict, 'Raw garment requires explicit rest_pos geometry'
            self.garments_dict[garment_name] = garment_dict
        


    
    def scale_restpos(self, sample, garment_name):
        garment_dict = self.garments_dict[garment_name]
        if 'rest_pos_scale' not in garment_dict:
            return sample
        
        rest_pos_scale = torch.FloatTensor(garment_dict['rest_pos_scale']).unsqueeze(1)
        rest_pos = sample['cloth'].rest_pos

        rest_pos = rest_pos * rest_pos_scale
        sample['cloth'].rest_pos = rest_pos

        return sample

    
    def add_uvs(self, sample, garment_name: str):
        garment_dict = self.garments_dict[garment_name]
        if 'uv_coords' not in garment_dict:
            return sample
        
        uv_coords = garment_dict['uv_coords']
        sample['cloth'].uv_coords = torch.FloatTensor(uv_coords)


        uv_faces = garment_dict['uv_faces']
        uv_faces = torch.LongTensor(uv_faces)
        sample['cloth'].uv_faces_batch = torch.LongTensor(uv_faces).T


        return sample

    
    def load_mesh_sequence(self, sequence_path):
        mesh_sequence = pickle_load(sequence_path)
        verts = mesh_sequence['vertices']
        faces = mesh_sequence['faces']
        return verts, faces      
    
    def add_verts(self, sample: HeteroData, verts: np.ndarray) -> HeteroData:
        all_verts = torch.tensor(verts).permute(1, 0, 2)

        if self.mcfg.chronological_registration:
            sample['cloth'].prev_pos = all_verts[:, 0]
            sample['cloth'].pos = all_verts[:, 1]
        else:
            sample['cloth'].pos = all_verts[:, 0]
            sample['cloth'].prev_pos = all_verts[:, 1]
        sample['cloth'].target_pos = all_verts[:, 2]
        sample['cloth'].lookup = all_verts[:, 2:]

        return sample

    def build(self, sample, sequence_path, garment_name, sequence_dict):
        self.load_garment_dict(garment_name)
        sample_temp = HeteroData()

        verts, faces = self.load_mesh_sequence(sequence_path)
        sample_temp = self.add_verts(sample_temp, verts)


        sample_temp = self.add_vertex_type(sample_temp, garment_name)
        sample_temp = self.add_restpos(sample_temp, sequence_dict, garment_name)
        sample_temp = self.add_faces_and_edges(sample_temp, garment_name)
        sample_temp = self.add_coarse(sample_temp, garment_name)
        sample_temp = self.add_garment_id(sample_temp, garment_name)
        sample_temp = self.scale_restpos(sample_temp, garment_name)

        sample = self.add_garment_to_sample(sample, sample_temp)

        return sample


class RawBodySequenceLoader:
    """Read fixed-topology raw body meshes without posing, resampling, or skinning."""

    def __init__(self, mcfg: Config) -> None:
        self.mcfg = mcfg

    def load_sequence(self, path: str | Path) -> dict:
        assert Path(path).is_file(), f'Missing raw body sequence: {path}'
        with np.load(path, allow_pickle=False) as archive:
            sequence = {name: archive[name] for name in ('verts', 'faces', 'mocap_frame_rate')}
        verts, faces = sequence['verts'], sequence['faces']
        assert verts.ndim == 3 and verts.shape[0] >= 3 and verts.shape[2] == 3, \
            'Raw body vertices must have shape [T >= 3, V, 3]'
        assert np.isfinite(verts).all(), 'Raw body vertices contain nonfinite values'
        assert faces.ndim == 2 and faces.shape[1] == 3 and faces.size > 0, \
            'Raw body faces must have shape [F, 3]'
        assert np.issubdtype(faces.dtype, np.integer), 'Raw body faces must be integer indices'
        assert faces.min() >= 0 and faces.max() < verts.shape[1], 'Raw body face index out of range'
        assert np.isclose(float(sequence['mocap_frame_rate']), self.mcfg.fps), \
            'Raw body FPS must match fitting FPS; implicit resampling is unsupported'
        return sequence


class Loader:
    def __init__(self, mcfg: Config, body_models_dict: dict, garment_dicts_dir: str | Path,
                 obstacle_dict: dict, betas_table: Optional[pd.DataFrame] = None) -> None:
        if mcfg.body_model == 'raw':
            from datasets.from_any_pose import BareMeshBodyBuilder

            self.sequence_loader = RawBodySequenceLoader(mcfg)
            self.body_builder = BareMeshBodyBuilder(mcfg, {})
        else:
            sequence_loader_module = importlib.import_module(f'datasets.sequence_loaders.{mcfg.sequence_loader}')
            SequenceLoader = sequence_loader_module.SequenceLoader
            body_sequence_root = mcfg.body_sequence_root or ''
            self.sequence_loader = SequenceLoader(mcfg, body_sequence_root, betas_table=betas_table)
            self.body_builder = BodyBuilder(mcfg, body_models_dict, obstacle_dict)

        self.garment_builder = GarmentBuilder(mcfg, body_models_dict, garment_dicts_dir)

        self.mcfg = mcfg

    def build_body_sample(self, sample: HeteroData, sequence: dict, gender: str) -> HeteroData:
        if self.mcfg.body_model == 'raw':
            return self.body_builder.build(sample, sequence)
        return self.body_builder.build(sample, sequence, 0, gender)

    def load_sample(self, subject: str, sequence: str, gender: str,
                    registration_root: Optional[str] = None,
                    body_sequence_root: Optional[str] = None) -> HeteroData:
        body_sequence_root = body_sequence_root or self.mcfg.body_sequence_root
        body_sequence_path = Path(body_sequence_root) / (sequence + '.npz')
        body_sequence = self.sequence_loader.load_sequence(body_sequence_path)

        sample = HeteroData()
        sample = self.build_body_sample(sample, body_sequence, gender)

        registration_root = registration_root or self.mcfg.registration_root
        garment_seq_path = Path(registration_root) / (sequence + '.pkl')
        sample = self.garment_builder.build(sample, garment_seq_path, subject, body_sequence)
        if self.mcfg.body_model == 'raw':
            assert sample['cloth'].lookup.shape[1] == sample['obstacle'].lookup.shape[1], \
                'Raw body and registration frame counts differ'

        return sample

class Dataset:
    def __init__(self, loader: Loader, datasplit: pd.DataFrame):
        """
        Dataset class for building training and validation samples
        :param loader: Loader object
        :param datasplit: pandas DataFrame with the following columns:
            id: sequence name relative to loader.data_path
            length: number of frames in the sequence
            garment: name of the garment
        """

        self.loader = loader
        self.datasplit = datasplit
        self._len = self.datasplit.shape[0] 

    def _find_idx(self, index: int) -> Tuple[str, int, str]:
        """
        Takes a global index and returns the sequence name, frame index and garment name for it
        """
        fi = 0
        while self.all_lens[fi] <= index:
            index -= self.all_lens[fi]
            fi += 1
        if 'gender' in self.datasplit:
            gender = self.datasplit.gender[fi]
        else:
            gender = 'female'

        return self.datasplit.id[fi], index, self.datasplit.garment[fi], gender


    def __getitem__(self, item: int) -> HeteroData:
        """
        Load a sample given a global index
        """

        fname = self.datasplit.id[item]
        garment_name = self.datasplit.garment[item]
        if 'gender' in self.datasplit:
            gender = self.datasplit.gender[item]
        else:
            gender = 'female'
        idx = 0

        if 'registration_root' in self.datasplit.columns:
            registration_root = self.datasplit.registration_root[item]
        else:
            registration_root = None

        if 'body_sequence_root' in self.datasplit.columns:
            body_sequence_root = self.datasplit.body_sequence_root[item]
        else:
            body_sequence_root = None

        sample = self.loader.load_sample(garment_name, fname, gender, 
                                         registration_root=registration_root, 
                                         body_sequence_root=body_sequence_root)
        sample['sequence_name'] = fname
        sample['garment_name'] = garment_name

        return sample

    def __len__(self) -> int:
        return self._len
