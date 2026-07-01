import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from aurora import Batch

from train_hydr import AuroraDataset
from train_hydr import LogTransformer
from train_hydr import HyperbolicArcSineTransformer
from train_hydr import locations
from train_hydr import scales


class AuroraDatasetLiteDecoder(AuroraDataset):
    def __init__(self, surf_paths, stat_ds, atmo_paths, hydr_paths, et_paths, p_paths, seq_len: int = 2):
        """make batch to input aurora-hydrological model

        Args:
            surf_paths (`list` of `str`): surface variables
            stat_ds (:obj:`xarray.Dataset`): static variables
            atmo_paths (`list` of `str`): atmospheric variables
            hydr_paths (`list` of `str`): hydrological variables
            et_paths (`list` of `str`): evaporation variables
            p_paths (`list` of `str`): precipitation variables
            seq_len (`int`, optional): time step for input aurora
        """
        super().__init__(surf_paths, stat_ds, atmo_paths, seq_len=seq_len)

        self.hydr_paths = hydr_paths
        self.et_paths = et_paths
        self.p_paths = p_paths
        self.set_transformer()
        return

    hydr_vars = {
#       'stl1':'stl1', 'stl2':'stl2', 'stl3':'stl3', 'stl4':'stl4',
#       'swvl1':'swvl1', 'swvl2':'swvl2', 'swvl3':'swvl3', 'swvl4':'swvl4',
        'swvl1': 'swvl1'
    }
    et_vars = {
#       'e':'e', 'pev':'pev', 'ro':'ro', 'sro':'sro', 'ssro':'ssro',
        'evaporation': 'e'
    }
    p_vars = {
        'precipitation': 'precipitation'
    }
    new_vars = tuple(hydr_vars.keys()) + tuple(et_vars.keys()) + tuple(p_vars.keys())
    transformer = {}

    @staticmethod
    def set_normalization():
        locations['swvl1'], scales['swvl1'] = 0.26556253, 0.13009237
        locations['evaporation'], scales['evaporation'] = -3.619041, 2.8426096 # arcsinh transform
        locations['precipitation'], scales['precipitation'] = 3.48366, 4.88110 # log transform

    @classmethod
    def set_transformer(cls):
        cls.transformer['evaporation'] = HyperbolicArcSineTransformer(1e-6)
        cls.transformer['precipitation'] = LogTransformer(1., 1e-5)
        return

    def __getitem__(self, idx: int):
        # idx: start index, end_idx: end index (exclusive)
        end_idx = idx + self.seq_len # used as [idx:end_idx], end_idx not included
        target_idx = end_idx # used as [target_idx]

        # 1. Input data
        surf_data = self._get_surf_data(idx, end_idx)
        atmo_data = self._get_atmo_data(idx, end_idx)
        static_data = self._get_stat_data() # use the static data of the first timestep, same as the inference batch (shared)
        metadata = self._get_meta_data(end_idx -1)

        # 2. Lite Decoder Target Data
        target_dict = {}
        target_dict.update(self._get_hydr_data(target_idx, target_idx +1))
        target_dict.update(self._get_et_data(target_idx, target_idx +1))
        target_dict.update(self._get_p_data(target_idx, target_idx +1))

        # 3. return
        input_batch = Batch(
            surf_vars={k: torch.from_numpy(v) for k, v in surf_data.items()},
            static_vars={k: torch.from_numpy(v) for k, v in static_data.items()},
            atmos_vars={k: torch.from_numpy(v) for k, v in atmo_data.items()},
            metadata=metadata
        )
        target_dict = {k: torch.from_numpy(v) for k, v in target_dict.items()}

        return input_batch, target_dict

    def _get_hydr_data(self, idx, end_idx):
        hydr_data = {}
        length = end_idx - idx
        for k, v in self.hydr_vars.items():
            data = self._get_continuous_values(self.hydr_paths, v, idx, length)
            hydr_data[k] = data[None]
            if k in self.transformer:
                hydr_data[k] = self.transformer[k].transform(hydr_data[k])
        return hydr_data

    def _get_et_data(self, idx, end_idx):
        et_data = {}
        length = end_idx - idx
        for k, v in self.et_vars.items():
            data = self._get_continuous_values(self.et_paths, v, idx, length)
            et_data[k] = data[None]
            if k in self.transformer:
                et_data[k] = self.transformer[k].transform(et_data[k])
        return et_data

    def _get_p_data(self, idx, end_idx):
        p_data = {}
        length = end_idx - idx
        for k, v in self.p_vars.items():
            data = self._get_continuous_values(self.p_paths, v, idx, length)
            p_data[k] = data[None]
            if k in self.transformer:
                p_data[k] = self.transformer[k].transform(p_data[k])
        return p_data


def load_data(surf_path_list, atmo_path_list, hydr_path_list, et_path_list, p_path_list, stat_ds, train_test_ratio=0.8):
    """load dataloader and split train and test data

    Args:
        surf_path_list (`list` of :obj:`xarray.Dataset`): surface variable dataset list
        atmo_path_list (`list` of :obj:`xarray.Dataset`): atmospheric variable dataset list
        hydr_path_list (`list` of :obj:`xarray.Dataset`): hydrological variable dataset list
        et_path_list (`list` of :obj:`xarray.Dataset`): evaporation variable dataset list
        stat_ds (:obj:`xarray.Dataset`): static variable dataset
        train_test_ratio (`float`, optional): train ratio (defulats 0.7)
    """
    dataset = AuroraDatasetLiteDecoder(surf_path_list, stat_ds, atmo_path_list, hydr_path_list, et_path_list, p_path_list)
    AuroraDatasetLiteDecoder.set_normalization()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    train_size = int(len(dataloader) * train_test_ratio)
    train_idx = range(0, train_size)
    test_idx = range(train_size, len(dataloader))

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    # train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])
    # test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])

    # return train_dataloader, test_dataloader
    return train_dataset, test_dataset


def dict_to_device(dict: dict, device: str) -> dict:
    res = {}
    for k, v in dict.items():
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        res[k] = v.to(device)
    return res


def dict_to_numpy(dict: dict) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in dict.items()}
