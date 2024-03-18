import os
import pathlib
import json
from typing import Union
from tqdm import tqdm

import numpy as np
import numpy.typing as npt

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from lips.dataset import DataSet
from lips.logger import CustomLogger
from lips.config import ConfigManager
from lips.utils import NpEncoder
from lips.augmented_simulators.torch_models.utils import LOSSES
from lips.dataset.scaler import Scaler


from einops import rearrange
from torch_uncertainty.layers import PackedLinear

class PackedMLP(nn.Module):
    """_summary_

    Parameters
    ----------
    sim_config_path : Union[``pathlib.Path``, ``str``]
        The path to the configuration file for simulator.
        It should contain all the required hyperparameters for this model.
    sim_config_name : Union[str, None], optional
        the simulator config section name, by default None
    name : Union[str, None], optional
        the simulator name used for save and load, by default None
    scaler : Union[Scaler, None], optional
        A scaler used to normalize the data, by default None
    bench_config_path : Union[str, pathlib.Path, None], optional
        a path to the benchmark configuration file, by default None
    bench_config_name : Union[str, None], optional
        the section name of the benchmark configuration, by default None
    log_path : Union[None, str], optional
        a path where the logs should be saved, by default None

    Raises
    ------
    RuntimeError
        You should provide a path to the configuration file for this augmented simulator
    """
    def __init__(self,
                 sim_config_path: Union[pathlib.Path, str],
                 bench_config_path: Union[str, pathlib.Path],
                 sim_config_name: Union[str, None]=None,
                 bench_config_name: Union[str, None]=None,
                 name: Union[str, None]="PackedMLP",
                 scaler: Union[Scaler, None]=None,
                 log_path: Union[None, pathlib.Path, str]=None,
                 **kwargs):
        super().__init__()
        if not os.path.exists(sim_config_path):
            raise RuntimeError("Configuration path for the simulator not found!")
        if not str(sim_config_path).endswith(".ini"):
            raise RuntimeError("The configuration file should have `.ini` extension!")
        sim_config_name = sim_config_name if sim_config_name is not None else "DEFAULT"
        self.sim_config = ConfigManager(section_name=sim_config_name, path=sim_config_path)
        self.bench_config = ConfigManager(section_name=bench_config_name, path=bench_config_path)
        self.name = name if name is not None else self.sim_config.get_option("name")
        # scaler
        self.scaler = scaler
        # Logger
        self.log_path = log_path
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        # model parameters
        self.params = self.sim_config.get_options_dict()
        self.params.update(kwargs)

        self.activation = {
            "relu": F.relu,
            "sigmoid": F.sigmoid,
            "tanh": F.tanh
        }

        self.input_size = None if kwargs.get("input_size") is None else kwargs["input_size"]
        self.output_size = None if kwargs.get("output_size") is None else kwargs["output_size"]

        self.hidden_sizes = self.params["hidden_sizes"]

        self.num_estimators = self.params["num_estimators"]
        self.alpha = self.params["alpha"]
        self.gamma = self.params["gamma"]

        self.device = self.params["device"]

        self.dropout = self.params["dropout"]

        self.input_layer = None
        self.input_dropout = None
        self.fc_layers = None
        self.dropout_layers = None
        self.output_layer = None

        # batch information
        self._data = None
        self._target = None

    def build_model(self):
        """Build the model flow
        """

        self.input_layer = PackedLinear(self.input_size, self.hidden_sizes[0], alpha=self.alpha, num_estimators=self.num_estimators,
                                        gamma=self.gamma, first=True,
                                        device=self.device)
        
        self.hidden_layers = nn.ModuleList(
            [PackedLinear(in_f, out_f, alpha=self.alpha, num_estimators=self.num_estimators, gamma=self.gamma, device=self.device) \
             for in_f, out_f in zip(self.hidden_sizes[:-1], self.hidden_sizes[1:])])
        
        self.output_layer = PackedLinear(self.hidden_sizes[-1], self.output_size, alpha=self.alpha, num_estimators=self.num_estimators,
                                         gamma=self.gamma, last=True,
                                         device=self.device)

        if self.dropout:
            self.dropout = nn.Dropout(p=0.2)
        else:
            self.dropout = nn.Identity()

    def forward(self, data):
        out = self.input_layer(data)
        for _, fc_ in enumerate(self.hidden_layers):
            out = fc_(out)
            out = self.activation[self.params["activation"]](out)
            out = self.dropout(out)
        out = self.output_layer(out)
        return out


    def process_dataset(self, dataset: DataSet, training: bool, **kwargs) -> DataLoader:
        """process the datasets for training and evaluation

        This function transforms all the dataset into something that can be used by the neural network (for example)

        Parameters
        ----------
        dataset : DataSet
            an object of the DataSet class including the required data
        scaler : Scaler, optional
            A scaler instance to be used for normalization, by default True
        training : bool, optional
            A boolean indicating whether we are in training or evaluation phases, by default False
            If `True`, the scaler will be fit to the data to estimate the parameters
            If `False`, the estimated parameters of the scaler during training will be used to normalize the 
            validation/test/test_ood data

        kwargs : dict
            The supplementary arguments to be used for acceleration of DataLoader which are:
                pin_memory : `bool`, optional
                    refere to pytorch documentation for more information
                num_workers : Union[None, int], optional
                    the number of CPU workers to be used to transfer the batches to device
                dtype : torch.types
                    the data type that will be used to transform the processed dataset
        Returns
        -------
        DataLoader
            A pytorch data loader from which the batches of data could be loaded for training
        """
        pin_memory = kwargs.get("pin_memory", True)
        num_workers = kwargs.get("num_workers", None)
        dtype = kwargs.get("dtype", torch.float32)

        if training:
            self._infer_size(dataset)
            batch_size = self.params["train_batch_size"]
            extract_x, extract_y = dataset.extract_data()
            if self.scaler is not None:
                extract_x, extract_y = self.scaler.fit_transform(extract_x, extract_y)
        else:
            batch_size = self.params["eval_batch_size"]
            extract_x, extract_y = dataset.extract_data()
            if self.scaler is not None:
                extract_x, extract_y = self.scaler.transform(extract_x, extract_y)

        torch_dataset = TensorDataset(torch.tensor(extract_x, dtype=dtype), torch.tensor(extract_y, dtype=dtype))
        if num_workers is None:
            data_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=self.params["shuffle"], pin_memory=pin_memory)
        else:
            data_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=self.params["shuffle"], pin_memory=pin_memory, num_workers=num_workers)
        #data_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=self.params["shuffle"])
        return data_loader

    def _post_process(self, data):
        if self.scaler is not None:
            try:
                processed = self.scaler.inverse_transform(data)
            except TypeError:
                processed = self.scaler.inverse_transform(data.cpu())
        else:
            processed = data
        return processed
    
    def _reconstruct_output(self, dataset: DataSet, data: npt.NDArray[np.float64]) -> dict:
        """Reconstruct the outputs to obtain the desired shape for evaluation

        In the simplest form, this function is implemented in DataSet class. It supposes that the predictions 
        obtained by the augmented simulator are exactly the same as the one indicated in the configuration file

        However, if some transformations required by each specific model, the extra operations to obtained the
        desired output shape should be done in this function.

        Parameters
        ----------
        dataset : DataSet
            An object of the `DataSet` class 
        data : npt.NDArray[np.float64]
            the data which should be reconstructed to the desired form
        """
        data_rec = dataset.reconstruct_output(data)
        return data_rec

    def _infer_size(self, dataset: DataSet):
        """Infer the size of the model

        Parameters
        ----------
        dataset : DataSet
            An object of the dataset class providing some functionalities to get sizes of inputs/outputs

        """
        *dim_inputs, self.output_size = dataset.get_sizes()
        self.input_size = np.sum(dim_inputs)

    def get_metadata(self):
        """getting the augmented simulator meta data

        Returns
        -------
        dict
            a dictionary containing the meta data for the augmented simulator
        """
        res_json = {}
        res_json["input_size"] = self.input_size
        res_json["output_size"] = self.output_size
        return res_json

    def _save_metadata(self, path: str):
        """Save the augmented simulator specific meta data

        These information are required to restore a saved model

        Parameters
        ----------
        path : str
            A path where the meta data should be saved
        """
        #super()._save_metadata(path)
        #if self.scaler is not None:
        #    self.scaler.save(path)
        res_json = {}
        res_json["input_size"] = self.input_size
        res_json["output_size"] = self.output_size
        with open((path / "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(obj=res_json, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    def _load_metadata(self, path: str):
        """Load the metadata for the augmentd simulator

        Parameters
        ----------
        path : str
            a path where the meta data are saved
        """
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        #super()._load_metadata(path)
        #if self.scaler is not None:
        #    self.scaler.load(path)
        with open((path / "metadata.json"), "r", encoding="utf-8") as f:
            res_json = json.load(fp=f)
        self.input_size = res_json["input_size"]
        self.output_size = res_json["output_size"]

    def _do_forward(self, batch, **kwargs):
        """Do the forward step through a batch of data

        This step could be very specific to each augmented simulator as each architecture
        takes various inputs during the learning procedure. 

        Parameters
        ----------
        batch : _type_
            A batch of data including various information required by an architecture
        device : _type_
            the device on which the data should be processed

        Returns
        -------
        ``tuple``
            returns the predictions made by the augmented simulator and also the real targets
            on which the loss function should be computed
        """
        non_blocking = kwargs.get("non_blocking", True)
        device = self.params.get("device", "cpu")
        self._data, self._target = batch
        self._data = self._data.to(device, non_blocking=non_blocking)
        self._target = self._target.to(device, non_blocking=non_blocking)

        predictions = self.forward(self._data)

        if self.training:
            return self._data, predictions, self._target.repeat(self.num_estimators, 1)
        else:
            packed_split = rearrange(predictions, '(n b) m -> b n m', n=self.num_estimators)
            packed_prediction = packed_split.mean(dim=1)
            return self._data, packed_prediction, self._target

    def get_loss_func(self, loss_name: str, **kwargs) -> torch.Tensor:
        """
        Helper to get loss. It is specific to each architecture
        """
        # if len(args) > 0:
        #     # for Masked RNN loss. args[0] is the list of sequence lengths
        #     loss_func = LOSSES[self.params["loss"]["name"]](args[0], self.params["device"])
        # else:
        loss_func = LOSSES[loss_name](**kwargs)
        
        return loss_func