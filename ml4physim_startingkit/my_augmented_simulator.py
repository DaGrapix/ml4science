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

#Installing einops if it is not already installed
import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('einops')
from einops import rearrange

# code from Torch uncertainty package
def check_packed_parameters_consistency(
    alpha: float, num_estimators: int, gamma: int
) -> None:
    if alpha is None:
        raise ValueError("You must specify the value of the arg. `alpha`")

    if alpha <= 0:
        raise ValueError(f"Attribute `alpha` should be > 0, not {alpha}")

    if num_estimators is None:
        raise ValueError(
            "You must specify the value of the arg. `num_estimators`"
        )
    if not isinstance(num_estimators, int):
        raise ValueError(
            "Attribute `num_estimators` should be an int, not "
            f"{type(num_estimators)}"
        )
    if num_estimators <= 0:
        raise ValueError(
            "Attribute `num_estimators` should be >= 1, not "
            f"{num_estimators}"
        )

    if not isinstance(gamma, int):
        raise ValueError(
            f"Attribute `gamma` should be an int, not " f"{type(gamma)}"
        )
    if gamma <= 0:
        raise ValueError(f"Attribute `gamma` should be >= 1, not {gamma}")

# code from Torch uncertainty package
class PackedLinear(nn.Module):
    r"""Packed-Ensembles-style Linear layer.

    This layer computes fully-connected operation for a given number of
    estimators (:attr:`num_estimators`) using a `1x1` convolution.

    Args:
        in_features (int): Number of input features of the linear layer.
        out_features (int): Number of channels produced by the linear layer.
        alpha (float): The width multiplier of the linear layer.
        num_estimators (int): The number of estimators grouped in the layer.
        gamma (int, optional): Defaults to ``1``.
        bias (bool, optional): It ``True``, adds a learnable bias to the
            output. Defaults to ``True``.
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Defaults to ``1``.
        rearrange (bool, optional): Rearrange the input and outputs for
            compatibility with previous and later layers. Defaults to ``True``.

    Explanation Note:
        Increasing :attr:`alpha` will increase the number of channels of the
        ensemble, increasing its representation capacity. Increasing
        :attr:`gamma` will increase the number of groups in the network and
        therefore reduce the number of parameters.

    Note:
        Each ensemble member will only see
        :math:`\frac{\text{in_features}}{\text{num_estimators}}` features,
        so when using :attr:`groups` you should make sure that
        :attr:`in_features` and :attr:`out_features` are both divisible by
        :attr:`n_estimators` :math:`\times`:attr:`groups`. However, the
        number of input and output features will be changed to comply with
        this constraint.

    Note:
        The input should be of size (`batch_size`, :attr:`in_features`, 1,
        1). The (often) necessary rearrange operation is executed by
        default.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float,
        num_estimators: int,
        gamma: int = 1,
        bias: bool = True,
        rearrange: bool = True,
        first: bool = False,
        last: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        check_packed_parameters_consistency(alpha, num_estimators, gamma)

        self.first = first
        self.num_estimators = num_estimators
        self.rearrange = rearrange

        # Define the number of features of the underlying convolution
        extended_in_features = int(in_features * (1 if first else alpha))
        extended_out_features = int(
            out_features * (num_estimators if last else alpha)
        )

        # Define the number of groups of the underlying convolution
        actual_groups = num_estimators * gamma if not first else 1

        # fix if not divisible by groups
        if extended_in_features % actual_groups:
            extended_in_features += num_estimators - extended_in_features % (
                actual_groups
            )
        if extended_out_features % actual_groups:
            extended_out_features += num_estimators - extended_out_features % (
                actual_groups
            )

        self.conv1x1 = nn.Conv1d(
            in_channels=extended_in_features,
            out_channels=extended_out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=actual_groups,
            bias=bias,
            padding_mode="zeros",
            **factory_kwargs,
        )

    def _rearrange_forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        if not self.first:
            x = rearrange(x, "(m e) c h -> e (m c) h", m=self.num_estimators)

        x = self.conv1x1(x)
        x = rearrange(x, "e (m c) h -> (m e) c h", m=self.num_estimators)
        return x.squeeze(-1)

    def forward(self, input: Tensor) -> Tensor:
        if self.rearrange:
            return self._rearrange_forward(input)
        else:
            return self.conv1x1(input)

    @property
    def weight(self) -> Tensor:
        r"""The weight of the underlying convolutional layer."""
        return self.conv1x1.weight

    @property
    def bias(self) -> Union[Tensor, None]:
        r"""The bias of the underlying convolutional layer."""
        return self.conv1x1.bias



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