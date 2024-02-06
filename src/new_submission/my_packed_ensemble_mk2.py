import os
import time
import random
import math 
import datetime as dt
from typing import Union

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Identity
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from lips import get_root_path
from lips.benchmark.airfransBenchmark import AirfRANSBenchmark
from lips.dataset.airfransDataSet import download_data
from lips.dataset.scaler.standard_scaler_iterative import StandardScalerIterative

from einops import rearrange

# fmt: on
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
    """
    A simple MLP with packed layers

    Parameters
    ----------
    name: str (default: "PackedMLP")
        The name of the model
    input_size: int (default: None)
        The size of the input
    output_size: int (default: None)
        The size of the output
    hidden_sizes: tuple (default: (100, 100,))
        The sizes of the hidden layers
    activation: torch.nn.functional (default: F.relu)
        The activation function
    dropout: bool (default: False)
        Whether to use dropout
    batch_normalization: bool (default: False)
        Whether to use batch normalization
    M: int (default: 4)
        The number of estimators
    alpha: int (default: 2)
        The alpha parameter
    gamma: int (default: 1)
        The gamma parameter
    device: str (default: "cpu")
        The device to use
    """

    def __init__(self, hparams: dict):
        super().__init__()

        dropout = hparams.get('dropout')
        # dropout
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

        self.device = hparams['device']

        self.activation = F.relu
        self.input_size = hparams['input_size']
        self.output_size = hparams['output_size']
        self.hidden_sizes = hparams['hidden_sizes']

        self.num_estimators = hparams['M']
        self.alpha = hparams['alpha']
        self.gamma = hparams['gamma']

        self.input_layer = PackedLinear(self.input_size, self.hidden_sizes[0], alpha=self.alpha, num_estimators=self.num_estimators,
                                        gamma=self.gamma, first=True,
                                        device=self.device)
        self.hidden_layers = nn.ModuleList(
            [PackedLinear(in_f, out_f, alpha=self.alpha, num_estimators=self.num_estimators, gamma=self.gamma, device=self.device) \
             for in_f, out_f in zip(self.hidden_sizes[:-1], self.hidden_sizes[1:])])
        self.output_layer = PackedLinear(self.hidden_sizes[-1], self.output_size, alpha=self.alpha, num_estimators=self.num_estimators,
                                         gamma=self.gamma, last=True,
                                         device=self.device)

    def forward(self, data):
        """
        The forward pass of the model

        Parameters
        ----------
        data: torch.Tensor
            The input data

        Returns
        -------
        out: torch.Tensor
            The forward-pass output of the model
        """

        out = self.input_layer(data)
        for _, fc_ in enumerate(self.hidden_layers):
            out = fc_(out)
            out = self.activation(out)
            if self.dropout is not None:
                out = self.dropout(out)
        out = self.output_layer(out)

        return out


class AugmentedSimulator():
    def __init__(self,benchmark,**kwargs):
        self.name = "AirfRANSSubmission"
        chunk_sizes=benchmark.train_dataset.get_simulations_sizes()
        scalerParams={"chunk_sizes":chunk_sizes}
        self.scaler = StandardScalerIterative(**scalerParams)

        self.model = None
        self.hparams = kwargs
        use_cuda = torch.cuda.is_available()
        self.device = 'cuda:0' if use_cuda else 'cpu'
        if use_cuda:
            print('Using GPU')
        else:
            print('Using CPU')

        self.model = PackedMLP(self.hparams)

    def process_dataset(self, dataset, training: bool) -> DataLoader:
        coord_x=dataset.data['x-position']
        coord_y=dataset.data['y-position']
        surf_bool=dataset.extra_data['surface']
        position = np.stack([coord_x,coord_y],axis=1)

        nodes_features,node_labels=dataset.extract_data()
        if training:
            print("Normalize train data")
            nodes_features, node_labels = self.scaler.fit_transform(nodes_features, node_labels)
            print("Transform done")
        else:
            print("Normalize not train data")
            nodes_features, node_labels = self.scaler.transform(nodes_features, node_labels)
            print("Transform done")

        torchDataset=[]
        nb_nodes_in_simulations = dataset.get_simulations_sizes()
        start_index = 0
        # check alive
        t = dt.datetime.now()

        for nb_nodes_in_simulation in nb_nodes_in_simulations:
            #still alive?
            if dt.datetime.now() - t > dt.timedelta(seconds=60):
                print("Still alive - index : ", end_index)
                t = dt.datetime.now()
            end_index = start_index+nb_nodes_in_simulation
            simulation_positions = torch.tensor(position[start_index:end_index,:], dtype = torch.float) 
            simulation_features = torch.tensor(nodes_features[start_index:end_index,:], dtype = torch.float) 
            simulation_labels = torch.tensor(node_labels[start_index:end_index,:], dtype = torch.float) 
            simulation_surface = torch.tensor(surf_bool[start_index:end_index])

            sampleData=Data(pos=simulation_positions,
                            x=simulation_features, 
                            y=simulation_labels,
                            surf = simulation_surface.bool()) 
            torchDataset.append(sampleData)
            start_index += nb_nodes_in_simulation
        
        return DataLoader(dataset=torchDataset,batch_size=1)

    def train(self,train_dataset, save_path=None):
        train_dataset = self.process_dataset(dataset=train_dataset,training=True)
        print("Start training")
        model = global_train(self.device, train_dataset, self.model, self.hparams,criterion = 'MSE_weighted')
        print("Training done")

    def predict(self,dataset,**kwargs):
        print(dataset)
        test_dataset = self.process_dataset(dataset=dataset,training=False)
        self.model.eval()
        avg_loss_per_var = np.zeros(4)
        avg_loss = 0
        avg_loss_surf_var = np.zeros(4)
        avg_loss_vol_var = np.zeros(4)
        avg_loss_surf = 0
        avg_loss_vol = 0
        iterNum = 0

        predictions=[]
        with torch.no_grad():
            for data in test_dataset:        
                data_clone = data.clone()
                data_clone = data_clone.to(self.device)
                packed_out = self.model(data_clone)

                # averaging the predictions of the different ensemble models
                out = rearrange(packed_out, '(n b) m -> b n m', n=self.model.num_estimators)
                out = out.mean(dim=1)

                targets = data_clone.y
                loss_criterion = nn.MSELoss(reduction = 'none')

                loss_per_var = loss_criterion(out, targets).mean(dim = 0)
                loss = loss_per_var.mean()
                loss_surf_var = loss_criterion(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim = 0)
                loss_vol_var = loss_criterion(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim = 0)
                loss_surf = loss_surf_var.mean()
                loss_vol = loss_vol_var.mean()  

                avg_loss_per_var += loss_per_var.cpu().numpy()
                avg_loss += loss.cpu().numpy()
                avg_loss_surf_var += loss_surf_var.cpu().numpy()
                avg_loss_vol_var += loss_vol_var.cpu().numpy()
                avg_loss_surf += loss_surf.cpu().numpy()
                avg_loss_vol += loss_vol.cpu().numpy()  
                iterNum += 1

                out = out.cpu().data.numpy()
                prediction = self._post_process(out)
                predictions.append(prediction)
        print("Results for test")
        print(avg_loss/iterNum, avg_loss_per_var/iterNum, avg_loss_surf_var/iterNum, avg_loss_vol_var/iterNum, avg_loss_surf/iterNum, avg_loss_vol/iterNum)
        predictions= np.vstack(predictions)
        predictions = dataset.reconstruct_output(predictions)
        return predictions

    def _post_process(self, data):
        try:
            processed = self.scaler.inverse_transform(data)
        except TypeError:
            processed = self.scaler.inverse_transform(data.cpu())
        return processed


def global_train(device, train_dataset, network, hparams, criterion = 'MSE', reg = 1):
    model = network.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = hparams['lr'],
            total_steps = (len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
        )
    start = time.time()

    train_loss_surf_list = []
    train_loss_vol_list = []
    loss_surf_var_list = []
    loss_vol_var_list = []

    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    epoch_nb = 0

    for epoch in pbar_train:
        epoch_nb += 1
        print('Epoch: ', epoch_nb)        
        train_dataset_sampled = []
        for data in train_dataset:
            data_sampled = data.clone()
            idx = random.sample(range(data_sampled.x.size(0)), hparams['subsampling'])
            idx = torch.tensor(idx)

            data_sampled.pos = data_sampled.pos[idx]
            data_sampled.x = data_sampled.x[idx]
            data_sampled.y = data_sampled.y[idx]
            data_sampled.surf = data_sampled.surf[idx]
            train_dataset_sampled.append(data_sampled)
        train_loader = DataLoader(train_dataset_sampled, batch_size = hparams['batch_size'], shuffle = True)
        del(train_dataset_sampled)

        train_loss, _, loss_surf_var, loss_vol_var, loss_surf, loss_vol = train_model(device, model, train_loader, optimizer, lr_scheduler, criterion, reg = reg)        
        if criterion == 'MSE_weighted':
            train_loss = reg*loss_surf + loss_vol
        del(train_loader)

        train_loss_surf_list.append(loss_surf)
        train_loss_vol_list.append(loss_vol)
        loss_surf_var_list.append(loss_surf_var)
        loss_vol_var_list.append(loss_vol_var)

    loss_surf_var_list = np.array(loss_surf_var_list)
    loss_vol_var_list = np.array(loss_vol_var_list)

    return model

def train_model(device, model, train_loader, optimizer, scheduler, criterion = 'MSE', reg = 1):
    model.train()
    avg_loss_per_var = torch.zeros(4, device = device)
    avg_loss = 0
    avg_loss_surf_var = torch.zeros(4, device = device)
    avg_loss_vol_var = torch.zeros(4, device = device)
    avg_loss_surf = 0
    avg_loss_vol = 0
    iterNum = 0
    
    for data in train_loader:
        data_clone = data.clone()
        data_clone = data_clone.to(device)   
        optimizer.zero_grad()  
        out = model(data_clone)
        targets = data_clone.y

        if criterion == 'MSE' or criterion == 'MSE_weighted':
            loss_criterion = nn.MSELoss(reduction = 'none')
        elif criterion == 'MAE':
            loss_criterion = nn.L1Loss(reduction = 'none')
        loss_per_var = loss_criterion(out, targets.repeat(model.num_estimators, 1)).mean(dim = 0)
        total_loss = loss_per_var.mean()
        loss_surf_var = loss_criterion(out[data_clone.surf, :], targets[data_clone.surf, :].repeat(model.num_estimators, 1)).mean(dim = 0)
        loss_vol_var = loss_criterion(out[~data_clone.surf, :], targets[~data_clone.surf, :].repeat(model.num_estimators, 1)).mean(dim = 0)
        loss_surf = loss_surf_var.mean()
        loss_vol = loss_vol_var.mean()

        if criterion == 'MSE_weighted':            
            (loss_vol + reg*loss_surf).backward()           
        else:
            total_loss.backward()
        
        optimizer.step()
        scheduler.step()
        avg_loss_per_var += loss_per_var
        avg_loss += total_loss
        avg_loss_surf_var += loss_surf_var
        avg_loss_vol_var += loss_vol_var
        avg_loss_surf += loss_surf
        avg_loss_vol += loss_vol 
        iterNum += 1

    return avg_loss.cpu().data.numpy()/iterNum, avg_loss_per_var.cpu().data.numpy()/iterNum, avg_loss_surf_var.cpu().data.numpy()/iterNum, avg_loss_vol_var.cpu().data.numpy()/iterNum, \
            avg_loss_surf.cpu().data.numpy()/iterNum, avg_loss_vol.cpu().data.numpy()/iterNum
