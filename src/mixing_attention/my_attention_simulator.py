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


# K-means clustering

# using a uniform at random initialisation
def centroid_uniform_initialisation(X:Tensor, k: int=100):
    samples = np.random.choice(a=X.shape[0], replace=False, size=k)
    return X[samples,:]

# using the k-means++ initialisation
def kmeans_plusplus_initialisation(X: torch.Tensor, k: int):
    n_samples = X.shape[0]
    # Step 1: Choose one center uniformly at random from among the data points.
    centroids = X[torch.randint(0, n_samples, (1,))]

    for _ in range(k - 1):
        # Step 2: For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
        distances = torch.cdist(X, centroids, p=2)
        min_distances = torch.min(distances, dim=1)[0]

        # Step 3: Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)^2.
        probabilities = min_distances / torch.sum(min_distances)
        new_centroid_idx = torch.multinomial(probabilities, 1)

        # Add the new centroid to the list of centroids
        centroids = torch.cat([centroids, X[new_centroid_idx]], dim=0)

    return centroids

def k_means(X:Tensor, k: int=6, n_iter: int=1000):
    centroids = kmeans_plusplus_initialisation(X,k)

    for i in range(n_iter):
        # compute the distance of each point to the centroids
        distances = torch.cdist(X, centroids, p=2)

        # define the index associated to each centroid
        cluster_idx = torch.argmin(distances, dim=1)

        # compute the new centroids
        new_centroids = torch.stack([X[cluster_idx == i].mean(0) for i in range(k)])

        # end when we have converged
        if torch.all(centroids == new_centroids):
            break

        centroids = new_centroids
    return cluster_idx, centroids


def skeleton_sampling(X:Tensor, k: int=1000, method: Union['uniform', 'kmeans']='kmeans', n_iter: int=1000):
    if method == 'uniform':
        clustroid_idx = np.random.choice(a=X.shape[0], size=k, replace=False)
    elif method == 'kmeans':
        # extracting a skeleton from the cloudpoint
        _, centroids = k_means(X, k=k, n_iter=n_iter)

        clustroid_idx = []
        for point in centroids:
            clustroid_idx.append(torch.argmin(((X - point)**2).mean(dim=1)).item())
    else:
        raise ValueError('Invalid method for skeleton sampling')

    return clustroid_idx


def smoothL1(pred, target, keptcomponent=False):
    x = pred - target
    y = torch.minimum(x.abs(), x * x)
    if keptcomponent:
        return y.mean(0)
    else:
        return y.mean()


def smoothSoftmax(x):
    s = torch.nn.functional.softmax(x, dim=1)
    ss = torch.nn.functional.relu(x)
    num = ss * 0.1 + s
    denom = num.sum(1).unsqueeze(-1)
    return num / denom


class AttentionBlock(torch.nn.Module):
    def __init__(self, sIN, sOUT, yDIM=7, sPROJ=None):
        super(AttentionBlock, self).__init__()

        if sPROJ is None:
            sPROJ = sOUT

        self.k = torch.nn.Linear(yDIM, sPROJ)
        self.q = torch.nn.Linear(sIN, sPROJ)
        self.v = torch.nn.Linear(yDIM, sPROJ)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        YK = self.k(y)
        XQ = self.q(x)
        YV = self.v(y)

        M = torch.matmul(XQ, YK.t())
        M = smoothSoftmax(M)
        output = torch.matmul(M, YV)

        return output

class MLP(torch.nn.Module):
    def __init__(self, layers_sizes):
        super(MLP, self).__init__()
        layers = [torch.nn.Linear(layers_sizes[i], layers_sizes[i+1]) for i in range(len(layers_sizes)-1)]
        self.mlp = torch.nn.Sequential()
        for i, layer in enumerate(layers):
            self.mlp.append(layer)
            if i < len(layers) - 1:
                self.mlp.append(torch.nn.ReLU())
    def forward(self, x):
        return self.mlp(x)

class Ransformer(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Ransformer, self).__init__()

        self.k = kwargs['k']

        self.batch_norm = torch.nn.BatchNorm1d(16)

        self.layer_norm1 = torch.nn.LayerNorm(16)
        self.att1 = AttentionBlock(16, 128, yDIM=8)
        self.mlp1 = MLP([128, 64, 64, 64, 128])

        self_layer_norm2 = torch.nn.LayerNorm(136)
        self.att2 = AttentionBlock(136, 64, yDIM=8)
        self.mlp2 = MLP([64, 32, 32, 32, 64])

        self_layer_norm3 = torch.nn.LayerNorm(72)
        self.att3 = AttentionBlock(72, 16, yDIM=8)
        self.mlp3 = MLP([16, 8, 8, 8, 16])

        self.encoder = torch.nn.Sequential(
            Linear(7, 64),
            nn.ReLU(),
            Linear(64, 64),
            nn.ReLU(),
            Linear(64, 16)
        )

        self.decoder = torch.nn.Sequential(
            Linear(16, 64),
            nn.ReLU(),
            Linear(64, 64),
            nn.ReLU(),
            Linear(64, 4)
        )

    def forward(self, x, y):
        # encoding x and y
        x_enc = self.encoder(x)
        y_enc = self.encoder(y)

        # batch normalization on the skeleton
        y_norm = self.batch_norm(y_enc)

        # first transformer layer 
        z = self.layer_norm1(x_enc)
        z = self.att1(z, y_norm)
        z = self.mlp1(z)

        # second transformer layer
        z = torch.cat([z, x_enc], dim=1)
        z = self.layer_norm2(z)
        z = self.att2(z, y_norm)
        z = self.mlp2(z)

        # third transformer layer
        z = torch.cat([z, x_enc], dim=1)
        z = self.layer_norm3(z)
        z = self.att3(z, y_norm)
        z = self.mlp3(z)

        # decoding the output
        out = self.decoder(z)

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

        self.model = Ransformer(**self.hparams)

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
        
        return DataLoader(dataset=torchDataset,batch_size=self.hparams["batch_size"])

    def train(self,train_dataset, save_path=None):
        train_dataset = self.process_dataset(dataset=train_dataset,training=True)
        print("Start training")
        model = global_train(self.device, train_dataset, self.model, self.hparams,criterion = 'L1Smooth', method=self.hparams["method"])
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
            # print(len(test_dataset))
            for i, data in enumerate(test_dataset):
                # print(i)
                data_clone = data.clone()
                data_clone = data_clone.to(self.device)

                clustroid_idx = skeleton_sampling(data.clone().pos, k=1000, method='kmeans')
                skeleton = torch.clone(data["x"][clustroid_idx, :])
                skeleton = skeleton.to(self.device)

                out = self.model(data_clone.x, skeleton)

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

def global_train(device, train_dataset, network, hparams, criterion = 'L1Smooth', reg = 1, method="kmeans"):
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
        if hparams['subsampling'] != "None":
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
        else:
            train_loader = train_dataset

        train_loss, _, loss_surf_var, loss_vol_var, loss_surf, loss_vol = train_model(device, model, train_loader, optimizer, lr_scheduler, criterion, reg = reg, method=method)        
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


def train_model(device, model, train_loader, optimizer, scheduler, criterion='L1Smooth', reg: Union[int, None]=1.0, method="kmeans"):
    model.train()
    avg_loss_per_var = torch.zeros(4, device = device)
    avg_loss = 0
    avg_loss_surf_var = torch.zeros(4, device = device)
    avg_loss_vol_var = torch.zeros(4, device = device)
    avg_loss_surf = 0
    avg_loss_vol = 0
    iterNum = 0

    # print(len(train_loader))
    for i, data in enumerate(train_loader):
        # print(i)
        data_clone = data.clone()
        data_clone = data_clone.to(device)   
        optimizer.zero_grad()

        clustroid_idx = skeleton_sampling(data.clone().pos, k=1000, method=method)
        skeleton = torch.clone(data["x"][clustroid_idx,:])
        skeleton = skeleton.to(device)

        out = model(data_clone.x, skeleton)
        targets = data_clone.y

        loss_criterion = nn.MSELoss(reduction = 'none')
        if criterion == 'MSE':
            loss_criterion = nn.MSELoss(reduction = 'none')
        elif criterion == 'MAE':
            loss_criterion = nn.L1Loss(reduction = 'none')
        elif criterion == 'L1Smooth':
            loss_criterion = smoothL1
        
        loss_per_var = loss_criterion(out, targets).mean(dim = 0)
        total_loss = loss_per_var.mean()
        loss_surf_var = loss_criterion(out[data_clone.surf, :], targets[data_clone.surf, :]).mean(dim = 0)
        loss_vol_var = loss_criterion(out[~data_clone.surf, :], targets[~data_clone.surf, :]).mean(dim = 0)
        loss_surf = loss_surf_var.mean()
        loss_vol = loss_vol_var.mean()

        if (reg is not None):            
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

    return  avg_loss.cpu().data.numpy()/iterNum,            \
            avg_loss_per_var.cpu().data.numpy()/iterNum,    \
            avg_loss_surf_var.cpu().data.numpy()/iterNum,   \
            avg_loss_vol_var.cpu().data.numpy()/iterNum,    \
            avg_loss_surf.cpu().data.numpy()/iterNum,       \
            avg_loss_vol.cpu().data.numpy()/iterNum
