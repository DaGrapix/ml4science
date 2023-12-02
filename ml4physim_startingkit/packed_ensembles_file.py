import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch_uncertainty.layers import PackedLinear
from einops import rearrange
from tqdm import tqdm


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

    def __init__(self,
                 name: str = "PackedMLP",
                 input_size: int = None,
                 output_size: int = None,
                 hidden_sizes: tuple = (100, 100,),
                 activation=F.relu,
                 dropout: bool = False,
                 M: int = 4,
                 alpha: int = 2,
                 gamma: int = 1,
                 device: str = "cpu"):
        super().__init__()

        # dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.2)
        else:
            self.dropout = nn.Identity()

        self.name = name
        self.device = device

        self.activation = activation
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        self.num_estimators = M

        self.input_layer = PackedLinear(self.input_size, self.hidden_sizes[0], alpha=alpha, num_estimators=M,
                                        gamma=gamma, first=True,
                                        device=device)
        self.hidden_layers = nn.ModuleList(
            [PackedLinear(in_f, out_f, alpha=alpha, num_estimators=M, gamma=gamma, device=device) \
             for in_f, out_f in zip(self.hidden_sizes[:-1], self.hidden_sizes[1:])])
        self.output_layer = PackedLinear(self.hidden_sizes[-1], self.output_size, alpha=alpha, num_estimators=M,
                                         gamma=gamma, last=True,
                                         device=device)

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
            out = self.dropout(out)
        out = self.output_layer(out)

        return out


def process_dataset(dataset, batch_size: int = 128000, training: bool = False, shuffle: bool = False,
                    n_workers: int = 0):
    """
    This function allows to create a DataLoader from the airfrans data files.

    Parameters
    ----------
    dataset: DataSet
        The airfrans dataset to process
    batch_size: int (default: 128000)
        The batch size
    training: bool (default: False)
        Whether the dataset is for training or not
    shuffle: bool (default: False)
        Whether to shuffle the data or not
    n_workers: int (default: 0)
        The number of cpu subprocesses to use for data loading

    Returns
    -------
    data_loader: DataLoader
        The data loader
    """
    
    if training:
        batch_size = batch_size
        extract_x, extract_y = dataset.extract_data()
    else:
        batch_size = batch_size
        extract_x, extract_y = dataset.extract_data()

    torch_dataset = TensorDataset(torch.from_numpy(extract_x).float(), torch.from_numpy(extract_y).float())
    data_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)

    return data_loader


def infer_input_output_size(dataset):
    """
    This function retrieves the input and output sizes of the airfrans dataset
    
    Parameters
    ----------
    dataset: DataSet
        The dataset to process
        
    Returns
    -------
    input_size: int
        The size of the input
    output_size: int
        The size of the output
    """

    *dim_inputs, output_size = dataset.get_sizes()
    input_size = np.sum(dim_inputs)

    return input_size, output_size


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after certain epochs
    """

    def __init__(self, tolerance=5, min_rate=0.0001):
        """
        Parameters
        ----------
        tolerance: int
            Number of epochs to wait for improvement before stopping the training
        min_rate: float
            Minimum rate of improvement to consider improvement
        """

        self.tolerance = tolerance
        self.min_rate = min_rate
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, previous_train_loss):
        """
        Parameters
        ----------
        train_loss: float
            The current training loss
        previous_train_loss: float
            The previous training loss
        """
        
        if (previous_train_loss - train_loss)/previous_train_loss > self.min_rate:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True


def train(model, train_loader, val_loader=None, epochs=100, lr=3e-4, device="cpu", tolerance=5, min_rate=0.0001, verbose=False):
    """
    This function allows to train a Packed-Ensemble model using the provided train_loader DataLoader.
    
    Parameters
    ----------
    model: nn.Module
        The Packed-Ensemble model to train
    train_loader: DataLoader
        The DataLoader for the training dataset
    val_loader: DataLoader (default: None)
        The DataLoader for the validation dataset
    epochs: int (default: 100)
        The number of training epochs
    lr: float (default: 3e-4)
        The learning rate
    device: str (default: "cpu")
        The device to use
    tolerance: int (default: 5)
        The number of epochs to wait for improvement before stopping the training
    min_rate: float (default: 0.0001)
        The minimum rate of improvement to consider it as improvement
    verbose: bool (default: False)
        Whether to print information or not

    Returns
    -------
    model: nn.Module
        The trained Packed-Ensemble model
    train_losses: list
        The training losses
    val_losses: list
        The validation losses
    """

    train_losses = []
    val_losses = []

    # select your optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # select your loss function
    loss_function = nn.MSELoss()

    early_stopping = EarlyStopping(tolerance=tolerance, min_rate=min_rate)

    if verbose:
        pbar = tqdm(range(epochs), desc="Epochs")
    else:
        pbar = range(epochs)

    for epoch in pbar:
        # set your model for training
        model.train()
        total_loss = 0

        if verbose:
            pbar_batch = tqdm(train_loader)
        else:
            pbar_batch = train_loader

        # iterate over the batches of data
        for batch in pbar_batch:
            data, target = batch
            # transfer your data on proper device. The model and your data should be on the same device
            data = data.to(device)
            target = target.to(device)

            # reset the gradient
            optimizer.zero_grad()
            
            # predict using your model on the current batch of data
            prediction = model(data)
            
            # compute the loss between prediction and real target, by repeating the target so it fits the different estimators 
            loss = loss_function(prediction, target.repeat(model.num_estimators, 1))
            
            # compute the gradient (backward pass of back propagation algorithm)
            loss.backward()
            
            # update the parameters of your model
            optimizer.step()
            total_loss += loss.item() * len(data)
        
        # the validation step is optional
        if val_loader is not None:
            val_loss = validate(model, val_loader, device)
            val_losses.append(val_loss)

        mean_loss = total_loss / len(train_loader.dataset)
        train_losses.append(mean_loss)

        if verbose: print(f"Train Epoch: {epoch}   Avg_Loss: {mean_loss:.5f}")
        
        if epoch > 0:
            early_stopping(train_losses[-1], train_losses[-2])
        if early_stopping.early_stop: 
            if verbose: print(f"Early stopping at epoch: {epoch}")
            break

    return model, train_losses, val_losses


def validate(model, val_loader, device="cpu", verbose=False):
    """
    This function allows to validate a Packed-Ensemble model using the provided validation dataset DataLoader.

    Parameters
    ----------
    model: nn.Module
        The Packed-Ensemble model to evaluate
    val_loader: DataLoader
        The data loader for the validation dataset
    device: str (default: "cpu")
        The device to use
    verbose: bool (default: False)
        Whether to print information or not

    Returns
    -------
    mean_loss: float
        The mean loss
    """
    # set the model for evaluation (no update of the parameters)
    model.eval()
    total_loss = 0
    loss_function = nn.MSELoss()
    
    with torch.no_grad():
        if verbose:
            pbar = tqdm(val_loader)
        else:
            pbar = val_loader

        for batch in pbar:
            data, target = batch
            data = data.to(device)
            target = target.to(device)
            prediction = model(data)
            loss = loss_function(prediction, target.repeat(model.num_estimators, 1))
            total_loss += loss.item() * len(data)
        mean_loss = total_loss / len(val_loader.dataset)
        if verbose:
            print(f"Eval:   Avg_Loss: {mean_loss:.5f}")
    return mean_loss


def predict(model, dataset, device="cpu", verbose=False):
    """
    This function computes the prediction of the trained Packed-Ensemble model on the dataset.
    It also provides the observation values of the dataset.

    Parameters
    ----------
    model: nn.Module
        The trained Packed-Ensemble model
    dataset: DataSet
        The dataset to infer on
    device: str
        The device to use

    Returns
    -------
    predictions: np.array
        The predictions
    observations: np.array
        The observations of the dataset
    """

    # set the model for the evaluation
    model.eval()
    predictions = []
    observations = []
    test_loader = process_dataset(dataset, training=False, shuffle=False)
    
    # we dont require the computation of the gradient
    with torch.no_grad():
        if verbose:
            pbar = tqdm(test_loader)
        else:
            pbar = test_loader

        for batch in pbar:
            data, target = batch
            data = data.to(device)
            target = target.to(device)
            prediction = model(data)

            #averaging the predictions of the different ensemble models
            packed_split = rearrange(prediction, '(n b) m -> b n m', n=model.num_estimators)
            packed_prediction = packed_split.mean(dim=1)

            if device == torch.device("cpu"):
                predictions.append(packed_prediction.numpy())
                observations.append(target.numpy())
            else:
                predictions.append(packed_prediction.cpu().data.numpy())
                observations.append(target.cpu().data.numpy())

    # reconstruct the prediction in the proper required shape of target variables
    predictions = np.concatenate(predictions)
    predictions = dataset.reconstruct_output(predictions)
    
    # Do the same for the real observations
    observations = np.concatenate(observations)
    observations = dataset.reconstruct_output(observations)

    return predictions, observations
