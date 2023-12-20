import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import itertools as it
from typing import Union
import matplotlib.pyplot as plt
from lips.dataset.scaler import Scaler
from my_packed_ensemble import *


def build_k_indices(num_row: int, k_fold: int, seed: Union[int, None] = None):
    """build k indices for k-fold.

    Parameters
    ----------
    num_row : int
        Number of rows in the dataset.
    k_fold : int
        Number of folds
    seed : int
        Seed for random generator

    Returns
    -------
    k_indices : np.array
        Array of indices for each fold"""
    
    if seed is not None:
        np.random.seed(seed)

    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    
    return np.array(k_indices)


def save_losses(train_losses_list: list, val_losses_list: list, folder: str, file_name: str):
    """
    Saves the training and validation losses.

    Parameters
    ----------
    train_losses_list : list
        List containing the training losses.
    val_losses_list : list
        List containing the validation losses.
    folder : str
        Folder where the losses will be saved.
    file_name : str
        Name of the file where the losses will be saved.

    Returns
    -------
    0 : int
        Returns 0 if the function runs successfully.
    """

    # create folder if it does not exist
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # save losses
    df = pd.DataFrame({'train_loss': train_losses_list, 'val_loss': val_losses_list})
    df.to_csv(folder + "/" + file_name, index=False)

    return 0



def save_training_validation_losses_plot(train_losses_list: list, val_losses_list: list,
                                         hyperparam_dict: dict, folder: str, plot_name: str):
    """
    Saves the training and validation losses plot.

    Parameters
    ----------
    train_losses_list : list
        List containing the training losses.
    val_losses_list : list
        List containing the validation losses.

    Returns
    -------
    0 : int
        Returns 0 if the function runs successfully.
    """

    # create folder if it does not exist
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # clear previous plot
    plt.clf()

    plt.plot(train_losses_list, label='Training loss', color='blue')
    plt.plot(val_losses_list, label='Validation loss', color='red')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.title(
        f'Losses for hidden_sizes={hyperparam_dict["hidden_sizes"]}, dropout={hyperparam_dict["dropout"]}, M={hyperparam_dict["M"]}, \n alpha={hyperparam_dict["alpha"]}, gamma={hyperparam_dict["gamma"]}, lr={hyperparam_dict["lr"]}')
    plt.legend()
    plt.savefig(folder + "/" + plot_name)

    return 0


def get_hyper_dict(param_grid: dict):
    """
    Returns a grid of hyperparameters.

    Parameters
    ----------
    param_grid : dict
        Dictionary containing the hyperparameters.

    Returns
    -------
    hyperparameter_dict : dict
        Dictionary containing the list of hyperparameters with the ith element of each list being one unique combination of the hyperparameters.
    """

    # generate all combinations of parameter values
    combinations = it.product(*(param_grid[key] for key in param_grid))

    # create a new dictionary with keys as hyperparameter names and values as lists of combinations
    hyperparameter_dict = {key: [] for key in param_grid}

    # fill in the values for each key in the new dictionary
    for combo in combinations:
        for i, key in enumerate(param_grid):
            hyperparameter_dict[key].append(combo[i])
    
    return hyperparameter_dict


def sample_data(data_x, data_y, size_scale, seed:Union[int, None] = None):
    """
    Samples the data to reduce the size of the dataset to size_scale.

    Parameters
    ----------
    data_x : np.array
        Array containing the input data.
    data_y : np.array
        Array containing the output data.
    size_scale : float
        Percentage of the data to be sampled.

    Returns
    -------
    processed_x : np.array
        Array containing the sampled input data.
    processed_y : np.array
        Array containing the sampled output data.        
    """

    if seed is not None:
        np.random.seed(seed)
    #sample uniformly
    sample_indices = np.random.choice(data_x.shape[0], int(size_scale*data_x.shape[0]), replace=False)

    processed_x = data_x[sample_indices]
    processed_y = data_y[sample_indices]

    return processed_x, processed_y


def hyperparameters_tuning(benchmark: DataSet, param_grid: dict, k_folds: int, num_epochs: int, batch_size: int = 128000,
                           shuffle: bool = False, n_workers: int = 0, seed: int = 27, scaler: Union[Scaler, None] = None,
                           partition: int = 0, verbose: bool=False, size_scale: float=0.3, device: str="cpu"):
    """
    Performs hyperparameter tuning using K-fold cross validation.

    Parameters
    ----------
    param_grid : dict
        Dictionary containing the values for each hyperparameter to be tested.
    k_folds : int
        Number of folds to be used in the cross validation.
    num_epochs : int
        Number of epochs to be used in the training.
    batch_size : int
        Batch size to be used in the training.
    shuffle : bool
        Whether to shuffle the training dataset.
    n_workers : int
        Number of workers to be used in the training.
    seed : int
        Random seed to be used in the training.
    scaler : Scaler
        Scaler to be used in the model.
    partition : int
        Partition of the hyperparameter grid to be used in this run.

    Returns
    -------
    0 : int
        Returns 0 if the function runs successfully.
    """

    # set the random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if verbose: print(f"Device: {device}")

    # get the hyperparameter grid dictionary
    hyperparameter_dict = get_hyper_dict(param_grid)
    hyperparameters_size = len(hyperparameter_dict[list(hyperparameter_dict.keys())[0]])

    dataset = benchmark.train_dataset
    input_size, output_size = infer_input_output_size(dataset)

    extract_x, extract_y = dataset.extract_data()

    # sample only a fraction of the dataset
    extract_x, extract_y = sample_data(extract_x, extract_y, size_scale, seed=None)

    try:
        results_df = pd.read_csv(f"CV/partition_{partition}/results_{partition}.csv")
    except:
        results_df = pd.DataFrame(columns=[*param_grid.keys(), "validation_loss"])

    try:
        # read the checkpoint
        with open(f"CV/partition_{partition}/checkpoint_{partition}.txt", "rb") as f:
            checkpoint = int(f.read())
    except:
        checkpoint = -1

    indices = range(hyperparameters_size)

    if partition == 0:
        indices = indices[:hyperparameters_size // 3]
    elif partition == 1:
        indices = indices[hyperparameters_size // 3: 2 * hyperparameters_size // 3]
    elif partition == 2:
        indices = indices[partition * hyperparameters_size // 3:]
    else:
        raise ValueError("Invalid partition value. It must be 0, 1 or 2.")

    # remove the indices that have already been processed
    indices = indices[checkpoint + 1:]
    if verbose: indices = tqdm(indices)
    for i in indices:
        param_dict = {
            'hidden_sizes': hyperparameter_dict["hidden_sizes"][i],
            'dropout': hyperparameter_dict["dropout"][i],
            'M': hyperparameter_dict["M"][i],
            'alpha': hyperparameter_dict["alpha"][i],
            'gamma': hyperparameter_dict["gamma"][i],
            'lr': hyperparameter_dict["lr"][i]
        }

        if verbose:
            print(f'Hyperparameters: {i}/hidden_sizes={hyperparameter_dict["hidden_sizes"][i]}, \
                    dropout={hyperparameter_dict["dropout"][i]}, M={hyperparameter_dict["M"][i]}, alpha={hyperparameter_dict["alpha"][i]}, \
                    gamma={hyperparameter_dict["gamma"][i]}, lr={hyperparameter_dict["lr"][i]}')

        # define the K-fold Cross Validator
        k_indices = build_k_indices(extract_y.shape[0], k_folds, seed=seed)
        summed_validation_loss = 0

        # k-fold Cross Validation model evaluation
        for fold in range(k_folds):
            if verbose: print(f"fold: {fold}")

            # initialize the Packed MLP model
            model = PackedMLP(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=hyperparameter_dict["hidden_sizes"][i],
                activation=F.relu,
                device=device,
                dropout=hyperparameter_dict["dropout"][i],
                M=hyperparameter_dict["M"][i],
                alpha=hyperparameter_dict["alpha"][i],
                gamma=hyperparameter_dict["gamma"][i],
                scaler=scaler
            )
            model.to(device)

            val_ids = k_indices[fold]
            train_ids = k_indices[~(np.arange(k_indices.shape[0]) == fold)]

            train_x = extract_x[train_ids]
            train_y = extract_y[train_ids]

            train_x = train_x.reshape(train_x.shape[0] * train_x.shape[1], -1)
            train_y = train_y.reshape(train_y.shape[0] * train_y.shape[1], -1)

            val_x = extract_x[val_ids]
            val_y = extract_y[val_ids]

            trainloader = model.process_dataset(data=(train_x, train_y), training=True, batch_size=batch_size,
                                                shuffle=shuffle, n_workers=n_workers)
            validateloader = model.process_dataset(data=(val_x, val_y), training=False, batch_size=batch_size,
                                                   shuffle=shuffle, n_workers=n_workers)

            model, train_losses, val_losses = train(model=model, train_loader=trainloader, val_loader=validateloader,
                                                    epochs=num_epochs, device=device, lr=hyperparameter_dict["lr"][i],
                                                    verbose=verbose)

            summed_validation_loss += val_losses[-1]

            # saving the losses
            save_losses(train_losses_list=train_losses, val_losses_list=val_losses,
                        folder=f"CV/partition_{partition}/losses_{partition}/hyperparameters_{i}", file_name=f'fold_{fold}.csv')

            # saving the curve
            save_training_validation_losses_plot(train_losses_list=train_losses, val_losses_list=val_losses,
                                                 hyperparam_dict=param_dict, folder=f"CV/partition_{partition}/plots_{partition}/hyperparameters_{i}",
                                                 plot_name=f'fold_{fold}.png')

        mean_validation_loss = summed_validation_loss / k_folds
        
        if verbose:
            # print fold results
            print('-------------------------------- \n')
            print(f'FOLD {fold} RESULTS FOR {i}th HYPERPARAMETERS')
            print(f'Average validation loss: {mean_validation_loss}')
            print('-------------------------------- \n')

        param_dict.update({'validation_loss': mean_validation_loss})
        results_df.loc[len(results_df)] = param_dict
        
        # save the results
        results_df.to_csv(f"CV/partition_{partition}/results_{partition}.csv", index=False)

        checkpoint += 1
        # save the checkpoint
        with open(f"CV/partition_{partition}/checkpoint_{partition}.txt", "wb") as f:
            f.write(str(checkpoint).encode())

    return 0