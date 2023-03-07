from sys import argv
from Searches._param_dicts import *
from stocksGAN import stocksGAN
import torch
import torch.nn as nn
import random

class RandomSearch(stocksGAN):

    """
    Random Search

    """
    
    def __init__(
        self,
        model: stocksGAN,
        data,
        param_dict: dict = dict_main,
        ):
        """
        Parameters:

        model: stocksGAN object
            Model to be trained.

        data: array-like, shape (n_samples, n_features)
            Data to train the model.

        param_dict: dict, default = dict_main
            Dictionary containing the hyperparameters to be optimized.

        Attributes:

        best_params_: dict
            Dictionary containing the best hyperparameters found
            according to the loss function.

        best_loss_: float
            Best loss found.

        Methods:

        fit(self, n_iter, verbose): Fit the model to the data.


        

        """
        self.model = model
        self.data = data
        self.param_dict = param_dict

        super().__init__(
            data = self.data,
            train_size = self.model.train_size,
            epochs = self.model.epochs,
            batch_size = self.model.batch_size,
            lr = self.model.lr,
            b1 = self.model.b1,
            b2 = self.model.b2,
            clip_value = self.model.clip_value,
            random_state = self.model.random_state,
            verbose = self.model.verbose
            )
        

    def fit(self,
            n_iter: int,
            verbose: bool = False,
            save_history: str = False
            ):
        """
        Fit the model to the data.

        Parameters:

        n_iter: int
            Number of iterations to run the search.

        verbose: bool, default = False
            If True, prints the best parameters found at each iteration.

        save_history: str, default = False
            Saves the history to the specified path.
            

        """
        best_loss = float('inf')
        best_params = {}
        for i in range(n_iter):
            params = {}
            for key, value in self.param_dict.items():
                params[key] = random.choice(value)

            self.model = stocksGAN(
                data = self.data,
                train_size = params['train_size'],
                epochs = params['epochs'],
                batch_size = params['batch_size'],
                lr = params['lr'],
                b1 = params['b1'],
                b2 = params['b2'],
                clip_value = params['clip_value'],
                random_state = params['random_state'],
                verbose = self.model.verbose
                )
            self.model.fit()
            loss = self.model.loss
            if loss < best_loss:
                best_loss = loss
                best_params = params
                if verbose:
                    print(f'Iteration {i+1}: Best loss = {best_loss}')
                    print(f'Best parameters: {best_params}')
                
            if save_history:
                with open(save_history, 'a') as f:
                    f.write(f'Iteration {i+1}: Best loss = {best_loss}\n')
                    f.write(f'Best parameters: {best_params}\n')

        self.best_params_ = best_params
        self.best_loss_ = best_loss
        






