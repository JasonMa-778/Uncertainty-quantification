# utils/early_stopping.py

import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, save_path='best_model.pt'):
        """
        Initialize EarlyStopping.
        
        Args:
            patience (int): Number of epochs to wait for improvement.
            min_delta (float): Minimum change to qualify as improvement.
            save_path (str): Path to save the best model.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        """
        Update EarlyStopping state and save model if improvement occurs.
        
        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): Model to save.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

    def save_checkpoint(self, model):
        """
        Save the model checkpoint.
        
        Args:
            model (torch.nn.Module): Model to save.
        """
        torch.save(model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")
