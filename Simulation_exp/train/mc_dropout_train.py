# train/mc_dropout_train.py

import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.mc_dropout import TransformerWithMCDropout
from utils.dataset import MyDataset
from utils.early_stopping import EarlyStopping
from utils.train_utils import train_predict, pad_sequences, generate_padding_mask
from data.generate_data import generate_data
from torch.utils.data import DataLoader

def mc_dropout_predict(model, data_loader, num_samples, device='cpu'):
    """
    Perform MC Dropout predictions.
    
    Args:
        model (nn.Module): Trained model.
        data_loader (DataLoader): Data loader for prediction.
        num_samples (int): Number of Monte Carlo samples.
        device (str): Device to run predictions on.
    
    Returns:
        pd.DataFrame: DataFrame containing prediction probabilities.
    """
    model.eval()
    model.enable_dropout()
    all_predictions = []
    for _ in range(num_samples):
        all_probs = []
        with torch.no_grad():
            for batch in data_loader:
                test_inputs, val_padding_mask, _ = batch
                test_inputs = test_inputs.float().to(device)
                val_padding_mask = val_padding_mask.to(device)

                outputs = model(test_inputs, src_key_padding_mask=val_padding_mask)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy()[:, 1])
        probabilities = np.concatenate(all_probs)
        all_predictions.append(probabilities)
    all_predictions = np.array(all_predictions)
    predictions_df = pd.DataFrame(all_predictions.T)
    return predictions_df

def transformer_mc_dropout_training(seed, pooling='cls', learning_rate=1e-3, epochs=50, 
                                   n_trainings=30, mc_samples=30, dropout_rate=0.5, 
                                   device='cpu'):
    """
    Train models using the MC Dropout method.
    
    Args:
        seed (int): Random seed.
        pooling (str): Pooling method ('cls', 'mean', 'mean_no_pad', 'mask_np_mp').
        learning_rate (float): Learning rate.
        epochs (int): Number of training epochs.
        n_trainings (int): Number of models to train.
        mc_samples (int): Number of Monte Carlo samples.
        dropout_rate (float): Dropout rate.
        device (str): Device to train on ('cpu' or 'cuda').
    """
    # Load data
    embedding_size = 100  # Assuming embedding_size is 100
    data = generate_data(embedding_size, 15000, 50, 0, seed)
    X_train, X_test, y_train, y_test, _, _, _, _, train_vi, test_vi, train_vib, test_vib = data

    # Define CLS token
    cls_token = np.random.randn(1, embedding_size) + 0

    # Pad sequences
    padded_train_data = pad_sequences(X_train, 50, cls_token)  # max_seq_length=50
    padded_test_data = pad_sequences(X_test, 50, cls_token)
    print(padded_train_data.shape)

    # Generate padding masks
    padding_masks_train = generate_padding_mask(padded_train_data)
    padding_masks_test = generate_padding_mask(padded_test_data)

    # Create datasets
    train_dataset = MyDataset(padded_train_data, y_train, padding_masks_train)
    val_dataset = MyDataset(padded_test_data, y_test, padding_masks_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=3000, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=3000, shuffle=False)

    # Transformer configurations
    configs = [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (4, 1)
    ]

    result = []

    # Iterate over different configurations
    for heads, layers in configs:
        print(f"\nTraining models with {heads} heads and {layers} layers.")
        n_samples = len(y_test)
        predictions_df = pd.DataFrame(index=range(n_samples), columns=range(n_trainings))
        subres = []

        for i in tqdm(range(n_trainings), desc=f"Config {heads}h{layers}l"):
            print(f"\nTraining Model {i+1}/{n_trainings} for config {heads}h{layers}l.")

            # Initialize model
            model = TransformerWithMCDropout(
                embedding_size=100, 
                nhead=heads, 
                num_layers=layers, 
                dim_feedforward=512, 
                num_classes=2, 
                pooling=pooling, 
                dropout=dropout_rate
            )

            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
            early_stopping = EarlyStopping(patience=5)

            # Train the model
            trained_model = train_predict(
                model=model,
                train_loader=train_loader, 
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                epochs=epochs,
                scheduler=scheduler,
                early_stopping=early_stopping,
                device=device
            )

            # Perform MC Dropout predictions
            predictions_df = mc_dropout_predict(trained_model, val_loader, mc_samples, device=device)
            subres.append(predictions_df)

            # Calculate accuracy
            predicted_classes = np.round(predictions_df.mean(axis=1))
            accuracy = np.mean(predicted_classes == y_test)
            print(f"Accuracy: {accuracy}")

            # Clean up
            del model
            torch.cuda.empty_cache()

        result.append(subres)

    # Save predictions
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    with open(f'model/mcdropout_{pooling}_seed{seed}_.pkl', 'wb') as f:
        pickle.dump(result, f)
    print("All results have been saved.")
