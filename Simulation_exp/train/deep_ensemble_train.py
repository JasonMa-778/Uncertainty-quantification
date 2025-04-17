# train/deep_ensemble_train.py

import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.transformer import TransformerWithPooling
from utils.dataset import MyDataset
from utils.early_stopping import EarlyStopping
from utils.train_utils import train_predict, pad_sequences, generate_padding_mask
from data.generate_data import generate_data
from torch.utils.data import DataLoader

def deep_ensemble_training(seed, pooling='cls', learning_rate=1e-3, epochs=50, n_trainings=30, v=0, 
                           max_seq_length=100, embedding_size=100, dim_feedforward=512, device='cpu'):
    """
    Train models using the Deep Ensemble method.
    
    Args:
        seed (int): Random seed.
        pooling (str): Pooling method ('cls', 'mean', 'mean_no_pad', 'mask_np_mp').
        learning_rate (float): Learning rate.
        epochs (int): Number of training epochs.
        n_trainings (int): Number of ensemble models to train.
        v (int): Data version or identifier.
        max_seq_length (int): Maximum sequence length.
        embedding_size (int): Embedding dimension.
        dim_feedforward (int): Dimension of the feedforward network.
        device (str): Device to train on ('cpu' or 'cuda').
    """
    # Load data
    data = generate_data(embedding_size, 15000, 50, v, seed)
    X_train, X_test, y_train, y_test, beta, beta0, maximum_auc, Vi, train_vi, test_vi, train_vib, test_vib = data

    # Define CLS token
    cls_token = np.random.randn(1, embedding_size) + 0

    # Pad sequences
    padded_train_data = pad_sequences(X_train, max_seq_length, cls_token)
    padded_test_data = pad_sequences(X_test, max_seq_length, cls_token)
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

        for i in tqdm(range(n_trainings), desc=f"Config {heads}h{layers}l"):
            print(f"\nTraining Model {i+1}/{n_trainings} for config {heads}h{layers}l.")

            # Initialize model
            model = TransformerWithPooling(
                embedding_size=embedding_size,
                nhead=heads,
                dim_feedforward=dim_feedforward,
                num_layers=layers,
                num_classes=2,
                pooling=pooling,
                dropout=0.1
            )

            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)
            early_stopping = EarlyStopping(patience=5, min_delta=0.001, 
                                           save_path=f'best_model_{heads}h{layers}l_model{i}.pt')

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

            # Evaluate the model
            all_probs = []
            trained_model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    test_inputs, val_padding_mask, _ = batch
                    test_inputs = test_inputs.float().to(device)
                    val_padding_mask = val_padding_mask.to(device)

                    outputs = trained_model(test_inputs, src_key_padding_mask=val_padding_mask)
                    probs = F.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu().numpy())

            probabilities = np.concatenate(all_probs)
            predictions_df[i] = probabilities[:, 1]
            predicted_classes = (probabilities[:, 1] >= 0.5).astype(int)
            accuracy = np.mean(predicted_classes == y_test)
            print(f"Model {i+1}/{n_trainings}, Test Accuracy: {accuracy:.4f}")

            # Clean up
            del model
            torch.cuda.empty_cache()

        result.append(predictions_df)

    # Save predictions
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    predictions_path = os.path.join(model_dir, f'ntrain_{pooling}_{heads}h{layers}l_{seed}.pkl')
    with open(predictions_path, 'wb') as f:
        pickle.dump(result, f)
