# utils/train_utils.py

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def train_predict(model, train_loader, val_loader, criterion, optimizer, epochs, scheduler=None, early_stopping=None, device='cuda'):
    """
    Train and validate the model.
    
    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epochs (int): Number of epochs.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        early_stopping (EarlyStopping, optional): Early stopping instance.
        device (str): Device to train on ('cpu' or 'cuda').
    
    Returns:
        nn.Module: Trained model.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            inputs, padding_mask, labels = batch
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()

            outputs = model(inputs, src_key_padding_mask=padding_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        epoch_loss = running_loss / total_batches

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        total_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                val_inputs, val_padding_mask, val_labels = batch
                val_inputs = val_inputs.float().to(device)
                val_labels = val_labels.long().to(device)
                val_padding_mask = val_padding_mask.to(device)

                val_outputs = model(val_inputs, src_key_padding_mask=val_padding_mask)
                loss = criterion(val_outputs, val_labels)
                val_running_loss += loss.item()
                total_val_batches += 1

        val_loss = val_running_loss / total_val_batches

        # Scheduler step
        if scheduler:
            scheduler.step(val_loss)

        # Early stopping check
        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    # Load best model weights
    model.load_state_dict(best_model_wts)
    model.eval()

    return model

def pad_sequences(sequences, max_len, cls_token):
    """
    Pad or truncate sequences and add a CLS token at the beginning.
    
    Args:
        sequences (list or np.ndarray): Original sequence data.
        max_len (int): Maximum sequence length.
        cls_token (np.ndarray): CLS token to add at the beginning, shape (1, feature_dim).
    
    Returns:
        torch.Tensor: Padded sequences, shape [num_samples, max_len+1, feature_dim].
    """
    max_len_with_cls = max_len + 1
    padded = []
    for seq in sequences:
        seq_with_cls = np.vstack([cls_token, seq])  
        if seq_with_cls.shape[0] < max_len_with_cls:
            seq_with_cls = np.pad(seq_with_cls, ((0, max_len_with_cls - seq_with_cls.shape[0]), (0, 0)), mode='constant')
        else:
            seq_with_cls = seq_with_cls[:max_len_with_cls]
        padded.append(seq_with_cls)
    return torch.tensor(padded, dtype=torch.float32)

def generate_padding_mask(sequences):
    """
    Generate padding masks.
    
    Args:
        sequences (torch.Tensor): Padded sequences, shape [num_samples, max_len, feature_dim].
    
    Returns:
        torch.Tensor: Padding masks, shape [num_samples, max_len], where True indicates padding.
    """
    return (sequences.abs().sum(dim=-1) == 0)


def train_predict_edl(model, train_loader, val_loader, l2e, criterion, optimizer, epochs, scheduler=None, early_stopping=None, device='cuda'):
    """
    训练和验证模型，处理 padding masks，并使用 Evidential Deep Learning 思路。

    Args:
        model (nn.Module): 待训练的模型。
        train_loader (DataLoader): 训练数据加载器，返回 (inputs, padding_mask, labels)。
        val_loader (DataLoader): 验证数据加载器，返回 (inputs, padding_mask, labels)。
        l2e (callable): logits to evidence 的转换函数。
        criterion (callable): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        epochs (int): 训练轮数。
        scheduler (torch.optim.lr_scheduler, optional): 学习率调度器。
        early_stopping (EarlyStopping, optional): 早停机制实例。
        device (str): 训练设备，'cpu' 或 'cuda'。

    Returns:
        nn.Module: 训练好的模型。
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            inputs, padding_mask, labels = batch
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            padding_mask = padding_mask.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs, src_key_padding_mask=padding_mask)
            evidence = l2e(outputs)  # logits to evidence
            alpha = evidence + 1  # Dirichlet 分布的参数 alpha

            # 计算不确定性和概率
            uncertainty, prob = uncertainty_and_probabilities(alpha)

            # 使用自定义损失函数
            loss = criterion(labels, alpha, global_step=epoch, device=device)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        epoch_loss = running_loss / total_batches

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        total_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                val_inputs, val_padding_mask, val_labels = batch
                val_inputs = val_inputs.float().to(device)
                val_labels = val_labels.long().to(device)
                val_padding_mask = val_padding_mask.to(device)

                val_outputs = model(val_inputs, src_key_padding_mask=val_padding_mask)

                # 转换 logits 为 evidence，计算 alpha
                evidence = l2e(val_outputs)
                alpha = evidence + 1

                # 计算不确定性和概率
                uncertainty, prob = uncertainty_and_probabilities(alpha)

                # 计算验证损失
                loss = criterion(val_labels, alpha, global_step=None,device=device)
                val_running_loss += loss.item()
                total_val_batches += 1

        val_loss = val_running_loss / total_val_batches

        # 学习率调度器更新
        if scheduler:
            scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    model.eval()

    return model

def uncertainty_and_probabilities(alpha):
    """
    Calculate uncertainty and probabilities from Dirichlet parameters.
    
    Args:
        alpha (torch.Tensor): Dirichlet parameters, shape [batch_size, num_classes].
    
    Returns:
        tuple: (uncertainty, probabilities)
    """
    total_evidence = torch.sum(alpha, dim=1, keepdim=True)
    K = alpha.size(1)  # Number of classes
    uncertainty = K / total_evidence  # Uncertainty measure
    probabilities = alpha / total_evidence  # Class probabilities
    return uncertainty, probabilities
