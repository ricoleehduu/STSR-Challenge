import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml
import time
import numpy as np

from data.dataset import DentalDataset
from models.main_model import RegistrationModel
from losses.registration_loss import registration_loss, transform_points
from utils.transform_utils import recover_original_coordinates


def main():
    # --- 1. Load configuration ---
    config_path = 'configs/train_config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # --- 2. Prepare experiment environment ---
    exp_dir = Path(config['output_dir']) / config['experiment_name']
    checkpoint_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"
    result_dir = exp_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard and log file
    writer = SummaryWriter(log_dir)
    log_file = open(exp_dir / "log.txt", "a")

    def print_and_log(message):
        print(message)
        log_file.write(message + '\n')

    print_and_log(f"--- start experiment: {config['experiment_name']} ---")
    print_and_log(f"configuration file:\n{yaml.dump(config)}")

    # --- 3. Device and model ---
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print_and_log(f"use device: {device}")

    model = RegistrationModel(feat_dim=config['feature_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # --- 4. Data loading ---
    train_dataset = DentalDataset(config['train_data_root'], config['jaw_type'],
                                  num_points_stl=config['num_points_stl'],
                                  num_points_cbct=config['num_points_cbct'],
                                  use_augmentation=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['num_workers'])

    val_dataset = DentalDataset(config['val_data_root'], config['jaw_type'],
                                num_points_stl=config['num_points_stl'],
                                num_points_cbct=config['num_points_cbct'],
                                use_augmentation=False)  # 验证集不使用数据增强
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'])

    # --- 5. Training and validation loop ---
    torch.cuda.empty_cache()
    best_val_loss = float('inf')
    start_time = time.time()
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()

        # -- Training --
        model.train()
        total_train_loss = 0

        for i, data in enumerate(train_loader):
            p_src = data['p_src'].to(device)
            p_tgt = data['p_tgt'].to(device)
            transform_gt = data['transform_gt'].to(device)

            optimizer.zero_grad()
            transform_pred = model(p_src, p_tgt)
            loss = registration_loss(p_src, transform_pred, transform_gt)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # -- Validation --
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for data in val_loader:
                p_src = data['p_src'].to(device)
                p_tgt = data['p_tgt'].to(device)
                transform_gt = data['transform_gt'].to(device)

                transform_pred = model(p_src, p_tgt)
                loss = registration_loss(p_src, transform_pred, transform_gt)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)

        # -- Print and save --
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time

        # print result
        print_and_log(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {epoch_time:.2f}s | Total Time: {total_time:.2f}s")

        # Save latest weights
        torch.save(model.state_dict(), checkpoint_dir / "latest_model.pth")

        # Save best weights
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            print_and_log(f"  -> Save best model，validating loss: {best_val_loss:.6f}")



if __name__ == '__main__':
    main()