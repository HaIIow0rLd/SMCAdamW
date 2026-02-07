import os
import sys
import time
import datetime
import logging
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tonic
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from spikingjelly.activation_based import functional
from model import create_model  

def set_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(3407)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

def load_shd_data(batch_size=32, time_steps=100):
    data_dir = os.path.abspath('./data')
    shd_dir = os.path.join(data_dir, 'SHD')
    os.makedirs(shd_dir, exist_ok=True)
    
    train_dataset = tonic.datasets.SHD(save_to=data_dir, train=True)
    test_dataset = tonic.datasets.SHD(save_to=data_dir, train=False)
    sensor_size = train_dataset.sensor_size
    
    def events_to_frames(events, sensor_size, time_steps=100):
        if len(events) > 0:
            t_max = events['t'].max()
        else:
            t_max = 1.0
        
        frames = np.zeros((time_steps, sensor_size[0], sensor_size[1], sensor_size[2]))
        
        if len(events) > 0:
            time_window = t_max / time_steps
            
            for i in range(time_steps):
                t_start = i * time_window
                t_end = (i + 1) * time_window
                mask = (events['t'] >= t_start) & (events['t'] < t_end)
                window_events = events[mask]
                
                if len(window_events) > 0:
                    for x in window_events['x']:
                        frames[i, x, 0, 0] += 1
        
        return frames

    def collate_fn(batch, time_steps=100):
        all_frames = []
        all_targets = []
        
        for events, target in batch:
            frames = events_to_frames(events, sensor_size, time_steps)
            all_frames.append(torch.from_numpy(frames).float())
            all_targets.append(target)

        data = torch.stack(all_frames, dim=0)
        targets = torch.tensor(all_targets, dtype=torch.long)
        
        return data, targets
    
    def create_collate(time_steps):
        return lambda batch: collate_fn(batch, time_steps)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=create_collate(time_steps)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=create_collate(time_steps)
    )
    
    return train_loader, test_loader, sensor_size[0]

def train_epoch(model, loader, optimizer, criterion, device, epoch, logger):
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        data = data / 5.0
        functional.reset_net(model)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    elapsed = time.time() - start_time
    
    return avg_loss, accuracy, elapsed

def test_epoch(model, loader, criterion, device, logger):
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            data = data / 5.0
            functional.reset_net(model)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    elapsed = time.time() - start_time
    
    return avg_loss, accuracy, elapsed

class CSVLogger:
    def __init__(self, filename):
        self.filename = filename
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'timestamp', 'epoch',
                'train_loss', 'train_acc',
                'test_loss', 'test_acc',
                'elapsed_time'
            ])
    
    def log(self, epoch, train_stats, test_stats):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                timestamp, epoch,
                f"{train_stats[0]:.4f}", f"{train_stats[1]:.2f}",
                f"{test_stats[0]:.4f}", f"{test_stats[1]:.2f}",
                f"{train_stats[2] + test_stats[2]:.2f}"
            ])

def plot_results(log_file):
    try:
        df = pd.read_csv(log_file)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(df['epoch'], df['train_acc'], 'b-o', label='Train')
        ax1.plot(df['epoch'], df['test_acc'], 'r-o', label='Test')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Training Dynamics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(df['epoch'], df['train_loss'], 'b-o', label='Train Loss')
        ax2.plot(df['epoch'], df['test_loss'], 'r-o', label='Test Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        png_file = log_file.replace('.csv', '.png')
        plt.savefig(png_file, dpi=150)
        print(f"Chart saved to: {png_file}")
    except Exception as e:
        print(f"Error while plotting results: {e}")

def main():
    MODEL_TYPE = 'snn'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    logger = logging.getLogger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    TIME_STEPS = 16
    
    os.makedirs('./logs', exist_ok=True)
    os.makedirs(f'./logs/{MODEL_TYPE}', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs(f'./checkpoints/{MODEL_TYPE}', exist_ok=True)
    os.makedirs('./data', exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'./logs/{MODEL_TYPE}/{MODEL_TYPE}_shd_{timestamp}.csv'
    
    file_handler = logging.FileHandler(log_file.replace('.csv', '.log'), encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    
    logger.info("Loading SHD dataset...")
    train_loader, test_loader, input_size = load_shd_data(BATCH_SIZE, TIME_STEPS)
    logger.info(f"Data loaded successfully! Input size: {input_size}")
    
    model = create_model(MODEL_TYPE, input_size, 20, TIME_STEPS).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    logger.info(f"Dataset: SHD (Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)})")
    logger.info(f"Time Steps: {TIME_STEPS}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Model Type: {MODEL_TYPE.upper()}")
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1024:.1f}K")
    logger.info(f"Optimizer: {optimizer} (lr={LEARNING_RATE})")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("="*60)
    
    csv_logger = CSVLogger(log_file)
    best_acc = 0
    test_accuracies = []
    
    for epoch in range(1, EPOCHS + 1):
        logger.info(f"Starting Epoch {epoch}/{EPOCHS}...")
        
        train_stats = train_epoch(model, train_loader, optimizer, criterion, device, epoch, logger)
        test_stats = test_epoch(model, test_loader, criterion, device, logger)
        
        logger.info(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_stats[0]:.4f} | Train Acc: {train_stats[1]:.2f}% | "
            f"Test Loss: {test_stats[0]:.4f} | Test Acc: {test_stats[1]:.2f}% | "
            f"Time: {train_stats[2] + test_stats[2]:.2f}s"
        )
        
        csv_logger.log(epoch, train_stats, test_stats)
        test_accuracies.append(test_stats[1])
        
        # Save the best model
        if test_stats[1] > best_acc:
            best_acc = test_stats[1]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'model_type': MODEL_TYPE
            }, f'./checkpoints/{MODEL_TYPE}/{MODEL_TYPE}_shd_best_{timestamp}.pth')
            logger.info(f"Best model saved with accuracy: {best_acc:.2f}%")
        
        # Save the latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_stats[0],
            'test_acc': test_stats[1],
        }, f'./checkpoints/{MODEL_TYPE}/{MODEL_TYPE}_shd_last_{timestamp}.pth')
        
    avg_acc = sum(test_accuracies) / len(test_accuracies)
    logger.info(f"Training Complete! Best Acc: {best_acc:.2f}% | Avg Acc: {avg_acc:.2f}%")
    logger.info(f"Log file: {log_file}")
    
    try:
        plot_results(log_file)
    except Exception as e:
        logger.warning(f"Could not plot results: {e}")

if __name__ == '__main__':
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')
    
    main()