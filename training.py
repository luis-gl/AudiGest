from genericpath import isfile
from ntpath import join
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn

from config_creator import get_config
from torch.utils import data
from tqdm import tqdm
from utils.model.MEADdataset import MEADDataset
from utils.model.AudiGest import AudiGest
from utils.plotter import plot_loss


def set_seed(seed: int, make_deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if make_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_last_epoch() -> int:
    checkpoints_dir = os.path.join('processed_data', 'training')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        print(f'No checkpoint detected.')
        return 0

    checkpoints = [int(file.split('.')[0].replace('AG_', ''))
                    for file in os.listdir(checkpoints_dir)
                    if os.path.isfile(os.path.join(checkpoints_dir, file))]

    if len(checkpoints) < 1:
        print(f'No checkpoint detected.')
        return 0

    last_checkpoint = max(checkpoints)
    print(f'Detected last checkpoint at epoch {last_checkpoint}.')
    return last_checkpoint


def train_step(train_dl: data.DataLoader, model: AudiGest, loss_fn: nn.MSELoss,
               optimizer: torch.optim.Optimizer, device: torch.device,
               epoch: int, num_epochs: int) -> float:
    model.train()

    optimizer.zero_grad()

    n = len(train_dl.dataset)
    train_loss = 0.0

    hidden = None

    train_loop = tqdm(enumerate(train_dl), total=len(train_dl), leave=False)
    for _, (melspec, mfcc, target, base_target) in train_loop:
        melspec = melspec.unsqueeze(1)
        mfcc = mfcc.permute(0, 2, 1)

        melspec = melspec.to(device)
        mfcc = mfcc.to(device)
        target = target.to(device)
        base_target = base_target.to(device)

        reconstructed, hidden = model(melspec, mfcc, base_target, hidden)
        loss = loss_fn(reconstructed, target)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        
        current_loss = loss.item() * train_dl.batch_size

        train_loss += current_loss
        
        train_loop.set_description(f'Epoch {epoch}/{num_epochs}')
        train_loop.set_postfix(loss=current_loss)

    train_loss /= n

    return train_loss


def validation_step(val_dl: data.DataLoader, model: AudiGest, loss_fn: nn.MSELoss,
                    device: torch.device, epoch: int, num_epochs: int) -> float:
    model.eval()

    n = len(val_dl.dataset)
    val_loss = 0.0

    hidden = None

    with torch.no_grad():
        val_loop = tqdm(enumerate(val_dl), total=len(val_dl), leave=False)
        for _, (melspec, mfcc, target, base_target) in val_loop:
            melspec = melspec.unsqueeze(1)
            mfcc = mfcc.permute(0, 2, 1)

            melspec = melspec.to(device)
            mfcc = mfcc.to(device)
            target = target.to(device)
            base_target = base_target.to(device)

            reconstructed, hidden = model(melspec, mfcc, base_target, hidden)
            loss = loss_fn(reconstructed, target)
            
            current_loss = loss.item() * val_dl.batch_size

            val_loss += current_loss
            
            val_loop.set_description(f'Epoch {epoch}/{num_epochs}')
            val_loop.set_postfix(loss=current_loss)

    val_loss /= n

    return val_loss


def train_model(train_dl: data.DataLoader, val_dl: data.DataLoader, model: AudiGest,
                optimizer: torch.optim, scheduler: torch.optim.lr_scheduler.ExponentialLR,
                train_hist: list[float], val_hist: list[float],
                device: torch.device, last_epoch: int, num_epochs: int) -> dict:

    train_loss_history = train_hist if train_hist is not None else []
    val_loss_history = val_hist if val_hist is not None else []
    loss_fn = nn.MSELoss()

    model = model.to(device)

    if last_epoch + 1 > 0:
        print('Resuming training')

    for epoch in range(last_epoch + 1, num_epochs + 1):
        s = time.time()
        train_loss = train_step(train_dl, model, loss_fn, optimizer, device, epoch, num_epochs)
        val_loss = validation_step(val_dl, model, loss_fn, device, epoch, num_epochs)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f'epoch {epoch}/{num_epochs}: {(time.time() - s) / 60}')
        print('train loss:', train_loss)
        print('val loss:', val_loss)

        scheduler.step()

        if epoch % 5 == 0:
            model.save(epoch, optimizer, scheduler, train_loss_history, val_loss_history)

    model_dict = {
        'train_history': train_loss_history,
        'val_history': val_loss_history
    }
    return model_dict


def main():
    config = get_config()
    set_seed(42, False)

    train_data = MEADDataset(partition='train', config=config)
    val_data = MEADDataset(partition='val', config=config)

    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    decay_rate = config['training']['decay_rate']
    epochs = config['training']['epochs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    train_dl = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dl = data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    model = AudiGest(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay_rate)

    last_epoch = get_last_epoch()

    optim_st, scheduler_st, train_hist, val_hist = model.load(last_epoch)
    if optim_st is not None:
        optimizer.load_state_dict(optim_st)
    
    if scheduler_st is not None:
        scheduler.load_state_dict(scheduler_st)
    
    model_dict = train_model(train_dl, val_dl, model, optimizer, scheduler, train_hist, val_hist, device, last_epoch, epochs)
    plot_loss(model_dict, 'AudiGest', save=True, test=True)


if __name__ == '__main__':
    main()
