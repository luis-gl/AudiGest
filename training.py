import numpy as np
import os
import random
import time
import torch.nn.functional as F

from config_creator import get_config
from utils.model.losses import *
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


def train_step(config: dict, train_dl: data.DataLoader, model: AudiGest, loss_fn_dict: dict,
               optimizer: torch.optim.Optimizer, device: torch.device,
               epoch: int, num_epochs: int) -> float:
    model.train()

    optimizer.zero_grad()

    emotion_count = len(config['emotions'])
    consecutive_seqs = config['training']['consecutive_seqs']
    n = config['training']['batch_size'] * consecutive_seqs * len(train_dl)
    train_loss = 0.0

    hidden = None

    train_loop = tqdm(enumerate(train_dl), total=len(train_dl), leave=False)
    for _, (emotion_idx, mfcc, target, base_target) in train_loop:
        # melspec = melspec.reshape(-1, *melspec.shape[2:])
        # melspec = melspec.unsqueeze(1)
        emotion_idx = emotion_idx.reshape(-1)
        emotion_idx = F.one_hot(emotion_idx, emotion_count)
        mfcc = mfcc.reshape(-1, *mfcc.shape[2:])
        mfcc = mfcc.permute(0, 2, 1)
        target = target.reshape(-1, *target.shape[2:])
        base_target = base_target.reshape(-1, *base_target.shape[2:])

        # melspec = melspec.to(device)
        emotion_idx = emotion_idx.to(device)
        mfcc = mfcc.to(device)
        target = target.to(device)
        base_target = base_target.to(device)

        reconstructed, hidden = model(emotion_idx, mfcc, base_target, hidden)
        rec_loss = loss_fn_dict['rec'](reconstructed, target)
        vel_loss = loss_fn_dict['vel'](reconstructed, target)
        loss = rec_loss + vel_loss
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        
        current_loss = loss.item() * train_dl.batch_size

        train_loss += current_loss
        
        train_loop.set_description(f'Epoch {epoch}/{num_epochs}')
        train_loop.set_postfix(loss=current_loss)

    train_loss /= n

    return train_loss


def validation_step(config: dict, val_dl: data.DataLoader, model: AudiGest, loss_fn_dict: dict,
                    device: torch.device, epoch: int, num_epochs: int) -> float:
    model.eval()

    emotion_count = len(config['emotions'])
    n = config['training']['batch_size'] * config['training']['consecutive_seqs'] * len(val_dl)
    val_loss = 0.0

    hidden = None

    with torch.no_grad():
        val_loop = tqdm(enumerate(val_dl), total=len(val_dl), leave=False)
        for _, (emotion_idx, mfcc, target, base_target) in val_loop:
            # melspec = melspec.reshape(-1, *melspec.shape[2:])
            # melspec = melspec.unsqueeze(1)
            emotion_idx = emotion_idx.reshape(-1)
            emotion_idx = F.one_hot(emotion_idx, emotion_count)
            mfcc = mfcc.reshape(-1, *mfcc.shape[2:])
            mfcc = mfcc.permute(0, 2, 1)
            target = target.reshape(-1, *target.shape[2:])
            base_target = base_target.reshape(-1, *base_target.shape[2:])

            # melspec = melspec.to(device)
            emotion_idx = emotion_idx.to(device)
            mfcc = mfcc.to(device)
            target = target.to(device)
            base_target = base_target.to(device)

            reconstructed, hidden = model(emotion_idx, mfcc, base_target, hidden)
            rec_loss = loss_fn_dict['rec'](reconstructed, target)
            vel_loss = loss_fn_dict['vel'](reconstructed, target)
            loss = rec_loss + vel_loss
            
            current_loss = loss.item() * val_dl.batch_size

            val_loss += current_loss
            
            val_loop.set_description(f'Epoch {epoch}/{num_epochs}')
            val_loop.set_postfix(loss=current_loss)

    val_loss /= n

    return val_loss


def train_model(config: dict, train_dl: data.DataLoader, val_dl: data.DataLoader, model: AudiGest,
                optimizer: torch.optim, scheduler: torch.optim.lr_scheduler.ExponentialLR,
                train_hist: list[float], val_hist: list[float],
                device: torch.device, last_epoch: int, num_epochs: int) -> dict:

    train_loss_history = train_hist if train_hist is not None else []
    val_loss_history = val_hist if val_hist is not None else []

    rec_loss = nn.L1Loss()
    vel_loss = VelocityLoss(config, rec_loss)

    loss_fn_dict = {
        'rec': rec_loss,
        'vel': vel_loss
    }

    model = model.to(device)

    if last_epoch > 0:
        print('Resuming training')

    for epoch in range(last_epoch + 1, num_epochs + 1):
        s = time.time()
        train_loss = train_step(config, train_dl, model, loss_fn_dict, optimizer, device, epoch, num_epochs)
        val_loss = validation_step(config, val_dl, model, loss_fn_dict, device, epoch, num_epochs)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f'epoch {epoch}/{num_epochs}: {(time.time() - s) / 60:.4f} minutes')
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
    set_seed(42, True)

    train_data = MEADDataset(partition='train', config=config, use_rescaled=False, use_norm=True)
    val_data = MEADDataset(partition='val', config=config, use_rescaled=False, use_norm=True)

    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    decay_rate = config['training']['decay_rate']
    epochs = config['training']['epochs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    train_dl = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dl = data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    # a, b, c, d = next(iter(train_dl))
    # print(a.shape)
    # print(b.shape)
    # print(c.shape)
    # print(d.shape)

    model = AudiGest(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay_rate)

    last_epoch = get_last_epoch()

    optim_st, scheduler_st, train_hist, val_hist = model.load(last_epoch)
    if optim_st is not None:
        optimizer.load_state_dict(optim_st)

    if scheduler_st is not None:
        scheduler.load_state_dict(scheduler_st)

    model_dict = train_model(config, train_dl, val_dl, model, optimizer, scheduler, train_hist, val_hist,
                             device, last_epoch, epochs)
    plot_loss(model_dict, 'AudiGest2', save=True, test=True)


if __name__ == '__main__':
    main()
