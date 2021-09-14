import time
import torch
import torch.nn as nn

from config_creator import get_config
from torch.utils import data
from tqdm import tqdm
from utils.model.MEADdataset import MEADDataset
from utils.model.AudiGest import AudiGest
from utils.plotter import plot_loss


def split_data(dataset: MEADDataset, proportion: float = 0.8):
    n = len(dataset)
    tran_size = int(n * proportion)
    val_size = n - tran_size
    train_set, val_set = data.random_split(dataset, [tran_size, val_size])
    return train_set, val_set


def train_step(train_dl: data.DataLoader, model: AudiGest, loss_fn: nn.MSELoss,
               optimizer: torch.optim.Optimizer, device: torch.device,
               epoch: int, num_epochs: int) -> float:
    model.train()

    optimizer.zero_grad()

    n = len(train_dl.dataset)
    train_loss = 0.0

    train_loop = tqdm(enumerate(train_dl), total=len(train_dl), leave=False)
    # melspec, mfcc, target = next(iter(train_dl))
    for _, (melspec, mfcc, target) in train_loop:
        melspec = melspec.unsqueeze(1)
        mfcc = mfcc.permute(0, 2, 1)

        melspec = melspec.to(device)
        mfcc = mfcc.to(device)
        target = target.to(device)

        reconstructed = model(melspec, mfcc)
        loss = loss_fn(reconstructed, target)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        
        train_loss += loss.item()
        
        train_loop.set_description(f'Epoch {epoch + 1}/{num_epochs}')
        train_loop.set_postfix(loss=loss.item())

    train_loss /= n

    return train_loss


def validation_step(val_dl: data.DataLoader, model: AudiGest, loss_fn: nn.MSELoss,
                    device: torch.device, epoch: int, num_epochs: int) -> float:
    model.eval()

    n = len(val_dl.dataset)
    val_loss = 0.0

    with torch.no_grad():
        val_loop = tqdm(enumerate(val_dl), total=len(val_dl), leave=False)
        # melspec, mfcc, target = next(iter(val_dl))
        for _, (melspec, mfcc, target) in val_loop:
            melspec = melspec.unsqueeze(1)
            mfcc = mfcc.permute(0, 2, 1)

            melspec = melspec.to(device)
            mfcc = mfcc.to(device)
            target = target.to(device)

            reconstructed = model(melspec, mfcc)
            loss = loss_fn(reconstructed, target)
            
            val_loss += loss.item()
            
            val_loop.set_description(f'Epoch {epoch + 1}/{num_epochs}')
            val_loop.set_postfix(loss=loss.item())

    val_loss /= n

    return val_loss


def train_model(train_dl: data.DataLoader, val_dl: data.DataLoader, model: AudiGest,
                optimizer: torch.optim, scheduler: torch.optim.lr_scheduler.ExponentialLR,
                device: torch.device, num_epochs: int) -> dict:
    train_loss_history = []
    val_loss_history = []
    loss_fn = nn.MSELoss()

    model = model.to(device)

    for epoch in range(num_epochs):
        s = time.time()
        train_loss = train_step(train_dl, model, loss_fn, optimizer, device, epoch, num_epochs)
        val_loss = validation_step(val_dl, model, loss_fn, device, epoch, num_epochs)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f'epoch {epoch + 1}/{num_epochs}: {(time.time() - s) / 60}')
        print('train loss:', train_loss)
        print('val loss:', val_loss)

        scheduler.step()

    model_dict = {
        'train_history': train_loss_history,
        'val_history': val_loss_history
    }
    return model_dict


def main():
    config = get_config()

    train_data = MEADDataset(train=True, config=config)
    test_data = MEADDataset(train=False, config=config)

    train_set, val_set = split_data(train_data, proportion=0.8)

    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    decay_rate = config['training']['decay_rate']
    epochs = config['training']['epochs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    train_dl = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)
    val_dl = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=6, drop_last=True)

    test_dl = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=6, drop_last=True)

    model = AudiGest(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay_rate)
    
    model_dict = train_model(train_dl, val_dl, model, optimizer, scheduler, device, epochs)
    plot_loss(model_dict, 'AudiGest')


if __name__ == '__main__':
    main()
