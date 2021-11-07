import numpy as np
import os
import random
import time
import torch.optim
import torch.nn.functional as F

from config_creator import get_config
from utils.model.losses import *
from torch.utils import data
from tqdm import tqdm
from utils.model.MEADdataset import MEADDataset
from utils.model.SequenceRegressor import SequenceRegressor
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


def train_step(config, train_dl, model, loss_fn, optimizer, device, epoch, num_epochs):
    model.train()

    optimizer.zero_grad()

    emotion_count = len(config['emotions'])
    subject_count = len(config['files']['train']['subjects'])
    n = len(train_dl)
    train_loss = 0.0

    train_loop = tqdm(enumerate(train_dl), total=len(train_dl), leave=False)
    for _, (subject_idx, emotion_idx, feature, target, base_target) in train_loop:
        subject_idx = subject_idx.squeeze(dim=0)
        subject_idx = F.one_hot(subject_idx, subject_count)

        emotion_idx = emotion_idx.squeeze(dim=0)
        emotion_idx = F.one_hot(emotion_idx, emotion_count)

        feature = feature.permute(0, 2, 1)

        subject_idx = subject_idx.to(device)
        emotion_idx = emotion_idx.to(device)
        feature = feature.to(device)
        target = target.to(device)
        base_target = base_target.to(device)

        reconstructed = model(feature, emotion_idx, subject_idx, base_target)
        rec_loss = loss_fn(reconstructed, target)
        loss = rec_loss

        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        train_loss += current_loss

        train_loop.set_description(f'Epoch {epoch}/{num_epochs}: training')
        train_loop.set_postfix(loss=current_loss)

    train_loss /= n

    return train_loss


def validation_step(config, val_dl, model, loss_fn, device, epoch, num_epochs):
    model.eval()

    emotion_count = len(config['emotions'])
    subject_count = len(config['files']['train']['subjects'])
    n = len(val_dl)
    val_loss = 0.0

    with torch.no_grad():
        val_loop = tqdm(enumerate(val_dl), total=len(val_dl), leave=False)
        for _, (subject_idx, emotion_idx, feature, target, base_target) in val_loop:
            subject_idx = subject_idx.squeeze(dim=0)
            subject_idx = F.one_hot(subject_idx, subject_count)

            emotion_idx = emotion_idx.squeeze(dim=0)
            emotion_idx = F.one_hot(emotion_idx, emotion_count)

            feature = feature.permute(0, 2, 1)

            subject_idx = subject_idx.to(device)
            emotion_idx = emotion_idx.to(device)
            feature = feature.to(device)
            target = target.to(device)
            base_target = base_target.to(device)

            reconstructed = model(feature, emotion_idx, subject_idx, base_target)
            rec_loss = loss_fn(reconstructed, target)
            loss = rec_loss

            current_loss = loss.item()

            val_loss += current_loss

            val_loop.set_description(f'Epoch {epoch}/{num_epochs}: validation')
            val_loop.set_postfix(loss=current_loss)

    val_loss /= n

    return val_loss


def train_model(config, train_dl, val_dl, model, optimizer, scheduler, train_hist, val_hist,
                device, last_epoch, num_epochs):
    train_loss_history = train_hist if train_hist is not None else []
    val_loss_history = val_hist if val_hist is not None else []

    rec_loss = nn.L1Loss()

    if last_epoch > 0:
        print('Resuming training')

    for epoch in range(last_epoch + 1, num_epochs + 1):
        print(f'epoch {epoch}/{num_epochs}:')
        s = time.time()
        train_loss = train_step(config, train_dl, model, rec_loss, optimizer, device, epoch, num_epochs)
        print('train loss:', train_loss)
        val_loss = validation_step(config, val_dl, model, rec_loss, device, epoch, num_epochs)
        print('val loss:', val_loss)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        epoch_time = time.time() - s
        print(f'time: {int(epoch_time / 60):02d}:{int(epoch_time % 60):02d} minutes')

        if scheduler is not None:
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

    train_data = MEADDataset(partition='train', config=config, use_centered=True)
    val_data = MEADDataset(partition='val', config=config, use_centered=True)

    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    decay_rate = config['training']['decay_rate']
    epochs = config['training']['epochs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    train_dl = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
    val_dl = data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)

    # subject_idx, emotion_idx, feature, target, base_target = next(iter(val_dl))
    # print('subject:', subject_idx.shape)
    # print('emotion:', emotion_idx.shape)
    # print('feature:', feature.shape)
    # print('target:', target.shape)
    # print('template:', base_target.shape)

    model = SequenceRegressor(config, device)
    model = model.to(device)
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
    plot_loss(model_dict, 'AudiGest3', save=True, test=True)


if __name__ == '__main__':
    main()
