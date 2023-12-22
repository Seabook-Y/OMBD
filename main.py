import datetime
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import loss as utl
from misc import init as cfg
from Dataload.PDMBDataset import PDMBDataSet
from loss.evaluate import OMBD_evaluate
from model.OMBDModel import OMBD_lstra, OMBD_tie
import torch.nn.functional as F
import numpy as np
from misc.utils import backup_code

def train_one_epoch(model_lstra,
                    model_tie,
                    criterion,
                    data_loader, optimizer,
                    device, max_norm):
    model_tie.train()
    model_lstra.train()
    criterion.train()
    losses = 0
    i = 0
    for camera_inputs, enc_target in data_loader:
        inputs = camera_inputs.to(device)
        enc_target = enc_target.argmax(dim=-1)
        target = enc_target.to(device=device)

        # todo
        optimizer.zero_grad()

        enc_score_tie = model_tie(inputs[:, -1:, :], device)
        loss_tie = criterion(enc_score_tie, target[:, -1:], 'CE')

        enc_score_lstra = model_lstra(inputs)
        loss_lstra = criterion(enc_score_lstra[:, :, -1:], target[:, -1:], 'CE')

        loss_EMD = criterion(enc_score_lstra[:, :9, -1:], enc_score_tie, 'EMD')
        loss = loss_tie + loss_lstra + 0.6*loss_EMD
        # loss = loss_tie + loss_lstra

        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_tie.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(model_lstra.parameters(), max_norm)

        optimizer.step()
        losses += loss
        i = i + 1
        print('\r train-------------------{:.4f}%'.format((i / 1027) * 100), end='')
    return losses / i, losses


def evaluate(model_lstra,
             model_tie,
             data_loader, device):
    model_tie.eval()
    model_lstra.eval()

    score_val_x = []
    target_val_x = []

    i = 0
    for camera_inputs, enc_target in data_loader:
        inputs = camera_inputs.to(device)
        target = enc_target.to(device)
        target_val = target[:, -1:, :9]

        with torch.no_grad():
            enc_score_lstra = model_lstra(inputs)
            enc_score_tie = model_tie(inputs[:, -1:, :], device)

        enc_score_tie = enc_score_tie.permute(0, 2, 1)
        enc_score_tie = enc_score_tie[:, :, :9]

        enc_score_lstra = enc_score_lstra.permute(0, 2, 1)
        enc_score_lstra = enc_score_lstra[:, -1:, :9]

        score_val = enc_score_tie * 0.3 + enc_score_lstra * 0.7
        score_val = F.softmax(score_val, dim=-1)

        score_val = score_val.contiguous().view(-1, 9).cpu().numpy()
        target_val = target_val.contiguous().view(-1, 9).cpu().numpy()

        score_val_x += list(score_val)
        target_val_x += list(target_val)
        i += 1
        print('\r train-------------------{:.4f}%'.format((i / 329) * 100), end='')
        # i += 1
    all_probs = np.asarray(score_val_x).T
    all_classes = np.asarray(target_val_x).T
    print(all_probs.shape, all_classes.shape)
    results = {'probs': all_probs, 'labels': all_classes}

    return results


def main(args):
    log_file = backup_code(args.exp_name)
    seed = args.seed + cfg.get_rank()
    cfg.set_seed(seed)

    device = torch.device('cuda:' + str(args.cuda_id))
    model_tie = OMBD_tie(args.input_size, args.numclass, device, args.KF)
    model_lstra = OMBD_lstra(args.input_size, args.numclass)

    model_lstra.apply(cfg.weight_init)
    model_lstra.to(device)
    model_tie.apply(cfg.weight_init)
    model_tie.to(device)

    criterion = utl.SetCriterion().to(device)
    optimizer = torch.optim.Adam([
        {"params": model_tie.parameters()},
        {"params": model_lstra.parameters()}],
        lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = PDMBDataSet(flag='train', args=args)
    dataset_val = PDMBDataSet(flag='test', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   pin_memory=True, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, pin_memory=True, num_workers=args.num_workers)

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, loss = train_one_epoch(
            model_lstra,
            model_tie,
            criterion, data_loader_train, optimizer, device, args.clip_max_norm)

        lr_scheduler.step()
        print('epoch:{}------loss:{}'.format(epoch, train_loss))

        test_stats = evaluate(
            model_lstra,
            model_tie,
            data_loader_val, device)
        print('---------------Calculation of the map-----------------')
        OMBD_evaluate(test_stats, epoch, args.command, log_file)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if '__main__' == __name__:
    args = cfg.parse_args()
    with open(args.dataset_file, 'r') as f:
        data_info = json.load(f)['PDMB']

    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']
    args.class_index = data_info['class_index']
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
