import torch
from itertools import cycle
from torch.utils.data import random_split
from next_sparseconvnet.utils.data_loaders import DataGen, collatefn, LabelType, worker_init_fn

def load_data(train_data_path, train_data_domain_path, label_type, nevents_train, augmentation, seglabel_name, feature_name, val_split, train_batch_size, num_workers, pin_mem):
    full_mc_gen = DataGen(train_data_path, label_type, nevents = nevents_train, augmentation = augmentation, seglabel_name = seglabel_name, feature_name = feature_name)
    n_total = len(full_mc_gen)
    n_val   = int(val_split * n_total)
    n_train = n_total - n_val

    train_mc_gen, valid_mc_gen = random_split(full_mc_gen, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    loader_train_mc = torch.utils.data.DataLoader(train_mc_gen,
                                            batch_size = train_batch_size,
                                            shuffle = True,
                                            num_workers = num_workers,
                                            collate_fn = collatefn,
                                            drop_last = True,
                                            pin_memory = pin_mem, 
                                            persistent_workers = True,
                                            worker_init_fn = worker_init_fn)
    loader_valid_mc = torch.utils.data.DataLoader(valid_mc_gen,
                                            batch_size = train_batch_size,
                                            shuffle = True,
                                            num_workers = num_workers,
                                            collate_fn = collatefn,
                                            drop_last = True,
                                            pin_memory = pin_mem, 
                                            persistent_workers = True,
                                            worker_init_fn = worker_init_fn)

    print('Loaded MC events')

    full_dt_gen = DataGen(train_data_domain_path, label_type, nevents = nevents_train, augmentation = augmentation, seglabel_name = seglabel_name, feature_name = feature_name)
    n_total = len(full_dt_gen)
    n_val   = int(val_split * n_total)
    n_train = n_total - n_val

    train_dt_gen, valid_dt_gen = random_split(full_dt_gen, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    loader_train_dt = torch.utils.data.DataLoader(train_dt_gen,
                                                        batch_size = train_batch_size,
                                                        shuffle = True,
                                                        num_workers = num_workers,
                                                        collate_fn = collatefn,
                                                        drop_last = True,
                                                        pin_memory = pin_mem, 
                                                        persistent_workers = True,
                                                        worker_init_fn = worker_init_fn)
    loader_valid_dt = torch.utils.data.DataLoader(valid_dt_gen,
                                                        batch_size = train_batch_size,
                                                        shuffle = True,
                                                        num_workers = num_workers,
                                                        collate_fn = collatefn,
                                                        drop_last = True,
                                                        pin_memory = pin_mem, 
                                                        persistent_workers = True,
                                                        worker_init_fn = worker_init_fn)
    print('Loaded data events')
    return loader_train_mc, loader_valid_mc, loader_train_dt, loader_valid_dt


def train_domain(
    net,
    domain_clf,
    loader_mc,
    loader_dt,
    optimizer,
    criterion,
    device
):
    net.eval()                 # feature extractor frozen
    domain_clf.train()

    loss_epoch = 0.0
    acc_mc_epoch = 0.0
    acc_dt_epoch = 0.0
    n_batches = 0

    for (coord_mc, ener_mc, _, evt_mc), \
        (coord_dt, ener_dt, _, evt_dt) in zip(loader_mc, cycle(loader_dt)):

        bs_mc = len(evt_mc)
        bs_dt = len(evt_dt)

        ener_mc = ener_mc.to(device)
        ener_dt = ener_dt.to(device)

        optimizer.zero_grad()

        # --------------------------------
        # Feature extraction (frozen)
        # --------------------------------
        with torch.no_grad():
            feat_mc = net.feature_extractor((coord_mc, ener_mc, bs_mc))
            feat_dt = net.feature_extractor((coord_dt, ener_dt, bs_dt))

        # --------------------------------
        # Domain predictions
        # --------------------------------
        out_mc = domain_clf(feat_mc)
        out_dt = domain_clf(feat_dt)

        dom_mc = torch.zeros((bs_mc, 1), device=device)
        dom_dt = torch.ones((bs_dt, 1), device=device)

        loss_mc = criterion(out_mc, dom_mc)
        loss_dt = criterion(out_dt, dom_dt)
        loss = 0.5 * (loss_mc + loss_dt)

        loss.backward()
        optimizer.step()

        # --------------------------------
        # Metrics
        # --------------------------------
        with torch.no_grad():
            acc_mc = ((out_mc > 0.5).float() == 0).float().mean().item()
            acc_dt = ((out_dt > 0.5).float() == 1).float().mean().item()

        loss_epoch += loss.item()
        acc_mc_epoch += acc_mc
        acc_dt_epoch += acc_dt
        n_batches += 1

    return (
        loss_epoch / n_batches,
        acc_mc_epoch / n_batches,
        acc_dt_epoch / n_batches
    )


@torch.no_grad()
def valid_domain(net, domain_clf, loader_mc, loader_dt, criterion, device):
    net.eval()
    domain_clf.eval()

    loss, acc_mc, acc_dt, n = 0, 0, 0, 0

    for (coord_mc, ener_mc, _, evt_mc), (coord_dt, ener_dt, _, evt_dt) in zip(loader_mc, cycle(loader_dt)):
        bs_mc, bs_dt = len(evt_mc), len(evt_dt)

        ener_mc = ener_mc.to(device)
        ener_dt = ener_dt.to(device)

        feat_mc = net.feature_extractor((coord_mc, ener_mc, bs_mc))
        feat_dt = net.feature_extractor((coord_dt, ener_dt, bs_dt))

        out_mc = domain_clf(feat_mc)
        out_dt = domain_clf(feat_dt)

        loss += 0.5 * (
            criterion(out_mc, torch.zeros(bs_mc,1,device=device)) +
            criterion(out_dt, torch.ones(bs_dt,1,device=device))
        ).item()

        acc_mc += ((out_mc > 0.5).float() == 0).float().mean().item()
        acc_dt += ((out_dt > 0.5).float() == 1).float().mean().item()
        n += 1

    return loss/n, acc_mc/n, acc_dt/n



def train_label(
    net,
    domain_clf,
    loader_mc,
    optimizer,
    criterion,
    device,
    w_min=0.1,
    w_max=10.0
):
    net.label_classifier.train()
    net.feature_extractor.eval()
    domain_clf.eval()

    wloss_epoch = 0.0
    loss_epoch = 0.0
    acc_epoch = 0.0
    n_batches = 0
    
    for coord_mc, ener_mc, label_mc, evt_mc in loader_mc:

        bs = len(evt_mc)
        ener_mc = ener_mc.to(device)
        label_mc = label_mc.to(device)

        optimizer.zero_grad()

        # --------------------------------
        # Frozen feature space
        # --------------------------------
        with torch.no_grad():
            feat = net.feature_extractor((coord_mc, ener_mc, bs))
            p_data = domain_clf(feat)
            weights = p_data / (1.0 - p_data + 1e-6)
            weights = torch.clamp(weights, w_min, w_max)
        
        # collect weights for histogram
        if n_batches == 0:  # first batch of the epoch
            epoch_weights = weights.detach().cpu()

        # --------------------------------
        # Label prediction
        # --------------------------------
        logits = net.label_classifier(feat)

        loss_evt = criterion(logits, label_mc)
        weighted_loss = (weights.squeeze() * loss_evt).mean()

        weighted_loss.backward()
        optimizer.step()

        # --------------------------------
        # Metrics
        # --------------------------------
        with torch.no_grad():
            acc = (logits.argmax(1) == label_mc).float().mean().item()

        wloss_epoch += weighted_loss.item()
        loss_epoch += loss_evt.mean().item()
        acc_epoch += acc
        n_batches += 1

    return (
        wloss_epoch / n_batches,
        loss_epoch / n_batches,
        acc_epoch / n_batches,
        epoch_weights
    )


@torch.no_grad()
def valid_label(net, domain_clf, loader, criterion, device):
    net.eval()
    domain_clf.eval()

    total_wloss, total_loss, total_acc, n = 0.0, 0.0, 0.0, 0

    for batch_id, (coord, ener, label, evt) in enumerate(loader):
        bs = len(evt)
        ener = ener.to(device)
        label = label.to(device)

        feat = net.feature_extractor((coord, ener, bs))
        p_data = domain_clf(feat)
        w = torch.clamp(p_data / (1 - p_data + 1e-6), 0.1, 10.0)

        if batch_id == 0:
            epoch_weights = w.detach().cpu()

        logits = net.label_classifier(feat)
        loss_evt = criterion(logits, label)
        weighted_loss = (w.squeeze() * loss_evt).mean()

        total_wloss += weighted_loss.item()
        total_loss += loss_evt.mean().item()
        total_acc += (logits.argmax(1) == label).float().mean().item()
        n += 1

    return total_wloss/n, total_loss/n, total_acc/n, epoch_weights
