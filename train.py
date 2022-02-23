from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model.insitu import Generator
from model.insitu import Discriminator

from data.data import IsabelDataset
from pathlib import Path
from option import basic_parser,add_train_args

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def add_sn(m):
    for name, c in m.named_children():
        m.add_module(name, add_sn(c))
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        return nn.utils.spectral_norm(m, eps=1e-4)
    else:
        return m

def main(args):
    print(args)

    # select device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # data loader
    train_dataset = IsabelDataset(args.data,'train.txt')

    test_dataset = IsabelDataset(args.data,'test.txt')

    kwargs = {"num_workers": 2, "pin_memory": True} if args.cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True,drop_last=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=True,drop_last=True, **kwargs)

    g_model = Generator(dsp=args.dsp,dspe=args.dspe,ch=args.ch)
    g_model.apply(weights_init)
    if args.data_parallel and torch.cuda.device_count() > 1:
        g_model = nn.DataParallel(g_model)
    g_model.to(device)

    d_model = Discriminator(dsp=args.dsp,dspe=args.dspe,ch=args.ch)
    d_model.apply(weights_init)
    d_model = add_sn(d_model)
    if args.data_parallel and torch.cuda.device_count() > 1:
        d_model = nn.DataParallel(d_model)
    d_model.to(device)

    train_losses, test_losses = [], []
    d_losses, g_losses = [], []

    g_optimizer = optim.Adam(g_model.parameters(), lr=args.lr,
                           betas=(args.beta1, args.beta2))
    d_optimizer = optim.Adam(d_model.parameters(), lr=args.d_lr,
                            betas=(args.beta1, args.beta2))

    # load checkpoint
    if args.checkpoint and Path(args.checkpoint).is_file():
        print(f"=> loading checkpoint {args.checkpoint}")
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint["epoch"]

        g_model.load_state_dict(checkpoint["g_model_state_dict"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        d_model.load_state_dict(checkpoint["d_model_state_dict"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        
        d_losses = checkpoint["d_losses"]
        g_losses = checkpoint["g_losses"]
        train_losses = checkpoint["train_losses"]
        test_losses = checkpoint["test_losses"]

        print(f"=> loaded checkpoint {args.checkpoint} (epoch {checkpoint['epoch']})")

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        g_model.train()
        d_model.train()
        train_loss = 0.
        
        for i, sample in enumerate(train_loader):
            field = sample["f"].to(device)
            sparams = sample["sp"].to(device)
            g_optimizer.zero_grad()
            fake_field = g_model(sparams)

            # update discriminator
            d_optimizer.zero_grad()
            
            decision = d_model(sparams,field)
            d_loss_real = torch.mean(F.relu(1. - decision))

            fake_decision = d_model(sparams, fake_field.detach())
            d_loss_fake = torch.mean(F.relu(1. + fake_decision))

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()

            d_optimizer.step()

            # update generator
            loss = 0.
            g_optimizer.zero_grad()
            fake_decision = d_model(sparams, fake_field)

            g_loss = -torch.mean(fake_decision)
            loss += g_loss
            
            loss.backward()
            g_optimizer.step()
            train_loss += loss.item() * len(sparams)

            # log training status
            if i % args.log_every == 0:
                print(f"Train Epoch: {epoch} [{i*len(sparams)}/{len(train_loader.dataset)}"
                      f"({100. * i / len(train_loader):.0f}%)]")
                print(f"DLoss: {d_loss.item():.6f}, GLoss: {g_loss.item():.6f}")
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                train_losses.append(train_loss/len(train_loader.dataset))

        print(f"====> Epoch: {epoch} Average loss: {train_loss/len(train_loader.dataset):.6f}")

        # testing...
        g_model.eval()
        d_model.eval()
        
        mse_criterion = nn.MSELoss()
        test_loss = 0.
        with torch.no_grad():
            for i, sample in enumerate(test_loader):
                field = sample["f"].to(device)
                sparams = sample["sp"].to(device)
                fake_field = g_model(sparams)
                test_loss += mse_criterion(field, fake_field).item() * len(sparams)

        test_losses.append(test_loss / len(test_loader.dataset))
        print(f"====> Epoch: {epoch} Test set loss: {test_losses[-1]:.6f}")

        # saving...
        if epoch % args.check_every == 0:
            print(f"=> saving checkpoint at epoch {epoch}")
            torch.save({"epoch": epoch + 1,
                        "g_model_state_dict": g_model.state_dict(),
                        "g_optimizer_state_dict": g_optimizer.state_dict(),
                        "d_model_state_dict": d_model.state_dict(),
                        "d_optimizer_state_dict": d_optimizer.state_dict(),
                        "d_losses": d_losses,
                        "g_losses": g_losses,
                        "train_losses": train_losses,
                        "test_losses": test_losses},
                        Path(args.storage,f'model_{epoch}_{train_losses[-1]:.6f}_{test_losses[-1]:.6f}.pth.tar'))

if __name__ == "__main__":
    parser = basic_parser()
    parser = add_train_args(parser)
    main(parser.parse_args())