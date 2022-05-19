from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model.insitu import Generator
from model.insitu import Discriminator
from model.vgg19 import VGG19

from data.deep_water import DeepWaterDataset
from pathlib import Path
from option import basic_parser,add_train_args
import numpy as np

from eval.metric import compute_metric
import concurrent.futures as furtures
from datetime import timedelta
import time

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

def main(args,time_f):
    print(args)

    # select device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # data loader
    train_dataset = DeepWaterDataset(args.data,'train.pkl')

    test_dataset = DeepWaterDataset(args.data,'test.pkl')

    kwargs = {"num_workers": args.num_workers, "pin_memory": True} if args.cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True,drop_last=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=True,drop_last=True, **kwargs)

    g_model = Generator(dsp=args.dsp,dspe=args.dspe,dvp=args.dvp,dvpe=args.dvpe,ch=args.ch,hidden_nums=args.hidden_nums)
    g_model.apply(weights_init)
    if args.data_parallel and torch.cuda.device_count() > 1:
        g_model = nn.DataParallel(g_model)
    g_model.to(device)

    d_model = Discriminator(dsp=args.dsp,dspe=args.dspe,dvp=args.dvp,dvpe=args.dvpe,ch=args.ch)
    d_model.apply(weights_init)
    d_model = add_sn(d_model)
    if args.data_parallel and torch.cuda.device_count() > 1:
        d_model = nn.DataParallel(d_model)
    d_model.to(device)

    if args.perc_loss != None:
        norm_mean = torch.tensor([.485, .456, .406]).view(-1, 1, 1).to(device)
        norm_std = torch.tensor([.229, .224, .225]).view(-1, 1, 1).to(device)
        vgg = VGG19(args.perc_loss).eval()
        if args.data_parallel and torch.cuda.device_count() > 1:
            vgg = nn.DataParallel(vgg)
        vgg.to(device)

    train_losses, test_losses = [], []
    d_losses, g_losses = [], []
    mse_criterion = nn.MSELoss()

    g_optimizer = optim.Adam(g_model.parameters(), lr=args.lr,
                           betas=(args.beta1, args.beta2))
    d_optimizer = optim.Adam(d_model.parameters(), lr=args.d_lr,
                            betas=(args.beta1, args.beta2))

    # load checkpoint
    if args.checkpoint and Path(args.checkpoint).is_file():
        print(f"=> loading checkpoint {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
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

    t_start = time.time()
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        g_model.train()
        d_model.train()
        train_loss = 0.
        
        for i, sample in enumerate(train_loader):
            images = sample['img'].to(device)
            
            sparams = sample['sparams'].to(device)
            vparams = sample['vparams'].to(device)

            g_optimizer.zero_grad()
            fake_images = g_model(sparams,vparams)

            # update discriminator
            d_optimizer.zero_grad()
            
            decision = d_model(sparams,vparams,images)
            d_loss_real = torch.mean(F.relu(1. - decision))

            fake_decision = d_model(sparams,vparams, fake_images.detach())
            d_loss_fake = torch.mean(F.relu(1. + fake_decision))

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()

            d_optimizer.step()

            # update generator
            loss = 0.
            g_optimizer.zero_grad()

            fake_decision = d_model(sparams,vparams, fake_images)
            g_loss = - args.adv_loss_weight * torch.mean(fake_decision)
            
            if args.perc_loss:
                images = ( (images +1.) * .5 - norm_mean) / norm_std
                fake_images = ( (fake_images +1. )*.5 - norm_mean) / norm_std

                features = vgg(images)
                fake_features = vgg(fake_images)
                perc_loss = mse_criterion(features,fake_features)
            
            loss += perc_loss + g_loss
            
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
        
        test_loss = 0.
        with torch.no_grad():
            for i, sample in enumerate(test_loader):
                images = sample['img'].to(device)
                sparams = sample['sparams'].to(device)
                vparams = sample['vparams'].to(device)

                fake_images = g_model(sparams,vparams)
                test_loss += mse_criterion(images, fake_images).item() * len(sparams)

                if i == 0 and epoch % args.check_every == 0:
                    n = min(len(sparams), 8)
                    comparison = torch.cat(
                        [images[:n], fake_images.view(len(sparams), 3, 256, 256)[:n]])
                    save_image(((comparison.cpu() + 1.) * .5),
                                Path(args.storage,f'{epoch}.png'), nrow=n)

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
    t_end = time.time()
    # save g_model alone
    torch.save(g_model.state_dict(),Path(args.storage,f'model_{epoch}_{train_losses[-1]:.6f}_{test_losses[-1]:.6f}.pth'))

    # write training time
    time_f.write(f'training time:{timedelta(seconds=t_end-t_start)}\n')

    # write forward time
    with torch.no_grad():
        t_start = time.time()
        g_model(sparams,vparams)
        t_end = time.time()

    time_f.write(f'forward time:{timedelta(seconds=t_end-t_start)}\n')


    if args.metric:
        truth_path = Path(args.storage,'truth_imgs')
        gen_path = Path(args.storage,'gen_imgs')
        truth_path.mkdir(exist_ok=True)
        gen_path.mkdir(exist_ok=True)

        with torch.no_grad():
            with furtures.ThreadPoolExecutor() as executor:
                for sample in test_loader:
                    images = ((sample['img'] + 1.) * .5).cpu()

                    sparams = sample['sparams'].to(device)
                    vparams = sample['vparams'].to(device)

                    fake_images = ((g_model(sparams,vparams) + 1.) * .5).cpu()

                    sparams = sparams.tolist()
                    vparams = vparams.tolist()
                    
                    str_fn = lambda x: f'{x:.6f}'
                    
                    for i in range(len(sparams)):
                        i_truth = images[i]
                        i_fake = fake_images[i]

                        i_sp = sparams[i]
                        i_vp = vparams[i]

                        img_name = f'{"_".join(map(str_fn,i_sp))}_{"_".join(map(str_fn,i_vp))}.png'
                        executor.submit(save_image(i_truth,truth_path / img_name))
                        executor.submit(save_image(i_fake,gen_path / img_name))


if __name__ == "__main__":
    parser = basic_parser()
    parser = add_train_args(parser)
    args = parser.parse_args()
    p_storage = Path(args.stroage)
    p_storage.mkdir(exist_ok=True)

    with open(p_storage/'time.txt','w') as time_f:
        main(args,time_f)

    if args.metric:
        compute_metric(args)