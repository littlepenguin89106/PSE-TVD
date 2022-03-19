import matplotlib.pyplot as plt
import torch
import fire

def plot(save_path,attr):
    checkpoint = torch.load(save_path)
    losses = checkpoint[attr]
    x = [i for i in range(len(losses))]
    plt.plot(x,losses)
    plt.show()

if __name__ == '__main__':
    fire.Fire()