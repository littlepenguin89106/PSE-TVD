import matplotlib.pyplot as plt
import numpy as np

def heatmap_from_file(src_path,dst_path,extent=None):
    data = np.fromfile(src_path)
    plt.imshow(data,extent=extent)
    plt.savefig(dst_path)

def heatmap_from_arr(arr,dst_path,extent=None):
    plt.imshow(arr,extent=extent)
    plt.savefig(dst_path)