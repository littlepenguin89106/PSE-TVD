def normalize(data,d_min,d_max):
    n_data = (data - d_min) / (d_max - d_min)
    return n_data * 2 - 1
    # [-1,1]

def denormalize(data,d_min,d_max):
    n_data = (data + 1) / 2
    return n_data * (d_max - d_min) + d_min