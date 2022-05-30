from deepwater.dataset import DeepWaterDataset
from nyx.dataset import NyxDataset

def get_dataset(args,data_list):
    if args.dataset.upper()=='NYX':
        return NyxDataset(args.data,data_list)
    else:
        return DeepWaterDataset(args.data,data_list)