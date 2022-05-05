from skimage import color
import numpy as np
import pyemd
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path
import torch
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from pytorch_fid.fid_score import calculate_fid_given_paths
from skimage import io
import concurrent.futures as futures

def compute_emd(im1, im2, cost_mat, l_bins=8,a_bins=12,b_bins=12):
    lab_im1 = color.rgb2lab(im1)
    lab_im1 = lab_im1.reshape((lab_im1.shape[0]*lab_im1.shape[1],lab_im1.shape[2]))
    lab_hist_1,_ = np.histogramdd(lab_im1, bins=(l_bins,a_bins,b_bins), range=[[0.0,100.0], [-86.185,98.254], [-107.863,94.482]], normed=False)

    lab_im2 = color.rgb2lab(im2)
    lab_im2 = lab_im2.reshape((lab_im2.shape[0]*lab_im2.shape[1],lab_im2.shape[2]))
    lab_hist_2,_ = np.histogramdd(lab_im2, bins=(l_bins,a_bins,b_bins), range=[[0.0,100.0], [-86.185,98.254], [-107.863,94.482]], normed=False)

    n_bins = l_bins*a_bins*b_bins
    lab_hist_1 = lab_hist_1.reshape((n_bins))
    lab_hist_2 = lab_hist_2.reshape((n_bins))
    img_res = lab_im1.shape[0]
    lab_hist_1 /= img_res
    lab_hist_2 /= img_res
    return pyemd.emd(lab_hist_1,lab_hist_2,cost_mat)

def compute_emd_cost_mat(l_bins=8,a_bins=12,b_bins=12):
    n_bins = l_bins*a_bins*b_bins
    index_mat = np.zeros((l_bins,a_bins,b_bins,3))
    for idx in range(l_bins):
        for jdx in range(a_bins):
            for kdx in range(b_bins):
                index_mat[idx,jdx,kdx] = np.array([idx,jdx,kdx])
    index_mat = index_mat.reshape(n_bins,3)
    all_dists = euclidean_distances(index_mat,index_mat)
    return all_dists / np.max(all_dists)

def compute_single_metric(predict,truth,emd_cost_mat):
    return {'psnr': peak_signal_noise_ratio(predict,truth),
            'ssim': structural_similarity(predict,truth),
            'emd': compute_emd(predict,truth,emd_cost_mat)}

def compute_metric(args):
    storage = Path(args.storage)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    truth_path = storage / 'truth_imgs'
    gen_path = storage / 'gen_imgs'

    psnr = 0.
    ssim = 0.
    emd = 0.
    count = 0
    emd_cost_mat = compute_emd_cost_mat()
    
    with futures.ProcessPoolExecutor(args.num_workers) as executor:
        metric_futures = []

        for p in truth_path.iterdir():
            img_name = p.name
            image = io.imread(p)
            fake_image = io.imread(gen_path/img_name)
            future = executor.submit(compute_single_metric,fake_image,image,emd_cost_mat)
            metric_futures.append(future)

        for future in futures.as_completed(metric_futures):
            metric_result = future.result()
            psnr += metric_result['psnr']
            ssim += metric_result['ssim']
            emd += metric_result['emd']
            count += 1

    fid = calculate_fid_given_paths([truth_path.as_posix(),gen_path.as_posix()],args.batch_size,device,2048,args.num_workers)
    
    with open(Path(args.storage,'metric.txt'),'w') as f:
        f.write(f'psnr: {psnr/count}\n')
        f.write(f'ssim: {ssim/count}\n')
        f.write(f'emd: {emd/count}\n')
        f.write(f'fid: {fid}\n')
