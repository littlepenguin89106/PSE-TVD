from skimage import io,metrics
from pathlib import Path
import fire

def compute_metric(truth_dir,gen_dir):
    truth_dir = Path(truth_dir)
    gen_dir = Path(gen_dir)

    psnr = 0.
    ssim = 0.
    count = 0
    for truth_img_path in truth_dir.iterdir():
        img_name = truth_img_path.name
        gen_img_path = gen_dir / img_name

        truth_img = io.imread(truth_img_path)
        gen_img = io.imread(gen_img_path)
        psnr += metrics.peak_signal_noise_ratio(truth_img,gen_img)
        ssim += metrics.structural_similarity(truth_img,gen_img,multichannel=True,data_range=255)

        count +=1

    print(psnr/count)
    print(ssim/count)

if __name__ == "__main__":
    fire.Fire()