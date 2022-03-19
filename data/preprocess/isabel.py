import numpy as np
import scipy.ndimage
import pathlib
import fire

def slice_and_reshape(src_path, dst_path):
    '''slicing and reshape'''

    for iter_path in pathlib.Path(src_path).iterdir():
        if iter_path.is_file():
            raw_data = np.fromfile(iter_path, dtype="float32")
            raw_data = raw_data.reshape(100, 500, 500)
            org_name = iter_path.stem
            org_str_time = int(org_name[2:])

            for height in range(100):
                # Resize the raw data from (500, 500) to (256, 256)
                resize_data = scipy.ndimage.interpolation.zoom(
                    raw_data[height, :, :], 256 / 500)

                # Write a file
                resize_path = pathlib.Path(dst_path,
                                           fr'pf{height}_{org_str_time}.bin')
                resize_data.tofile(resize_path)

def interpolate_z(skip, inter_num, path):
    '''
    interpolating with z axis

    Args:
        skip:
            missing timestamps need to skip
        inter_num:
            numbers of slices it creates
        path: 
            dataset path
    '''

    inter_range = 1 / (inter_num + 1)

    for t in range(1, 49):
        if t not in skip:
            for z1 in range(99):
                z3 = z1 + 1
                first = np.fromfile(pathlib.Path(path, f'pf{z1}_{t}.bin'),
                                    'float32')
                third = np.fromfile(pathlib.Path(path, f'pf{z3}_{t}.bin'),
                                    'float32')
                for i in range(1, inter_num + 1):
                    z2 = i * inter_range
                    inter_res = first * z2 + third * (1 - z2)
                    inter_res.tofile(pathlib.Path(path, f'pf{z1+z2}_{t}.bin'))


def interpolate_time(skip, inter_num, path):
    '''
    interpolating with timestamp

    Args:
        skip:
            missing timestamps need to skip
        inter_num:
            numbers of slices it creates
        path: 
            dataset path
    '''

    inter_range = 1 / (inter_num + 1)

    for t1 in range(1, 48):
        t3 = t1 + 1
        if t1 not in skip and t3 not in skip:
            for z in range(100):
                first = np.fromfile(pathlib.Path(path, f'pf{z}_{t1}.bin'),
                                    'float32')
                third = np.fromfile(pathlib.Path(path, f'pf{z}_{t3}.bin'),
                                    'float32')
                for i in range(1, inter_num + 1):
                    t2 = i * inter_range
                    inter_res = first * t2 + third * (1 - t2)
                    inter_res.tofile(pathlib.Path(path, f'pf{z}_{t1+t2}.bin'))

def split_dataset(skip, path):
    def write_z(fd, t):
        for sp in range(0, 100):
            fd.write(f'pf{sp}_{t}.bin {sp} {t}\n')

    with open(pathlib.Path(path, "train.txt"),'w') as trainf \
            , open(pathlib.Path(path, "test.txt"),'w') as testf:
        for t in range(1, 49):
            if t not in skip:
                if t % 6 == 0:
                    write_z(testf, t)
                else:
                    write_z(trainf, t)


if __name__ == '__main__':
    fire.Fire()