import os

def mkdir(path: str):
    if not os.path.isdir(path): os.makedirs(path)

root_dir = '/home/seungchan/Desktop/Projs/graduation-project'
data_root = os.path.join(root_dir, 'dataset/2022_02_all')
pkl_file = 'features.pkl'
res_path = os.path.join(root_dir, 'result')

mkdir(res_path)