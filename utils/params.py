import os

def mkdir(path: str):
    if not os.path.isdir(path): os.makedirs(path)

root_dir = '/home/seungchan/Desktop/Projs/graduation-project'
data_root = os.path.join(root_dir, 'dataset/2022_02_all')
pkl_file = 'features.pkl'
res_path = os.path.join(root_dir, 'result')
mkdir(res_path)

categories = [
    'A funny clips, animals, comics or strip cartoons',
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 
    'It is about political issues or politicians in governments',
    'It is about social issues, pandemics or vaccine for COVID-19 virus or photos of family',
    'It is about monetary economy, stocks or bitcoins', 
    'A photo of fashions, clothings, wearings or shoes',
    'A photo of musicians, entertainers, celebrity, music bands or idols', 
    'A photo of sports or athletes', 
    'It is about dishes, foods, eating things', 
    # 'A photo of pet animals or companion animals',
    'A photo of self-taken (portrait) photographs',
    # 'A photo or picture of strip cartoons, comics, sketches or drawings', 
    'It is about movies or cinema', 
    'A photo that has taken in video games',
    'A photo of landscape or natural scenery',
]

catname = [
    'meme',
    'etc',
    'politic',
    'social',
    'economy',
    'fashion',
    'entertain',
    'sport',
    'food',
    'animal',
    'selfi',
    'cartoon',
    'movie',
    'game',
    'landscape'
]