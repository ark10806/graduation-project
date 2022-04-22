import os

def mkdir(path: str):
    if not os.path.isdir(path): os.makedirs(path)

topk = 10
root_dir = '/home/seungchan/Desktop/Projs/graduation-project'
# data_root = os.path.join(root_dir, 'dataset/2022_02_all')
data_root = os.path.join(root_dir, 'dataset/202104-202204_cchal')
feature_path = 'features_ko.pkl'
meme_feature_path = os.path.join(root_dir, 'result', 'meme.pkl')
res_path = os.path.join(root_dir, 'result')
scweet_path = os.path.join(root_dir, 'dataset/scweet')
papago_monitor_path = os.path.join(root_dir, 'utils/translator/papago_monitor')
mkdir(res_path)
mkdir(papago_monitor_path)

categories = [
    # 'A funny clips, animals, comics or strip cartoons',
    "happy dog"
    # 'It is about dishes, foods, eating things', 
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    # 'It is about political issues or politicians in governments',
    # 'It is about social issues, pandemics or vaccine for COVID-19 virus or photos of family',
    # 'It is about monetary economy, stocks or bitcoins',
    # 'A photo of fashions, clothings, wearings or shoes',
    # 'A photo of musicians, entertainers, celebrity, music bands or idols',
    # 'A photo of sports or athletes', 
    # 'A photo of pet animals or companion animals',
    # 'A photo of self-taken (portrait) photographs',
    # 'A photo or picture of strip cartoons, comics, sketches or drawings', 
    # 'It is about movies or cinema', 
    # 'A photo that has taken in video games',
    # 'A photo of landscape or natural scenery',
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