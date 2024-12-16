import os
from torch_geometric.datasets import HeterophilousGraphDataset

def download_data():
    # カレントディレクトリにデータ保存用のフォルダを作成
    data_dir = "./experiments/datasets"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # データセットのダウンロードと読み込み
    dataset = HeterophilousGraphDataset(root=data_dir, name='Tolokers')

    return dataset[0]