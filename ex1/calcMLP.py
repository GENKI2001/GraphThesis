import os
import torch
from torch_geometric.datasets import WebKB, Actor, WikipediaNetwork, HeterophilousGraphDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch_geometric.utils import to_dense_adj
import numpy as np
from tools.compute_laplacian import compute_laplacian
from tools.compute_matrix_series import compute_matrix_series
from tools.compute_alternating_series import compute_alternating_series

# cornell, texas, wisconsin, chameleon, squirrel, film
DATASET_NAME = "cornell"
# /geom_gcn, 
SUB_DIR = ""
SPLITS = range(10)  # 0~9
EPOCHS = 200
# MLP, ET, ALT_ET
MODEL = "MLP"
# ハイパーパラメータ t
t = 0.3

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        # 隠れ層なしで、入力層から直接出力層へ
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)

# カレントディレクトリにデータ保存用のフォルダを作成
data_dir = "./data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# データセットのダウンロードと読み込み
if DATASET_NAME == "wisconsin" or DATASET_NAME == "cornell" or DATASET_NAME == "texas":
    dataset = WebKB(root=data_dir, name=DATASET_NAME)
elif DATASET_NAME == "chameleon" or DATASET_NAME == "crocodile" or DATASET_NAME == "squirrel":
    dataset = WikipediaNetwork(root=data_dir, name=DATASET_NAME)
elif DATASET_NAME == "film":
    dataset = Actor(root=data_dir+"/film")

data = dataset[0]
# 隣接行列を取得 (通常は data.edge_index が入力)
edge_index = data.edge_index
num_nodes = data.num_nodes
# ノード特徴量とラベル
x = data.x 
y = data.y 
adjacency_matrix = to_dense_adj(edge_index)[0] # 隣接行列
laplacian_matrix = compute_laplacian(adjacency_matrix) # ラプラシアン行列
M_t = compute_matrix_series(laplacian_matrix, t) # フィルター行列
M_alt_t = compute_alternating_series(laplacian_matrix, t) # フィルター行列

# モデルの初期化
input_dim = dataset.num_features
output_dim = dataset.num_classes

# 訓練ループの前にテストを繰り返す回数を設定
test_accuracies = []
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MODEL == "MLP":
    pass
elif MODEL == "ET":
    x = M_t @ x  # フィルター行列を適用
elif MODEL == "ALT_ET":
    x = M_alt_t @ x  # フィルター行列を適用

for split in SPLITS:
    # splitデータの読み込み
    split_data = np.load(data_dir+'/'+DATASET_NAME+SUB_DIR+'/raw/'+DATASET_NAME+'_split_0.6_0.2_'+str(split)+'.npz')  # Wisconsinデータ用のsplitファイルを使用
    train_mask = torch.from_numpy(split_data['train_mask']).bool()
    val_mask = torch.from_numpy(split_data['val_mask']).bool()
    test_mask = torch.from_numpy(split_data['test_mask']).bool()

    # 訓練データとテストデータ
    x_train = x[train_mask]
    y_train = y[train_mask]
    x_test = x[test_mask]
    y_test = y[test_mask]

    # モデルの初期化
    model = SimpleNN(input_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 訓練ループ
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # フォワードパス
        out = model(x_train)

        # 損失計算とバックプロパゲーション
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    # テスト
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test).argmax(dim=1)

    # 精度を計算
    accuracy = accuracy_score(y_test.cpu(), y_pred.cpu())
    test_accuracies.append(accuracy)
    print(f"Test Accuracy for run : {accuracy:.4f}")

# 精度の平均を計算
average_accuracy = sum(test_accuracies) / len(SPLITS)
print(f"Average Test Accuracy over {len(SPLITS)} runs: {average_accuracy:.4f}")
