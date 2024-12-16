import os
import torch
from torch_geometric.utils import homophily
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix


# カレントディレクトリにデータ保存用のフォルダを作成
data_dir = "./data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# データセットのダウンロードと読み込み
# "Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers" and "Questions"
dataset = HeterophilousGraphDataset(root=data_dir, name='Questions')
data = dataset[0]

# 隣接行列を取得 (通常は data.edge_index が入力)
edge_index = data.edge_index
num_nodes = data.num_nodes

# 隣接行列を n-hop に一般化する関数
def compute_n_hop_adj(edge_index, num_nodes, n):
    """
    edge_index を基に n-hop の隣接行列を計算し、edge_index に戻す。
    :param edge_index: torch.Tensor, shape [2, num_edges]
    :param num_nodes: int, グラフのノード数
    :param n: int, hop の数
    :return: n-hop の edge_index
    """
    # 1. スパース隣接行列を作成
    adj_matrix = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    
    # 2. n-hop の隣接行列を計算
    n_hop_adj = adj_matrix
    for _ in range(n - 1):
        n_hop_adj = n_hop_adj @ adj_matrix

    # 3. 自己ループを削除
    n_hop_adj.setdiag(0)

    # 4. edge_index に戻す
    n_hop_edge_index, _ = from_scipy_sparse_matrix(n_hop_adj)
    return n_hop_edge_index

# edge_index 同士で重複する部分を削除する関数
def remove_duplicates_from_multiple_edge_indices(edge_index_a, *edge_indices_to_remove):
    """
    edge_index_a から複数の edge_index に含まれるエッジを削除する。
    :param edge_index_a: torch.Tensor, shape [2, num_edges_a]
    :param edge_indices_to_remove: 可変長引数, 他の edge_index
    :return: torch.Tensor, 重複部分を除いた edge_index_a
    """
    # edge_index_a をセット形式に変換
    edges_a = set(map(tuple, edge_index_a.t().tolist()))
    
    # 削除対象のエッジをすべて結合 (和集合)
    edges_to_remove = set()
    for edge_index in edge_indices_to_remove:
        edges_to_remove.update(map(tuple, edge_index.t().tolist()))

    # edge_index_a から削除対象のエッジを除外
    unique_edges = edges_a - edges_to_remove

    # 結果を edge_index の形式に戻す
    if len(unique_edges) == 0:
        return torch.empty((2, 0), dtype=edge_index_a.dtype)  # 空の edge_index
    unique_edge_index = torch.tensor(list(unique_edges), dtype=edge_index_a.dtype).t()
    return unique_edge_index

# 任意の hop 数でホモフィリーを計算する関数
def compute_homophily_for_n_hop(edge_index, num_nodes, labels):
    """
    最大 max_hop までの各 hop でのホモフィリーを計算する。
    :param edge_index: torch.Tensor, shape [2, num_edges]
    :param num_nodes: int, グラフのノード数
    :param labels: torch.Tensor, shape [num_nodes]
    :param max_hop: int, 最大の hop 数
    :return: 各 hop のホモフィリー値の辞書
    """
    one_hop_edge_index = edge_index
    h = homophily(one_hop_edge_index, labels)
    print(f"{1}-hop homophily: {h}")

    _two_hop_edge_index = compute_n_hop_adj(edge_index, num_nodes, 2)
    two_hop_edge_index = remove_duplicates_from_multiple_edge_indices(_two_hop_edge_index, one_hop_edge_index)
    h = homophily(two_hop_edge_index, labels)
    print(f"{2}-hop homophily: {h}")
    
    _three_hop_edge_index = compute_n_hop_adj(edge_index, num_nodes, 3)
    three_hop_edge_index = remove_duplicates_from_multiple_edge_indices(_three_hop_edge_index, two_hop_edge_index, one_hop_edge_index)
    h = homophily(three_hop_edge_index, labels)
    print(f"{3}-hop homophily: {h}")

    _four_hop_edge_index = compute_n_hop_adj(edge_index, num_nodes, 4)
    four_hop_edge_index = remove_duplicates_from_multiple_edge_indices(_four_hop_edge_index, three_hop_edge_index, two_hop_edge_index, one_hop_edge_index)
    h = homophily(four_hop_edge_index, labels)
    print(f"{4}-hop homophily: {h}")

    _five_hop_edge_index = compute_n_hop_adj(edge_index, num_nodes, 5)
    five_hop_edge_index = remove_duplicates_from_multiple_edge_indices(_five_hop_edge_index, four_hop_edge_index, three_hop_edge_index, two_hop_edge_index, one_hop_edge_index)
    h = homophily(five_hop_edge_index, labels)
    print(f"{5}-hop homophily: {h}")


# 最大3-hopのホモフィリーを計算
homophily_results = compute_homophily_for_n_hop(data.edge_index, data.num_nodes, data.y)
