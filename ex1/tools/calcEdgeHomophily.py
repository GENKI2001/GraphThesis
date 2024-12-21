
# edge_indexからエッジhomophilyを計算する関数
def calculate_edge_homophily(edge_index, y):
    """
    エッジhomophilyを計算する関数
    edge_index: エッジの接続関係 (2 x num_edges)
    y: ノードのラベル (num_nodes)
    """
    # エッジの両端のノードのラベルを取得
    src_labels = y[edge_index[0]]
    dst_labels = y[edge_index[1]]
    
    # 同じラベルを持つエッジの数を計算
    same_label_edges = (src_labels == dst_labels).sum().item()
    total_edges = edge_index.shape[1]
    
    # homophily率を計算
    edge_homophily = same_label_edges / total_edges
    
    return edge_homophily