
import torch

# ラプラシアン行列を計算する関数
def compute_laplacian(adjacency_matrix):
    # 隣接行列 A
    A = adjacency_matrix

    # ノード数
    num_nodes = A.size(0)

    # 単位行列 I
    I = torch.eye(num_nodes, device=A.device)

    # 次数行列 D (Aの行和)
    D = torch.diag(A.sum(dim=1))

    # D + I と A + I
    D_tilde = D + I
    A_tilde = A + I

    # (D + I)^(-1/2)
    D_tilde_inv_sqrt = torch.linalg.inv(torch.sqrt(D_tilde))

    # ラプラシアン行列の計算
    L = I - D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt

    return L
