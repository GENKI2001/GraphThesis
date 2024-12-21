import torch
from math import factorial

# フィルター行列を計算する関数(改良版)
def compute_alternating_series(L, t):
    # ノード数
    num_nodes = L.size(0)
    
    # 単位行列 I
    I = torch.eye(num_nodes, device=L.device)
    
    # M(t) の初期値 (I)
    M_t = I.clone()
    
    # 級数の計算 (0次から4次まで)
    power_L = I.clone()  # L^0 = I

    for n in range(1, 5):  # n=1,2,3,4
        power_L = power_L @ L  # L^n を計算
        M_t += ((-1)**(n+1)) * (t**n / factorial(n)) * power_L
    
    return M_t
