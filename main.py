import numpy as np
import tensorly as tl

def tensor_svd(X, rank):
   
    
    
    dims = X.shape

   
    X_unfolds = [tl.base.unfold(X, mode) for mode in range(len(dims))]

    # initialezing the svd tensor
    U_list = []
    S_list = []
    V_list = []

    # SVD
    for mode, X_unfold in enumerate(X_unfolds):
        U, S, V = np.linalg.svd(X_unfold)
        U_list.append(U[:, :rank])
        S_list.append(S[:rank])
        V_list.append(V[:rank, :])

    return U_list, S_list, V_list

# a random tensor
X = np.random.random((3, 4, 5))

#svd 
U_list, S_list, V_list = tensor_svd(X, rank=2)

for i, U in enumerate(U_list):
    print(f"Left singular vectors for mode {i}:\n{U}")


for i, S in enumerate(S_list):
    print(f"Singular values for mode {i}:\n{S}")

for i, V in enumerate(V_list):
    print(f"Right singular vectors for mode {i}:\n{V}")
