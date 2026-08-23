import numpy as np
from matplotlib import pyplot as plt

from tt_cross_aca import TTCrossACA
from tt_cross_vanilla import TTCross

# Basic settings
dim = 20                                    # number of dimension
n_physical = 10                             # number of states in each dimension
rank = np.ones(dim - 1, dtype=int) * 4      # targeted TT-rank

# The Hilbert tensor, modified to be the gaussian 
def input_oracle(idx, gaussian=True):
    if gaussian:
        def f(x, m1=0.759, m2=0.740, a=1e-7):
            return np.exp( - (x - m1)**2 / a ) + np.exp( - (x - m2)**2 / a )
        step = (1 - 0) / (10**20)
        x_i = 0 + (idx) * step 
        return f(x = x_i)     
    else:
        return 1 / (np.sum(idx) + 1)
    

# Large Sparse Tensor
def sparse_input_oracle(idx, density = 0.0001):
    """
    Here, density represents the number of nonzeros
    """
    num_nonzeros = int(np.prod(idx) * density)
    pass

tt_init = []
for k in range(dim):
    if k == 0:
        tt_init.append(np.random.normal(0, 1, [1, n_physical, rank[k]]))
    elif k < dim - 1:
        tt_init.append(np.random.normal(0, 1, [rank[k-1], n_physical, rank[k]]))
    else:
        tt_init.append(np.random.normal(0, 1, [rank[k-1], n_physical, 1]))

# Run TT-cross and generate a TT approximation tt_cores
print('Running TT-cross-ACA')
tt_cores_aca = TTCrossACA(dim, n_physical, input_oracle, rank, num_sweep=10).run()
print('Running TT-cross-vanilla')
tt_cores_vanilla = TTCross(dim, n_physical, input_oracle, rank, tt_init, num_sweep=10).run()


def tt_entry(tt_cores,idx):
    """
    Calculating a given index of a TT

    Parameters:
    - tt_cores: list of arrays
    - idx: an array of size dim

    Returns:
    - entry: the corresponding entry value
    """

    entry = tt_cores[0][0,idx[0],:]
    for i in np.arange(1,dim):
        entry = np.dot(entry, tt_cores[i][:,idx[i],:])
    entry = entry.item()
    return entry

# Randomly generate some indices, and compare its approximated value with the true one

error_aca = []
error_vanilla = []

for _ in range(1000):
    idx = np.random.choice(n_physical, dim)

    true_value = input_oracle(idx)
    tt_value_aca = tt_entry(tt_cores_aca, idx)
    if true_value == 0:
        error = abs(tt_value_aca - true_value)
    else: 
        error = abs((tt_value_aca - true_value) / true_value)
    error_aca.append(error)

    tt_value_vanilla = tt_entry(tt_cores_vanilla, idx)
    if true_value == 0:
        error = abs(tt_value_vanilla - true_value)
    else:
        error = abs((tt_value_vanilla - true_value) / true_value)
    error_vanilla.append(error)

plt.figure()
plt.hist(error_aca, bins=50, label='TT-cross-ACA', alpha = 0.5)
plt.hist(error_vanilla, bins=50, label='TT-cross-vanilla', alpha = 0.5)
plt.legend()
plt.xlabel('Relative error')
plt.ylabel('Frequency')
plt.show()
