import numpy as np

class TTCrossACA:
    def __init__(self, dim, n_physical, input_oracle, rank, num_sweep) -> None:
        self.dim = dim                                                      # dimension of the input function
        self.n_physical = n_physical                                        # number of states in each dimension
        # the size of the full tensor is n_physical^dim
        self.input_oracle = input_oracle                                    # the input function
        self.rank = rank                                                    # the targeted TT-rank, array of size dim - 1
        self.num_sweep = num_sweep                                          # number of sweeps (back and forth)

        self.input_oracle_reverse = lambda idx: input_oracle(idx[::-1])     # the reversed input oracle

    def make_oracle(self, oracle_old, left_index_set, k):
        """
        Wraps a higher-dimensional oracle to simulate a partially unfolded tensor 
        at TT core index 'k'. This enables TT-cross computation at intermediate steps.

        Parameters:
        - oracle_old: callable
            The oracle from the previous level that evaluates full tensor entries.
        - left_index_set: list or array
            The set of interpolation row indices selected at level (k - 1), used to 
            reverse map from flattened row indices back to high-dimensional ones.
        - k: int
            Current TT core index (1 \le k \le dim - 1)

        Returns:
        - new_oracle: callable
            A function that takes a lower-dimensional index (of length dim - k) and
            returns a tensor entry, as if it were a flattened unfolding matrix.
        """

        def new_oracle(idx):
            # idx has length dim - k
            # idx[0] is an integer in range [0, rank[k-1] * n_physical)
            # It indexes the "rows" in the unfolding matrix.

            # Convert the flat index into (r_{k-1}, n_physical)
            i_rank, i_phys = np.unravel_index(idx[0], (self.rank[k - 1], self.n_physical))

            # Recover the original i_k index using interpolation history
            # i_rank indexes into left_index to recover the earlier prefix
            i_prefix = left_index_set[i_rank]

            # Form the full tensor multi-index
            # Final index has length dim - k + 1
            idx_long = np.hstack([i_prefix, i_phys, idx[1:]])

            # Evaluate the original high-dimensional oracle
            return oracle_old(idx_long)

        return new_oracle

    def residue(self, oracle, cross_inv, left_index_set, right_index_set, left_idx, right_idx):
        """
        Compute the ACA residual of A[left_idx, right_idx] - approx value from cross.

        Parameters:
        - oracle: callable, oracle(index) returns a scalar entry of the tensor
        - cross_inv: numpy.ndarray of shape (r, r), inverse of A[I, J]
        - left_index_set: selected row indices, list of length r, each is a row index, 
            which is a scalar if sweeping from left to right, array otherwise
        - right_index_set: selected column indices, list of length r, each is a column index,
            which is a array if sweeping from left to right, scalar otherwise
        - left_idx: candidate row index (array-like)
        - right_idx: candidate column index (array-like)

        Returns:
        - res: scalar, residual at position (left_idx, right_idx)
        """
        r = len(left_index_set)  # Current ACA rank

        if r == 0:
            # No previous cross selected - the residue is the tensor itself
            return oracle(np.hstack([left_idx, right_idx]))

        # Step 1: A[r_idx, c_idx] - the true entry
        full_idx = np.hstack([left_idx, right_idx])
        a_ij = oracle(full_idx)

        # Step 2: A[c_idx, J] - row vector of shape (1, r)
        left_cross_mat = np.empty(r)
        for j in range(r):
            idx = np.hstack([left_idx, right_index_set[j]])
            left_cross_mat[j] = oracle(idx)
        left_cross_mat = left_cross_mat.reshape([1, r])

        # Step 3: A[I, r_idx] — column vector of shape (r, 1)
        right_cross_mat = np.empty(r)
        for i in range(r):
            idx = np.hstack([left_index_set[i], right_idx])
            right_cross_mat[i] = oracle(idx)
        right_cross_mat = right_cross_mat.reshape([r, 1])

        # Step 4: Residual = A[c_idx, r_idx] - approx
        approx = (left_cross_mat @ cross_inv @ right_cross_mat)[0, 0]
        res = a_ij - approx

        return res
    
    def coordinate_descent(self, oracle, dim_physical, cross_inv, left_index_set, right_index_set, left_idx_init, right_idx_init):
        """
        Perform coordinate descent to find the next (row, column) pair that maximizes
        the absolute residual in the ACA algorithm, 
        sweeping from left to right

        Parameters:
        - oracle: function that returns a matrix entry given a multi-index
        - dim_physical: list of physical dimensions for the current unfolding
        - cross_inv: inverse of the current interpolation matrix U = A[I,J]
        - left_index_set: list of selected row indices (I)
        - right_index_set: list of selected column indices (J)
        - left_idx_init: initial guess for the row index (scalar)
        - right_idx_init: initial guess for the column multi-index (list or array)

        Returns:
        - left_idx: updated row index (scalar)
        - right_idx: updated column multi-index (array)
        """
        left_idx = left_idx_init
        right_idx = np.array(right_idx_init, copy=True)

        # Initial residual value
        value_old = self.residue(oracle, cross_inv, left_index_set, right_index_set, left_idx, right_idx)

        while True:
            # === Step 1: Optimize row index ===
            # Fix right_idx, find left_idx that maximizes |Residue[left_idx, right_idx]|

            val_row = np.empty(dim_physical[0])
            for i in range(dim_physical[0]):
                left_idx = np.array([i])
                val_row[i] = self.residue(oracle, cross_inv, left_index_set, right_index_set, left_idx, right_idx)
            left_idx = np.argmax(np.abs(val_row))

            # === Step 2: Optimize column indices one by one ===
            # Fix left_idx, optimize each coordinate of right_idx individually
            for col in np.random.permutation(len(right_idx)):
                val_col = np.empty(dim_physical[col + 1])
                for j in range(dim_physical[col + 1]):
                    right_idx_test = right_idx.copy()
                    right_idx_test[col] = j
                    val_col[j] = self.residue(oracle, cross_inv, left_index_set, right_index_set, left_idx, right_idx_test)
                right_idx[col] = np.argmax(np.abs(val_col))

            # === Step 3: Check convergence ===
            value_new = self.residue(oracle, cross_inv, left_index_set, right_index_set, left_idx, right_idx)
            if value_new == value_old:
                break
            value_old = value_new

        return left_idx, right_idx
    
    def aca_1st_unfolding(self, oracle, dim_physical, rank_current, right_idx_init_local):
        """
        Perform adaptive cross approximation (ACA) on the first unfolding of the tensor, 
        sweeping from left to right
        
        Parameters:
        ----------
        oracle : callable
            A function that returns a tensor entry for a given multi-index.
        
        dim_physical : array-like of ints
            Describes the shape of the tensor dimensions in the current unfolding.
            Usually dim_physical[0] is the "row" dimension, and the rest are "columns".
        
        rank_current : int
            Target rank for ACA; number of interpolation points to select.

        - right_idx_init_local : array
            Initialization for the coordinate descent. 
            If right_idx_init_local = 'random', then randomly pick the initialization. 

        Returns:
        -------
        left_index_set : list of ints
            List of selected row indices (scalar). 

        right_index_set : list of np.ndarray
            List of selected column indices (as multi-indices).
        """

        left_index_set = []
        right_index_set = []

        cross_inv = None  # Inverse of the current cross matrix

        for r in range(rank_current):

            # === Select a new row index that is not already chosen ===
            while True:
                left_idx_init = np.random.choice(dim_physical[0], 1)[0]  # extract scalar
                if left_idx_init not in left_index_set:
                    break
            
            # === Select a new column multi-index not already chosen ===
            if isinstance(right_idx_init_local, str) and right_idx_init_local == 'random':
                while True:
                    right_idx_init = np.random.choice(self.n_physical, len(dim_physical) - 1)
                    if tuple(right_idx_init) not in set(tuple(existing_idx) for existing_idx in right_index_set):
                        break

            else:
                right_idx_init = right_idx_init_local[r]

            # === Run coordinate descent to refine the chosen row and column ===
            left_idx, right_idx = self.coordinate_descent(oracle, dim_physical, cross_inv, left_index_set, right_index_set, left_idx_init, right_idx_init)

            # === Append the selected pivot row/column indices ===
            left_index_set.append(left_idx)
            right_index_set.append(right_idx)

            # === Build the new (r+1) x (r+1) cross matrix ===
            cross_mat = np.empty((r + 1, r + 1))
            for i in range(r + 1):
                for j in range(r + 1):
                    index = np.hstack([left_index_set[i], right_index_set[j]])
                    cross_mat[i, j] = oracle(index)

            # === Update the pseudoinverse of the cross matrix ===
            cross_inv = np.linalg.pinv(cross_mat)

        return left_index_set, right_index_set
    
    def tt_aca_sweep(self, right_idx_init):
        """
        TT-cross approximation based on ACA and coordinate descent, 
        sweeping from left to right

        Parameters:
        - right_index_init
            For the first sweep, set right_index_init = 'random'

        Returns:
        - tt_cores: list of arrays
            TT approximation
        - row_index_list: list of arrays, length being dim - 1, size being (rank[k], k+1)
            Selected row indices, can be used as an initialization for sweeping
        - col_index_list: list of arrays, length being dim - 1, size being (rank[dim-k-1], dim-1-k)
            Selected column indices, can be used as an initialization for sweeping

        Variables:
        - left_idx, right_idx: single index
        - left_index_set, right_index_set: selected indices of a certain (the k-th) unfolding matrix
        - left_index_list, right_index_list: all selected indices
        """

        # Container for TT cores
        tt_cores = []

        # First TT core (k = 0)
        k = 0
        dim_physical = np.ones(self.dim, dtype=int) * self.n_physical

        oracle = self.input_oracle  # Initial oracle function

        # Select interpolation indices for first unfolding
        if isinstance(right_idx_init, str) and right_idx_init == 'random':
            right_idx_init_local = 'random'
        else:
            right_idx_init_local = right_idx_init[k]

        left_index_set, right_index_set = self.aca_1st_unfolding(oracle, dim_physical, self.rank[k], right_idx_init_local)

        # Construct cross matrix C = A[:,J] for current unfolding
        mat_c = np.empty([self.n_physical, self.rank[k]])
        for i in range(self.n_physical):
            for j in range(self.rank[k]):
                j_long = right_index_set[j]  # Right multi-index
                idx = np.hstack([i, j_long])  # Full tensor index
                mat_c[i, j] = oracle(idx)

        # U = submatrix at selected rows (left_index)
        mat_u = mat_c[left_index_set, :]  # U = C[I,:] = A[I,J]

        # Compute interpolation core: C * U^{-1}
        core = np.dot(mat_c, np.linalg.pinv(mat_u))  # shape: (n_physical, rank[k])
        core = core.reshape([1, self.n_physical, self.rank[k]])  # reshape to (1, n_physical, rank[k]) for TT format

        tt_cores.append(core.copy())  # Save core
        left_index_list = [np.array(left_index_set).reshape([-1,1])]  # Save the selected row index: list of left_index_set
        right_index_list = [np.array(right_index_set)]  # Save the selected column index: list of right_index_set

        # Loop over intermediate TT cores
        for k in np.arange(1, self.dim - 1):
            # Adjust unfolding shape: physical × remaining dims
            dim_physical = np.ones(self.dim - k, dtype=int) * self.n_physical
            dim_physical[0] = self.rank[k - 1] * self.n_physical  # merged previous TT-rank and physical size

            oracle = self.make_oracle(oracle, left_index_set, k)

            # Select interpolation indices
            if isinstance(right_idx_init, str) and right_idx_init == 'random':
                right_idx_init_local = 'random'
            else:
                right_idx_init_local = right_idx_init[k]

            left_index_set, right_index_set = self.aca_1st_unfolding(oracle, dim_physical, self.rank[k], right_idx_init_local)

            # Build cross matrix C
            mat_c = np.empty([dim_physical[0], self.rank[k]])
            for i in range(dim_physical[0]):
                for j in range(self.rank[k]):
                    j_long = right_index_set[j]
                    idx = np.hstack([i, j_long])
                    mat_c[i, j] = oracle(idx)

            mat_u = mat_c[left_index_set, :]  # Submatrix U

            # Compute core and reshape to TT format: (r_{k-1}, n_physical, r_k)
            core = np.dot(mat_c, np.linalg.pinv(mat_u))
            core = core.reshape([self.rank[k - 1], self.n_physical, self.rank[k]])

            tt_cores.append(core.copy()) # Save core

            long_left_index_temp_1 = np.repeat(left_index_list[-1], repeats = self.n_physical, axis=0)
            long_left_index_temp_2 = np.tile(np.arange(self.n_physical), reps = self.rank[k - 1]).reshape([-1,1])
            long_left_index = np.hstack([long_left_index_temp_1, long_left_index_temp_2])  # size: (rank[k-1] * n_physical, k+1)
            left_index_full = long_left_index[left_index_set,:]

            left_index_list.append(left_index_full)  # Save the selected row index
            right_index_list.append(np.array(right_index_set))  # Save the selected row index

        # Final TT core (k = dim - 1)
        k = self.dim - 1
        mat_r = np.empty([self.rank[k - 1], self.n_physical])
        for i in range(self.rank[k - 1]):
            for j in range(self.n_physical):
                i_long = left_index_set[i]  # From previous ACA
                idx = np.hstack([i_long, j])
                mat_r[i, j] = oracle(idx)

        core = mat_r.reshape([self.rank[k - 1], self.n_physical, 1])  # Shape: (r_{d-1}, n_physical, 1)
        tt_cores.append(core.copy())  # Append last core

        return tt_cores, left_index_list, right_index_list
    
    def run(self):
        """
        TT-cross approximation based on ACA and coordinate descent, sweeping back and forth. 

        The first sweep is with random initialiation for the right index set. 
        For the following sweeps, use the index set obtained from the previous sweep as intialization. 

        Returns:
        - tt_cores: list of arrays, each array is 3-dimensional, including the first core and the last core. 
            TT approximation
        """
        # First sweep: with random initializaiton
        tt_cores, left_index_list, right_index_list = self.tt_aca_sweep('random')

        # Further improvement, using the previously obtained left index set as the initialization
        for sweep in range(self.num_sweep):
            right_idx_init = []
            for index_set in left_index_list[::-1]:
                right_idx_init.append(index_set[:,::-1])

            tt_cores, left_index_list, right_index_list = self.tt_aca_sweep(right_idx_init)

            right_idx_init = []
            for index_set in left_index_list[::-1]:
                right_idx_init.append(index_set[:,::-1])

            tt_cores, left_index_list, right_index_list = self.tt_aca_sweep(right_idx_init)

            print('sweeps:', sweep)

        return tt_cores