import numpy as np
from scipy.linalg import qr
import copy

class TTCross:
    def __init__(self, dim, n_physical, input_oracle, rank, tt_init, num_sweep) -> None:
        self.dim = dim                                                      # dimension of the input function
        self.n_physical = n_physical                                        # number of states in each dimension
        # the size of the full tensor is n_physical^dim
        self.input_oracle = input_oracle                                    # the input function
        self.rank = rank                                                    # the targeted TT-rank, array of size dim - 1
        self.tt_init = tt_init                                              # the initial TT
        self.num_sweep = num_sweep                                          # number of sweeps (back and forth)


    def right_to_left_orthogonal(self):
        '''
        input: a tt (initial guess of the tt-cross)
        output: a right orthogonal tt and a right index set
        '''
        right_index = []

        tt_cores = copy.deepcopy(self.tt_init)

        k = self.dim - 1
        core = self.tt_init[k][:,:,0]
        qq,rr = qr(core.T,mode='economic')

        tt_cores[k] = copy.deepcopy((qq.T)[:,:,np.newaxis])
        tt_cores[k-1] = np.einsum('ijk,lk->ijl', tt_cores[k-1], rr)

        pp = qr(qq.T, pivoting=True)[2]
        r_idx = pp[:self.rank[k-1]]
        right_index.append(r_idx.reshape([self.rank[k-1], 1]))

        for k in np.arange(1, self.dim - 1)[::-1]:
            core = tt_cores[k].reshape([self.rank[k-1], self.n_physical * self.rank[k]])
            qq,rr = qr(core.T,mode='economic')

            tt_cores[k] = (qq.T).reshape([self.rank[k-1], self.n_physical, self.rank[k]])
            tt_cores[k-1] = np.einsum('ijk,lk->ijl', tt_cores[k-1], rr)

            pp = qr(qq.T, pivoting=True)[2]

            r_idx_fold = []
            for i in range(self.n_physical):
                for j in range(self.rank[k]):
                    r_idx_fold.append(np.hstack([i, r_idx[j]]))
            r_idx_fold = np.array(r_idx_fold)

            r_idx = r_idx_fold[pp[:self.rank[k-1]]]
            right_index.append(copy.deepcopy(r_idx))

        right_index = right_index[::-1]

        return tt_cores, right_index


    def sweep(self):
        tt_cores, right_index = self.right_to_left_orthogonal()
        left_index = list(np.zeros(self.dim - 1))

        for sweep in range(self.num_sweep):
            for k in range(self.dim - 1):
                if k == 0:
                    r_idx = right_index[0]

                    mat_v = np.empty([self.n_physical, len(r_idx)])
                    for i in range(self.n_physical):
                        for j in range(len(r_idx)):
                            index = np.hstack([i, r_idx[j]])
                            mat_v[i,j] = self.input_oracle(index)

                    qq,rr = qr(mat_v, mode='economic')
                    pp = qr(qq.T, pivoting=True)[2]
                    index_sub = pp[:self.rank[0]]

                    l_idx = index_sub.reshape([self.rank[0],1])
                    left_index[0] = copy.deepcopy(l_idx)

                    core = np.dot(qq, np.linalg.inv(qq[index_sub,:]))
                    tt_cores[0] = copy.deepcopy(core[np.newaxis,:,:])

                    temp = np.dot(qq[index_sub,:],rr)
                    tt_cores[1] = np.einsum('ij,jkl->ikl', temp, tt_cores[1])

                else:
                    r_idx = right_index[k]
                    l_idx_fold = []
                    for i in range(len(l_idx)):
                        for n in range(self.n_physical):
                            l_idx_fold.append(np.hstack([l_idx[i], n]))
                    l_idx_fold = np.array(l_idx_fold)

                    mat_v = np.empty([len(l_idx_fold), len(r_idx)])
                    for i in range(len(l_idx_fold)):
                        for j in range(len(r_idx)):
                            index = np.hstack([l_idx_fold[i], r_idx[j]])
                            mat_v[i,j] = self.input_oracle(index)

                    qq,rr = qr(mat_v, mode='economic')
                    pp = qr(qq.T, pivoting=True)[2]
                    index_sub = pp[:self.rank[k]]
                    
                    l_idx = l_idx_fold[index_sub]
                    left_index[k] = copy.deepcopy(l_idx)

                    core = np.dot(qq, np.linalg.inv(qq[index_sub,:]))
                    tt_cores[k] = copy.deepcopy(core.reshape([self.rank[k-1], self.n_physical, self.rank[k]]))
                    
                    temp = np.dot(qq[index_sub,:],rr)
                    tt_cores[k+1] = np.einsum('ij,jkl->ikl', temp, tt_cores[k+1])

            ### right-to-left sweep
            for k in np.arange(1, self.dim)[::-1]:
                if k == self.dim - 1:
                    l_idx = left_index[k-1]

                    mat_v = np.empty([len(l_idx), self.n_physical])
                    for i in range(len(l_idx)):
                        for j in range(self.n_physical):
                            index = np.hstack([l_idx[i], j])
                            mat_v[i,j] = self.input_oracle(index)

                    qq,rr = qr(mat_v.T, mode='economic')
                    pp = qr(qq.T, pivoting=True)[2]
                    index_sub = pp[:self.rank[k-1]]

                    r_idx = index_sub.reshape([self.rank[k-1],1])
                    right_index[k-1] = copy.deepcopy(r_idx)

                    core = np.dot(np.linalg.inv(qq[index_sub,:]).T,qq.T)
                    tt_cores[k] = copy.deepcopy(core[:,:,np.newaxis])

                    temp = np.dot(rr.T, qq[index_sub,:].T)
                    tt_cores[k-1] = np.einsum('ijk,kl->ijl',tt_cores[k-1],temp)

                else:
                    l_idx = left_index[k-1]
                    r_idx_fold = []
                    for n in range(self.n_physical):
                        for j in range(len(r_idx)):
                            r_idx_fold.append(np.hstack([n,r_idx[j]]))
                    r_idx_fold = np.array(r_idx_fold)

                    mat_v = np.empty([len(l_idx), len(r_idx_fold)])
                    for i in range(len(l_idx)):
                        for j in range(len(r_idx_fold)):
                            index = np.hstack([l_idx[i], r_idx_fold[j]])
                            mat_v[i,j] = self.input_oracle(index)

                    qq,rr = qr(mat_v.T, mode='economic')
                    pp = qr(qq.T, pivoting=True)[2]
                    index_sub = pp[:self.rank[k-1]]

                    r_idx = r_idx_fold[index_sub]
                    right_index[k-1] = copy.deepcopy(r_idx)

                    core = np.dot(np.linalg.inv(qq[index_sub,:]).T,qq.T)
                    tt_cores[k] = copy.deepcopy(core.reshape([self.rank[k-1], self.n_physical, self.rank[k]]))

                    temp = np.dot(rr.T, qq[index_sub,:].T)
                    tt_cores[k-1] = np.einsum('ijk,kl->ijl', tt_cores[k-1], temp)

            print('sweeps:', sweep)

        return tt_cores
    
    def run(self):
        tt_cores = self.sweep()
        return tt_cores