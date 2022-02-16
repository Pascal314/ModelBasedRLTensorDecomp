import numpy as np
import math
import torch
import itertools

def istype(_object, _type):
    return type(_object) == _type

class CPTensor:
    def __init__(self, components, weights):
        
        convert_to_tensor = lambda array: array.clone() if isinstance(array, torch.Tensor) else torch.tensor(array)

        self.components = [convert_to_tensor(c) for c in components]
        for i, array in enumerate(self.components):
            if len(array.shape) == 1:
                self.components[i] = self.components[i][:, None]
        
        self.rank = self.components[0].shape[1]
        
        for array in self.components:
            assert self.rank == array.shape[1]
        
        self.weights = convert_to_tensor(weights)
        assert weights.shape[0] == self.rank
        
        self.order = len(self.components)
        self.shape = tuple(component.shape[0] for component in self.components)

    def to_tensor(self):
        return self.from_components(self.components, self.weights)

    def map(self, vectors, modes):
        weights = self.weights.clone()
        remaining_modes = [i for i in range(self.order) if i not in modes]
        for i, mode in enumerate(modes):
            # print(self.components[mode].dtype, vectors[i].dtype, i, mode)
            inner_products = self.components[mode].t() @ vectors[i]
            weights *= inner_products
        
        if len(remaining_modes) > 0:
            return self.from_components( [self.components[mode] for mode in remaining_modes], weights)
        else:
            return torch.sum(weights)

    def drop_components(self, components_to_drop):
        # maybe writing a masking functionality is faster, but as long as the __init__ of CPTensor is fast everything should be fine
        # check if rank is a list
        if isinstance(components_to_drop, int):
            components_to_drop = [components_to_drop]
        else:
            assert isinstance(components_to_drop, list)
        
        mask = torch.ones((self.rank, ), dtype=bool)
        mask[components_to_drop] = False
        
        new_components = [component[:, mask] for component in self.components]
        new_weights = self.weights[mask]

        return CPTensor(new_components, new_weights)

    @staticmethod
    def from_components(components, weights):
        # Surely this can't be the best way to do this, maybe expand_dims is better?
        # The 'z' can also be removed by multiplying the first factor with the weights first.
        assert len(components) < 25, "This function doesnt work for tensors of order bigger than 25"
        a = ord('a')

        expression = ','.join([chr(a + i) + 'z' for i in range(len(components))])
        expression = expression + ', z -> ' + ''.join([chr(a + i) for i in range(len(components))])
        
        T = torch.einsum(expression, *(*components, weights))
        return T


    def  __add__(self, cp_tensor):

        if istype(cp_tensor, CPTensor):
            components = []
            for i in range(self.order):
                components.append(torch.cat([self.components[i], cp_tensor.components[i]], axis=1))
            weights = torch.cat([self.weights, cp_tensor.weights])
            return CPTensor(components, weights)

        elif istype(cp_tensor, SparseCPTensor):
            return MixedSparseDenseCPTensor(self, cp_tensor)

        else:
            return NotImplemented
    
    def __sub__(self, cp_tensor):
        return self + (-cp_tensor)
    
    def __neg__(self):
        new = self.copy()
        new.weights *= -1
        return new
    
    
    def get_items_at_index(self, index):
        index = index.T
        return (self.weights * math.prod([self.components[i][j, :] for i, j in enumerate(index)])).sum(axis=1)

    def copy(self):
        components = [c.clone() for c in self.components]
        weights = self.weights.clone()
        return CPTensor(components, weights)
    
    def approximate_frobenius_norm(self, n=None):
        product_shape = math.prod([d for d in self.shape])
        
        if not n:
            n = min(100000, int(0.1 ** self.order * product_shape))
        
        index = torch.tensor([np.random.randint(0, d, size=(n, )) for d in self.shape], dtype=torch.int64).T
        assert index.shape == (n, self.order)
        
        values = self.get_items_at_index(index)
        return math.sqrt((values**2).sum() * product_shape / n)

    def sort(self):
        index = torch.argsort(self.weights, descending=True)
        self.weights = self.weights[index]
        self.components = [component[:, index] for component in self.components]

class SparseCPTensor(CPTensor):
    def __init__(self, *args):
        # print(args)
        super().__init__(*args)

        for i in range(self.order):
            assert self.components[i].is_sparse

    def __add__(self, cp_tensor):
        if istype(cp_tensor, CPTensor):
            return MixedSparseDenseCPTensor(cp_tensor, self)
        
        elif istype(cp_tensor, SparseCPTensor):
            components = []
            for i in range(self.order):
                components.append(torch.cat([self.components[i], cp_tensor.components[i]], axis=1))
            weights = torch.cat([self.weights, cp_tensor.weights])
            return SparseCPTensor(components, weights)
        
        else:
            return NotImplemented
    
    @staticmethod
    def from_components(components, weights):
        if len(components) == 1:
            return (components[0] @ weights)
        else:
            raise NotImplementedError

    def copy(self):
        components = [c.clone() for c in self.components]
        weights = self.weights.clone()
        return SparseCPTensor(components, weights)

    def get_items_at_index(self, index):
        return torch.tensor([self._get_item_at_index(i) for i in index])

    def _get_item_at_index(self, index):
        temp = torch.zeros((self.order, self.rank), dtype=torch.float64)
        for mode in range(self.order):
            temp[mode, :] = self.components[mode][index[mode]].to_dense()
        return torch.dot(torch.prod(temp, axis=0), self.weights)
    
    def find_nonzero_indices(self):
        non_zero_indices = torch.empty((0, self.order), dtype=torch.int64)
        for l in range(self.rank): 
            # What happens if torch.select(self.components[mode], 1, l)._indices().squeeze(0) is empty? not entirely clear.
            non_zero = torch.tensor(list(itertools.product(*[torch.select(self.components[mode], 1, l)._indices().squeeze(0) for mode in range(self.order)])), dtype=torch.int64)
            non_zero_indices = torch.cat([non_zero, non_zero_indices], axis=0)
        return non_zero_indices

    def approximate_frobenius_norm(self, n=None):
        # Currently very slow.
        product_shape =  math.prod([d for d in self.shape])
        if not n:
            n = min(1000, int(0.1**self.rank *product_shape))

        index = self.find_nonzero_indices()
        number_of_nonzeros = index.shape[0]
        random_choice = np.random.randint(0, number_of_nonzeros, size=(n,), dtype=np.int64)
        index = index[random_choice, :]
        values = self.get_items_at_index(index)
        return math.sqrt((values**2).sum() * number_of_nonzeros / n)
    # override approximate_frobenius_norm!!

class MixedSparseDenseCPTensor:
    def __init__(self, dense_cp_tensor, sparse_cp_tensor):
        self.dense = dense_cp_tensor.copy()
        self.sparse = sparse_cp_tensor.copy()
        assert self.dense.shape == self.sparse.shape
        self.shape = self.dense.shape
        self.order = self.dense.order
        self.rank = self.dense.rank + self.sparse.rank

    def __add__(self, cp_tensor):

        if istype(cp_tensor, CPTensor):
            return MixedSparseDenseCPTensor(self.dense + cp_tensor, self.sparse)
        
        elif istype(cp_tensor, SparseCPTensor):
            return MixedSparseDenseCPTensor(self.dense, self.sparse + cp_tensor)

        elif istype(cp_tensor, MixedSparseDenseCPTensor):
            return MixedSparseDenseCPTensor(self.dense + cp_tensor.dense, self.sparse + cp_tensor.sparse)

        else:
            return NotImplemented

    def __radd__(self, cp_tensor):
        return self + cp_tensor

    def __neg__(self):
        self.dense = -self.dense
        self.sparse = -self.sparse

    def __sub__(self, cp_tensor):
        return self + -cp_tensor

    def __rsub__(self, cp_tensor):
        return self - cp_tensor

    def map(self, vectors, modes):
        return self.dense.map(vectors, modes) + self.sparse.map(vectors, modes)

    def approximate_frobenius_norm(self, n=None):
        raise NotImplementedError

    def get_items_at_index(self, index):
        return self.dense.get_items_at_index(index) + self.sparse.get_items_at_index(index)

    def drop_components(self, components_to_drop):
        raise NotImplementedError

    def copy(self):
        return MixedSparseDenseCPTensor(self.dense, self.sparse)

    def to_tensor(self):
        return self.dense.to_tensor() + self.sparse.to_tensor()


def cp_power_iter(T, vectors, max_iter=100, tolerance=1e-12):
    for _ in range(max_iter):
        difference = 0
        old_vectors = vectors.copy()
        for i in range(T.order):
            vectors[i] = T.map( [vectors[j] for j in range_except(T.order, i)], modes=list(range_except(T.order, i)))
            vectors[i] = vectors[i] / torch.linalg.norm(vectors[i])
            difference = max(difference, torch.linalg.norm(vectors[i] - old_vectors[i]))

        if difference < tolerance:
            break

    return vectors


def range_except(n, excluded):
    if type(excluded) == int:
        excluded = [excluded]
    for i in range(n):
        if not i in excluded:
            yield i


def alternating_cp_minimization(cp_tensor, start, max_iter=100, tolerance=1e-12):
    A = start.copy()
    # loss = np.infty
    for j in range(max_iter):
        # previous_loss = loss
        previous_A = A.copy()
        for l in range(start.rank):
            for i in range(start.order):
                A.components[i][:, l] = (cp_tensor - A.drop_components(l)).map( [A.components[j][:, l] for j in range_except(A.order, i)], list(range_except(A.order, i)))
                # unsure about this next line. Should it just be normalized or should it be divided by the actual denominator in the equation? (the product of the norms of the other components)
                # dividing by the denominator seems like it could be more prone to instability.
                # what do I do with the weights during optimization?
                # the solution of the optimization of component i is independent of weight i, so updating the weight at the end is perfectly valid
                # one could also distribute every weight over each component at the start when creating A and then set all weights to 1
                A.components[i][:, l] /= torch.linalg.norm(A.components[i][:, l])
            A.weights[l] = (cp_tensor - A.drop_components(l)).map([A.components[j][:, l] for j in range(A.order)], list(range(A.order)))

        # This is not exactly a good idea as there the approximate frobenius norm is not necessarily accurate.
        # loss = (cp_tensor - A).approximate_frobenius_norm(10000)
        difference = max([((previous_A.components[mode] - A.components[mode])**2).sum() for mode in range(A.order)])
        # print([((previous_A.components[mode] - A.components[mode])**2).sum() for mode in range(A.order)])
        # print(difference)
        if difference < tolerance:
            break
    return A


def random_normalized_vector(dim):
    v = np.random.uniform(-1, 1, size=(dim,))
    return torch.tensor(v / np.linalg.norm(v))

def alternating_rank_1_updates(cp_tensor, rank, N_power_iter=100, N_altmin=100, start=None, tolerance=1e-5):

    T = cp_tensor.copy()

    if not start:
        components = [torch.zeros( (i, rank), dtype=torch.float64) for i in T.shape]
        weights = torch.zeros(rank, dtype=torch.float64)
    else:
        components = start.components.copy()
        weights = start.weights.copy()


    for i in range(rank):
        if not start:
            vectors = [random_normalized_vector(d) for d in T.shape]
        else:
            vectors = [components[j][:, i] for j in range(T.order)]
        vectors = cp_power_iter(T, vectors, max_iter=N_power_iter, tolerance=tolerance)
        # weights[i] = torch.linalg.norm(T.map(vectors, list(range(T.order))))
        weights[i] = (T.map(vectors, list(range(T.order))))
        
        T = T - CPTensor(vectors, torch.tensor([weights[i]]))
        for j in range(T.order):
            components[j][:, i] = vectors[j].squeeze()
    
    approximation = CPTensor(components, weights)

    approximation = alternating_cp_minimization(cp_tensor, approximation, max_iter=N_altmin, tolerance=tolerance)
    return approximation

def cp_tensor_completion(cp_tensor, rank, omega, N_power_iter=20, tolerance=1e-5):
    T = cp_tensor.copy()

    if not start:
        components = [torch.zeros( (i, rank), dtype=torch.float64) for i in T.shape]
        weights = torch.zeros(rank, dtype=torch.float64)
    else:
        components = start.components.copy()
        weights = start.weights.copy()

    for i in range(rank):
        if not start:
            vectors = [random_normalized_vector(d) for d in T.shape]
        else:
            vectors = [components[j][:, i] for j in range(T.order)]
        vectors = cp_power_iter(T, vectors, max_iter=N_power_iter, tolerance=tolerance)
        # weights[i] = torch.linalg.norm(T.map(vectors, list(range(T.order))))
        weights[i] = (T.map(vectors, list(range(T.order))))
        
        T = T - CPTensor(vectors, torch.tensor([weights[i]]))
        for j in range(T.order):
            components[j][:, i] = vectors[j].squeeze()

    # make a masked tensor with modified map function that computes the map naively since the mask is assumed to be sparse.
    


if __name__ == "__main__":
    # Test a huge tensor: (a trillion (10^12) entries)
    from time import time
    import cProfile, pstats

    start = time()

    m, n, k = 100, 100, 100
    r = 30
    
    def generate_factor_matrix(shape):
        X = np.random.uniform(-1, 1, size=shape)
        X = X / np.sqrt(np.sum(X**2, axis=0))
        return torch.tensor(X)

    A = generate_factor_matrix((m, r))
    B = generate_factor_matrix((n, r))
    C = generate_factor_matrix((k, r))

    T = CPTensor([A, B, C], torch.tensor(np.exp(10 * (np.linspace(0.1, 1, r)[::-1] - 1))))
    print(T.components[0].dtype)
    profiler=cProfile.Profile()
    profiler.enable()
    B = alternating_rank_1_updates(T, 10, N_altmin=1000, N_power_iter=1000, tolerance=1e-16)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.print_stats()
    stats.dump_stats('dense_stats')
    elapsed_time = time() - start

    print((B - T).approximate_frobenius_norm(), T.approximate_frobenius_norm(), (B - T).approximate_frobenius_norm()/ (T).approximate_frobenius_norm())
    # print(B.weights, T.weights)

    print(f'Dense test took {elapsed_time:.2f} seconds')

    # # test a sparse tensor

    # start = time()

    # m, n, k = 1000, 1000, 1000
    # r = 100
    
    # def generate_sparse_factor_matrix(shape):
    #     # X = np.random.uniform(-1, 1, size=shape)
    #     # X = X * (np.random.uniform(size=shape) < 0.999)
    #     X = (np.random.uniform(size=shape) < 0.001)
    #     X = X / np.sqrt(np.sum(X**2, axis=0) + 1e-12)
    #     indices = np.where(X > 0)
        
    #     values = X[indices]
    #     indices = np.array(indices)
    #     return torch.sparse_coo_tensor(indices=indices, values=values, size=shape)

    # def generate_sparse_factor_matrix(shape):
    #     indices = np.array([np.random.randint(0, shape[0], size=(shape[1])), np.arange(shape[1])])
    #     values = np.ones(shape[1])
    #     return torch.sparse_coo_tensor(indices=indices, values=values, size=shape)
        

    # A = generate_sparse_factor_matrix((m, r))
    # B = generate_sparse_factor_matrix((n, r))
    # C = generate_sparse_factor_matrix((k, r))

    # import cProfile, pstats
    # T = SparseCPTensor([A, B, C], 2*torch.ones(r, dtype=torch.float64))

    # profiler=cProfile.Profile()
    # profiler.enable()
    # B = alternating_rank_1_updates(T, 100, N_altmin=10, N_power_iter=10, tolerance=1e-16)
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('ncalls')
    # stats.print_stats()
    # stats.dump_stats('sparse_stats')
    # elapsed_time = time() - start

    # print((B - T).approximate_frobenius_norm(1000), T.approximate_frobenius_norm(1000), (B - T).approximate_frobenius_norm(1000)/ ((T).approximate_frobenius_norm(1000) + 1e-12))
    # # print(B.weights, T.weights)

    # print(f'Sparse test took {elapsed_time:.2f} seconds')


