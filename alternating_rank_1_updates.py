import numpy as np
import torch
import torch_CPDecomp as cpd

def tensormap(T, vectors, modes):
    remaining_modes = [i for i in range(len(T.shape)) if i not in modes]
    a = ord('a')
    expression = ''.join([chr(a + i) for i in range(len(T.shape))]) + ',' + ','.join([chr(a + i) for i in modes]) + '->' + ''.join([chr(a + i) for i in remaining_modes])
    output = torch.einsum(expression, T, *vectors)
    return output

def power_iter(T, vectors, max_iter=100, tolerance=1e-12):
    order = len(T.shape)
    for _ in range(max_iter):
        difference = 0
        old_vectors = vectors.copy()
        for i in range( order ):
            vectors[i] = tensormap(T, [vectors[j] for j in range_except(order, i)], modes=list(range_except(order, i)))
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


def alternating_minimization(tensor, start, max_iter=100, tolerance=1e-12,):
    A = start.copy()
    order = len(tensor.shape)
    for j in range(max_iter):
        previous_A = A.copy()
        for l in range(start.rank):
            for i in range(start.order):

                map_args = ([A.components[j][:, l] for j in range_except(A.order, i)], list(range_except(A.order, i)))

                A.components[i][:, l] = tensormap(tensor, *map_args) - A.drop_components(l).map(*map_args)
                A.components[i][:, l] /= torch.linalg.norm(A.components[i][:, l])

            map_args = ([A.components[j][:, l] for j in range(A.order)], list(range(A.order)))
            A.weights[l] = tensormap(tensor, *map_args) - A.drop_components(l).map(*map_args)

        difference = max([((previous_A.components[mode] - A.components[mode])**2).sum() for mode in range(A.order)])
        if difference < tolerance:
            break
    return A

def masked_alternating_minimization(tensor, start, mask, max_iter=100, tolerance=1e-12):
    A = start.copy()
    for j in range(max_iter):
        previous_A = A.copy()
        for l in range(start.rank):
            for i in range(start.order):
                A.components[i][:, l] = tensormap(tensor - mask * A.drop_components(l).to_tensor(), [A.components[j][:, l] for j in range_except(A.order, i)], list(range_except(A.order, i)))
                A.components[i][:, l] /= tensormap(mask.double(), [A.components[j][:, l]**2 for j in range_except(A.order, i)], list(range_except(A.order, i)) )
                A.components[i][:, l] /= torch.linalg.norm(A.components[i][:, l])
                A.components[i][:, l] = torch.nan_to_num(A.components[i][:, l], 0)
            A.weights[l] = torch.nan_to_num(tensormap(tensor - mask * A.drop_components(l).to_tensor(), [A.components[j][:, l] for j in range(A.order)], list(range(A.order))) / tensormap(mask.double(), [A.components[j][:, l]**2 for j in range(A.order)], list(range(A.order))), 0)

        difference = max([((previous_A.components[mode] - A.components[mode])**2).sum() for mode in range(A.order)])
        if difference < tolerance:
            break
    return A

def random_normalized_vector(dim):
    v = np.random.uniform(-1, 1, size=(dim,))
    return torch.tensor(v / np.linalg.norm(v))

def alternating_rank_1_updates(tensor, rank, N_power_iter=100, N_altmin=100, start=None, tolerance=1e-5, mask=None):

    T = tensor.clone()
    order = len(T.shape)

    components = [torch.zeros( (i, rank), dtype=torch.float64) for i in T.shape]
    weights = torch.zeros(rank, dtype=torch.float64)

    for i in range(rank):
        if not start:
            vectors = [random_normalized_vector(d) for d in T.shape]
        else:
            if i < start.rank:
                vectors = [start.components[j][:, i] for j in range(order)]
            else:
                vectors = [random_normalized_vector(d) for d in T.shape]

        # print(vectors)
        vectors = power_iter(T, vectors, max_iter=N_power_iter, tolerance=tolerance)
        weights[i] = tensormap(T, vectors, list(range(order)))
        
        T = T - cpd.CPTensor(vectors, torch.tensor([weights[i]])) .to_tensor()
        for j in range(order):
            components[j][:, i] = vectors[j].squeeze()
    
    approximation = cpd.CPTensor(components, weights)
    if mask is None:
        approximation = alternating_minimization(tensor, approximation, max_iter=N_altmin, tolerance=tolerance)
    else:
        approximation = masked_alternating_minimization(tensor, approximation, mask, max_iter=N_altmin, tolerance=tolerance)
    return approximation


if __name__ == '__main__':

    T = torch.tensor(np.random.uniform(-1, 1, size=(10, 10, 10)))
    A = alternating_rank_1_updates(T, 100)

    print(torch.sum( ( A.to_tensor() - T)**2).item() )

    T = A.drop_components(list(range(4, 100)))
    print(T.rank, T.weights)
    mask = torch.tensor(np.random.uniform(size=(10, 10, 10))) < 0.5
    T = T.to_tensor()
    A = alternating_rank_1_updates(mask * T, 4, mask=mask)

    print(torch.sum( ( A.to_tensor() - T)**2).item() )
    print(torch.sum( ( A.to_tensor() - mask * T)**2).item() )
    print(torch.sum( ( T - mask * T)**2).item() )
    # print("--------------------")

    # A = alternating_rank_1_updates(T, 4)
    # print(torch.sum( ( A.to_tensor() - T)**2).item() )

    # A = alternating_rank_1_updates(1/T.sum(axis=2), 4)
    # print(torch.sum( ( A.to_tensor() - 1/T.sum(axis=2))**2).item() )
    # print(A.weights, A.components)

    # A = alternating_rank_1_updates(T.sum(axis=2), 4)
    # print(torch.sum( ( A.to_tensor() - T.sum(axis=2))**2).item() )


    # T = T / T.sum(axis=2)[:, :, None]
    # print(T.sum(axis=2))
    # A = alternating_rank_1_updates(T, 16)
    # print(torch.sum( ( A.to_tensor() - T)**2).item() )

