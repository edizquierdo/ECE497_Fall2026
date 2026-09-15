import torch

# Population (N individuals, D dimensions)
pop = torch.randn(5, 2)  
print(pop)

# First measure: Mean per-gene standard deviation
diversity_std = pop.std(dim=0).mean()
print(diversity_std.item())

# Second measure: Mean pairwise distance
dists = torch.cdist(pop, pop)                    # (N, N) pairwise distances
diversity_pairwise = dists[torch.triu_indices(len(pop), len(pop), offset=1).unbind()].mean()
print(diversity_pairwise.item())
