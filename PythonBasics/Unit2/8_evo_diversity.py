import matplotlib.pyplot as plt
import torch
from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PandasLogger

def count_ones(x: torch.Tensor) -> torch.Tensor:
    return (x > 0.5).sum(dim=-1)

def diversity() -> dict:
    values = searcher.population.access_values(keep_evals=True)  # (N, D)
    dists = torch.cdist(values, values)
    n = values.shape[0]
    pairwise = dists[torch.triu_indices(n, n, offset=1).unbind()].mean()
    return {"diversity": pairwise.item()}

problem = Problem(
    "max",
    count_ones,
    initial_bounds=(0.0, 1.0),
    solution_length=100,
    vectorized=True,
)

searcher = SNES(
    problem,
    popsize=1000,
    stdev_init=0.1,
)

searcher.after_step_hook.append(diversity)   # <-- the only addition needed

StdOutLogger(searcher, interval=50)          # will now also print "diversity"
pandas_logger = PandasLogger(searcher)       # collects full history for plotting

searcher.run(num_generations=1000)

best = searcher.status["best"]
print(f"Best fitness: {best.evals.item():.2g}")
print(f"Solution norm: {torch.linalg.norm(best.values).item():.2g}")
print(f"Maximum absolute value: {best.values.abs().max().item():.2g}")
print(f"Mean absolute value: {best.values.abs().mean().item():.2g}")
print("Best solution:")
print(best.values)

# Visualize diversity over generations
df = pandas_logger.to_dataframe()
df["diversity"].plot(title="Population diversity over generations", xlabel="generation", ylabel="mean pairwise distance")
plt.show()