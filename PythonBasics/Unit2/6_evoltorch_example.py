import torch
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm,SNES
from evotorch.logging import StdOutLogger
from evotorch.operators import OnePointCrossOver, GaussianMutation

def count_ones(x: torch.Tensor) -> torch.Tensor:
    return (x > 0.5).sum(dim=-1)

problem = Problem(
    "max",
    count_ones,
    initial_bounds=(0.0, 1.0),
    solution_length=100,
    vectorized=True,
)

searcher = SNES(
    problem,
    popsize=10,
    stdev_init=0.1,
)

StdOutLogger(searcher, interval=50)

searcher.run(num_generations=1000)

best = searcher.status["best"]

print(f"Best fitness: {best.evals.item():.2g}")
print(f"Solution norm: {torch.linalg.norm(best.values).item():.2g}")
print(f"Maximum absolute value: {best.values.abs().max().item():.2g}")
print(f"Mean absolute value: {best.values.abs().mean().item():.2g}")
print("Best solution:")
print(best.values)