import random
import math
from typing import List, Tuple

# Set seed for reproducibility
random.seed(42)

# Knapsack instance
weights = [2, 5, 4, 2, 10]
values  = [6, 10, 7, 3, 20]
C = 15
alpha = 5  # penalty factor

# GA parameters
N = 6           # population size
Pc = 0.8        # crossover rate
Pm = 1/len(weights)  # per-bit mutation rate
tournament_k = 3
elitism = 1     # Number of elite individuals to carry over

def fitness(chrom: List[int]) -> float:
    """Calculates fitness of a chromosome, penalizing overweight solutions."""
    W = sum(g * w for g, w in zip(chrom, weights))
    V = sum(g * v for g, v in zip(chrom, values))
    if W <= C:
        return V
    else:
        return V - alpha * (W - C)

def one_point_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int], int]:
    """Performs one-point crossover between two parents."""
    n = len(p1)
    if n < 2:
        return p1[:], p2[:]
    k = random.randint(1, n - 1)
    c1 = p1[:k] + p2[k:]
    c2 = p2[:k] + p1[k:]
    return c1, c2, k

def mutate(chrom: List[int], pm: float) -> Tuple[List[int], List[int]]:
    """Mutates a chromosome by flipping bits based on mutation rate."""
    ch = chrom[:]
    flip_positions = []
    for i in range(len(ch)):
        if random.random() < pm:
            ch[i] = 1 - ch[i]
            flip_positions.append(i + 1)  # 1-based for reporting
    return ch, flip_positions

def tournament_select(pop: List[List[int]], fits: List[float], k: int) -> List[int]:
    """Selects one individual using k-tournament selection."""
    selected_indices = random.sample(range(len(pop)), k)
    best_index = max(selected_indices, key=lambda i: fits[i])
    return pop[best_index]

def chrom_to_str(ch: List[int]) -> str:
    """Converts a chromosome list to a string."""
    return ''.join(str(b) for b in ch)

# 1) Initial population
pop = [
    [1, 0, 1, 1, 0],  # x1
    [0, 1, 0, 0, 1],  # x2
    [1, 1, 0, 0, 0],  # x3
    [0, 0, 1, 1, 0],  # x4
    [1, 0, 0, 0, 1],  # x5
    [0, 1, 1, 0, 0],  # x6
]

# --- GA Generation Step ---

# Calculate fitness for the initial population
fits = [fitness(ch) for ch in pop]

print("Table 1: Initial population and fitness")
print("| i | Chromosome | f(xi)  |")
print("| - | ---------- | ------ |")
for i, (ch, f) in enumerate(zip(pop, fits), start=1):
    print(f"| {i} | {chrom_to_str(ch)} | {f:7.4f} |")
print()

# 2) Elitism: Identify and preserve the best individuals
# ## ADDED: Elitism implementation
sorted_pop = sorted(zip(pop, fits), key=lambda x: x[1], reverse=True)
elites = [p for p, f in sorted_pop[:elitism]]

# 3) Selection: Build a mating pool for the remaining spots
# ## CORRECTED: Use tournament selection as defined by parameters
mating_pool = []
num_to_select = N - elitism
for _ in range(num_to_select):
    selected_parent = tournament_select(pop, fits, tournament_k)
    mating_pool.append(selected_parent)

# ## ADDED: Shuffle mating pool for better random pairing
random.shuffle(mating_pool)
pairs = []
# Ensure we have an even number of parents to create pairs
num_pairs = len(mating_pool) // 2
for i in range(num_pairs):
    p1 = mating_pool[2*i]
    p2 = mating_pool[2*i + 1]
    pairs.append((p1, p2))

# 4) Crossover
children = []
crossover_records = []
for p1, p2 in pairs:
    if random.random() < Pc:
        c1, c2, k = one_point_crossover(p1, p2)
        crossover_records.append((p1, p2, f"1-pt k={k}", c1, fitness(c1), c2, fitness(c2)))
    else:
        c1, c2 = p1[:], p2[:]
        crossover_records.append((p1, p2, "none", c1, fitness(c1), c2, fitness(c2)))
    children.extend([c1, c2])

# Handle odd-sized mating pool if necessary
if len(mating_pool) % 2 != 0:
    children.append(mating_pool[-1]) # Carry over the last parent

print("Table 2: Crossover and fitness after crossover (pre-mutation)")
print("| Pair | Parent 1   | Parent 2   | Crossover | Child 1    | f(child1) | Child 2    | f(child2) |")
print("| ---- | ---------- | ---------- | --------- | ---------- | --------- | ---------- | --------- |")
for j, (p1, p2, how, c1, f1, c2, f2) in enumerate(crossover_records, start=1):
    print(f"| {j:<4} | {chrom_to_str(p1):<10} | {chrom_to_str(p2):<10} | {how:<9} | {chrom_to_str(c1):<10} | {f1:9.4f} | {chrom_to_str(c2):<10} | {f2:9.4f} |")
print()


# 5) Mutation
mutated_offspring = []
mutation_records = []
for ch in children:
    ch2, flips = mutate(ch, Pm)
    mutated_offspring.append(ch2)
    mutation_records.append((ch, flips, ch2, fitness(ch2)))

print("Table 3: Mutation and fitness after mutation")
print("| Offspring | Pre-mutation | Mutation details | Post-mutation | f(post) |")
print("| --------- | ------------ | ---------------- | ------------- | ------- |")
for idx, (pre, flips, post, fpost) in enumerate(mutation_records, start=1):
    detail = "none" if not flips else f"flip bit(s) {','.join(map(str, flips))}"
    print(f"| {idx:<9} | {chrom_to_str(pre):<12} | {detail:<16} | {chrom_to_str(post):<13} | {fpost:7.4f} |")
print()

# 6) Form the new population and report the best
# ## CORRECTED: New population includes elites + mutated offspring
new_pop = elites + mutated_offspring

# Final: report best from the *complete new population*
best_chrom, best_fit = max(zip(new_pop, [fitness(ch) for ch in new_pop]), key=lambda x: x[1])
W_best = sum(b * w for b, w in zip(best_chrom, weights))
V_best = sum(b * v for b, v in zip(best_chrom, values))

print("--- Generation 1 Complete ---")
print("\nBest individual in new population:")
print(f"  Chromosome: {chrom_to_str(best_chrom)}")
print(f"  Weight: {W_best} (Capacity: {C})")
print(f"  Value: {V_best}")
print(f"  Fitness: {best_fit:.4f}")
