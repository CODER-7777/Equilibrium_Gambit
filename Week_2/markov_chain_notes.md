# Markov Chains & Market Regime Simulation — Study Notes & Resources

## What is a Markov Chain?

A **Markov Chain** is a mathematical model for a system that transitions between a finite set of **states** over discrete time steps. The core idea is the **Markov Property** (memorylessness):

> The probability of the next state depends **only on the current state**, not on the history of how you got there.

$$P(X_{t+1} = j \mid X_t = i, X_{t-1}, \ldots, X_0) = P(X_{t+1} = j \mid X_t = i)$$

### Key Components
| Concept | Meaning |
|---|---|
| **State space** | The set of all possible states (e.g., Bull, Bear, Sideways) |
| **Transition matrix P** | P[i][j] = probability of going from state i to state j. Each row sums to 1. |
| **Initial state** | The state the chain starts in at time t=0 |
| **Stationary distribution π** | The long-run fraction of time spent in each state. Satisfies **πP = π** |

---

## How the Steady-State Distribution Works

After running the chain for a long time, the fraction of time spent in each state converges to a fixed distribution **π**, regardless of the starting state. This is the **stationary** or **steady-state** distribution.

**How to compute it:**
- Solve **πP = π** subject to **Σπᵢ = 1**
- Equivalently, find the left eigenvector of P for eigenvalue 1
- Or use **power iteration**: start with any distribution and keep multiplying by P

---

## How This Applies to the Assignment

In your assignment:
- **States** = Bull (0), Bear (1), Sideways (2)
- **Transition matrix** tells you the probability of tomorrow's regime given today's
- You **simulate** by randomly sampling the next state using `np.random.choice` with the row probabilities
- **Frequency analysis** counts how often each state appears and compares to the theoretical π
- **Portfolio simulation** links each state to a daily return and compounds it

---

## Resources to Read

### Markov Chains (Beginner-Friendly)
1. **Setosa.io — Markov Chains Visual Explanation**
   - https://setosa.io/ev/markov-chains/
   - *Interactive visual explainer — best starting point*

2. **3Blue1Brown-style intuition (StatQuest)**
   - https://www.youtube.com/watch?v=i3AkTO9HLXo
   - *YouTube video: "Markov Chains Clearly Explained"*

3. **Brilliant.org — Markov Chains**
   - https://brilliant.org/wiki/markov-chains/
   - *Concise article with worked examples*

4. **Khan Academy — Probability and Markov Chains**
   - https://www.khanacademy.org/computing/computer-science/informationtheory/moderninfotheory/v/markov_chains
   - *Video lectures with exercises*

### Transition Matrices & Eigenvectors
5. **MIT OpenCourseWare 18.06 — Markov Matrices (Gilbert Strang)**
   - https://www.youtube.com/watch?v=IFqnEBWYsJk
   - *Professor Strang explains why eigenvalue=1 gives the steady state*

6. **Seeing Theory — Markov Chains (Brown University)**
   - https://seeing-theory.brown.edu/compound-probability/index.html#section3
   - *Beautiful interactive probability visualizations*

### PageRank & Markov Chains
7. **The Original PageRank Paper (simplified)**
   - http://infolab.stanford.edu/~backrub/google.html
   - *Brin & Page's original Stanford paper*

8. **3Blue1Brown — PageRank (Essence of Linear Algebra context)**
   - https://www.youtube.com/watch?v=JGQe4kiPnrU
   - *Visual explanation of eigenvectors → PageRank connection*

9. **Wikipedia — PageRank**
   - https://en.wikipedia.org/wiki/PageRank
   - *Good reference for the damping factor and Markov Chain formulation*

### Python Implementation References
10. **NumPy documentation — `np.random.choice`**
    - https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
    - *How to sample from a discrete probability distribution*

11. **NumPy documentation — `np.linalg.eig`**
    - https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
    - *How to compute eigenvalues/eigenvectors (used for steady state)*

---

## Quick Cheat Sheet

```python
import numpy as np

# Transition matrix
P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.5, 0.2],
              [0.4, 0.3, 0.3]])

# Simulate one step from state i
next_state = np.random.choice(3, p=P[current_state])

# Find steady-state distribution
eigenvalues, eigenvectors = np.linalg.eig(P.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi = eigenvectors[:, idx].real
pi = pi / pi.sum()  # normalize

# Portfolio update
V_new = V_old * (1 + daily_return[state])
```

> [!TIP]
> Start by reading the **Setosa.io visual explainer** (#1) — it will give you an intuitive feel for everything before you touch the math.
