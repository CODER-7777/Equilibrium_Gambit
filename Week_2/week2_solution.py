import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

np.random.seed(42)
random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATE_NAMES = {0: "Bull", 1: "Bear", 2: "Sideways"}
STATE_COLORS = {0: "#2ecc71", 1: "#e74c3c", 2: "#f1c40f"}


P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.5, 0.2],
    [0.4, 0.3, 0.3],
])

print("Transition Matrix P:")
print(pd.DataFrame(P, index=["Bull","Bear","Sideways"], columns=["Bull","Bear","Sideways"]))
print()


NUM_DAYS = 500
INITIAL_STATE = 0  

def simulate_markov_chain(trans_mat, init_state, n_steps):
    """Simulate a discrete-time Markov Chain for n_steps."""
    n_states = trans_mat.shape[0]
    states = [init_state]
    cur = init_state
    for _ in range(n_steps):
        cur = np.random.choice(n_states, p=trans_mat[cur])
        states.append(cur)
    return states

states = simulate_markov_chain(P, INITIAL_STATE, NUM_DAYS)
print(f"Simulated {NUM_DAYS} days starting from {STATE_NAMES[INITIAL_STATE]}")
print(f"First 20 states: {[STATE_NAMES[s] for s in states[:20]]}\n")

# ==== TASK 1 PART 2: Frequency Analysis ====
state_counts = {s: 0 for s in range(3)}
for s in states:
    state_counts[s] += 1
total = len(states)
emp_frac = {s: state_counts[s]/total for s in range(3)}

# Steady-state: solve pi @ P = pi  =>  P^T @ pi^T = pi^T (eigenvector for eigenvalue 1)
eigenvalues, eigenvectors = np.linalg.eig(P.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi = eigenvectors[:, idx].real
pi = pi / pi.sum()

print("Frequency Analysis:")
print(f"{'State':>10} | {'Count':>6} | {'Simulated':>10} | {'Theoretical':>11}")
print("-" * 50)
for s in range(3):
    print(f"{STATE_NAMES[s]:>10} | {state_counts[s]:>6} | {emp_frac[s]:>10.4f} | {pi[s]:>11.4f}")
print()

# ==== TASK 1 PART 3: Visualization ====
days = np.arange(len(states))

# Plot 1: Market State vs Time
fig, ax = plt.subplots(figsize=(14, 4))
for s in range(3):
    mask = np.array(states) == s
    ax.scatter(days[mask], np.array(states)[mask], c=STATE_COLORS[s],
               label=STATE_NAMES[s], s=8, alpha=0.7, edgecolors='none')
ax.set_xlabel("Day"); ax.set_ylabel("Market State")
ax.set_title("Market State vs Time (500-Day Markov Chain Simulation)")
ax.set_yticks([0,1,2]); ax.set_yticklabels(["Bull","Bear","Sideways"])
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_market_state_vs_time.png"), dpi=150)
plt.close()
print("Saved: plots/1_market_state_vs_time.png")

# Plot 2: Histogram of State Frequencies
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(3); w = 0.35
ax.bar(x - w/2, [emp_frac[s] for s in range(3)], w, label="Simulated",
       color=["#2ecc71","#e74c3c","#f1c40f"], edgecolor="black")
ax.bar(x + w/2, [pi[s] for s in range(3)], w, label="Theoretical",
       color=["#27ae60","#c0392b","#f39c12"], edgecolor="black", alpha=0.7)
ax.set_xlabel("Market State"); ax.set_ylabel("Fraction")
ax.set_title("State Frequency: Simulated vs Theoretical")
ax.set_xticks(x); ax.set_xticklabels(["Bull","Bear","Sideways"])
ax.legend(); ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_state_frequency_histogram.png"), dpi=150)
plt.close()
print("Saved: plots/2_state_frequency_histogram.png")

# Plot 3: Running Probability Estimates Over Time
fig, ax = plt.subplots(figsize=(14, 5))
running_counts = np.zeros((len(states), 3))
for t, s in enumerate(states):
    if t > 0:
        running_counts[t] = running_counts[t-1]
    running_counts[t, s] += 1
running_frac = running_counts / np.arange(1, len(states)+1).reshape(-1,1)

for s in range(3):
    ax.plot(days, running_frac[:, s], color=STATE_COLORS[s],
            label=f"{STATE_NAMES[s]} (sim)", linewidth=1.5)
    ax.axhline(y=pi[s], color=STATE_COLORS[s], linestyle='--', alpha=0.6,
               label=f"{STATE_NAMES[s]} (π={pi[s]:.3f})")
ax.set_xlabel("Day"); ax.set_ylabel("Running Probability")
ax.set_title("Convergence of Running Probabilities to Steady-State")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3); ax.set_xlim(0, NUM_DAYS); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_running_probability.png"), dpi=150)
plt.close()
print("Saved: plots/3_running_probability.png")

# ==== TASK 2: Portfolio Simulation ====
DAILY_RETURNS = {0: 0.01, 1: -0.012, 2: 0.001}
INITIAL_CAPITAL = 100_000

portfolio = [INITIAL_CAPITAL]
for t in range(1, len(states)):
    portfolio.append(portfolio[-1] * (1 + DAILY_RETURNS[states[t]]))

final_val = portfolio[-1]
total_ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

print(f"\nPortfolio Simulation:")
print(f"  Initial Capital: Rs {INITIAL_CAPITAL:,.2f}")
print(f"  Final Value:     Rs {final_val:,.2f}")
print(f"  Total Return:    {total_ret:+.2f}%")

# Plot 4: Portfolio Value Over Time
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(days, portfolio, color="#3498db", linewidth=1.5, label="Portfolio Value")
ax.axhline(y=INITIAL_CAPITAL, color="gray", linestyle='--', alpha=0.5, label="Initial Capital")
for t in range(len(states)-1):
    ax.axvspan(t, t+1, alpha=0.08, color=STATE_COLORS[states[t]])
ax.set_xlabel("Day"); ax.set_ylabel("Portfolio Value (Rs)")
ax.set_title(f"Portfolio Evolution | Final: Rs {final_val:,.0f} ({total_ret:+.2f}%)")
ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_portfolio_evolution.png"), dpi=150)
plt.close()
print("Saved: plots/4_portfolio_evolution.png")

# ==== BONUS: PageRank Explanation ====
print("\n" + "="*60)
print("BONUS: Google PageRank & Markov Chains")
print("="*60)
print("""
Google's PageRank models a "random surfer" clicking hyperlinks. Each webpage
is a STATE in a Markov Chain, and each hyperlink defines a transition
probability (if page A has n links, prob of following any one = 1/n). The
Markov Property holds: the next page depends ONLY on the current page.

A damping factor d=0.85 is added: with prob d the surfer follows a link,
with prob (1-d) they teleport to a random page. This makes the chain
irreducible and aperiodic, guaranteeing a unique steady-state distribution.
The steady-state probability of each page IS its PageRank score — pages
visited more often by the random surfer are ranked higher.
""")

print("All tasks complete!")
