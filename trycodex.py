import numpy as np

# trycodex.py
# GitHub Copilot
#
# Small self-contained agent-based model (ABM) of panic buying dynamics.
# Uses only NumPy and matplotlib. Run: python trycodex.py

# === TODO [Copilot]: Refactor to a clean, reusable structure ===
# Requirements:
# 1) Create dataclasses or simple classes: Agent, Store, and a Simulation wrapper.
#    - Agent: stock, susceptibility, panic (0/1), methods to decide_panic(phi, neigh_frac), desired_purchase(capacity).
#    - Store: inventory, capacity, restock(quantity) per step.
# 2) Keep pure functions where possible:
#    - sigmoid(x), neighbors_fraction(grid), run_step(sim, params) -> diagnostics
# 3) Add CLI using argparse with defaults matching current globals, but allow overriding:
#    --steps, --nx, --ny, --restock, --purchase_limit, --seed
# 4) All plots must be saved to ./figs/ (create the folder if missing) in addition to showing.
# 5) main():
#    - set seed, build Simulation, run loop, collect time series, call plotting functions.
# 6) Do not remove the existing behavior; just reorganize to match this structure.

import matplotlib.pyplot as plt

# ---------------------------
# Model configuration
# ---------------------------
SEED = 42
np.random.seed(SEED)

# Grid (agents on 2D lattice)
NX = 50
NY = 50
N_AGENTS = NX * NY

# Time
T_STEPS = 200

# Store / supply
INITIAL_STORE = 10000.0
MAX_STORE = 10000.0
RESTOCK_PER_STEP = 50.0  # replenishment each time step (could be 0)

# Agent attributes
INITIAL_STOCK_MEAN = 5.0
INITIAL_STOCK_STD = 2.0
MIN_INITIAL_STOCK = 0.0

CAPACITY = 50.0  # maximum stock an agent can hold
BASE_CONSUMPTION_PROB = 0.01  # baseline probability to consume each step
BASE_CONSUMPTION_AMOUNT = 1.0

PANIC_EXTRA_BUY = 10.0  # how many extra units a panicking agent tries to buy
# probability to remain panicked next step (so recovery prob = 1-0.95)
PANIC_PERSIST_PROB = 0.95

# Decision function parameters (logistic)
ALPHA_SCARCITY = 8.0   # weight on perceived scarcity
BETA_SOCIAL = 6.0      # weight on fraction of neighbors panicking
INTERCEPT = -4.0       # baseline bias (negative => low baseline panic)

# Heterogeneity (idiosyncratic predisposition)
SUSCEPTIBILITY_STD = 1.0  # drawn per agent as extra intercept noise

# ---------------------------
# Helper functions
# ---------------------------


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def neighbors_fraction(panic_grid):
    """
    Compute fraction of 8 neighbors that are panicking for each cell.
    Uses periodic boundary conditions (wrap-around).
    """
    # panic_grid shape NX x NY with values 0/1
    up = np.roll(panic_grid, -1, axis=0)
    down = np.roll(panic_grid, 1, axis=0)
    left = np.roll(panic_grid, -1, axis=1)
    right = np.roll(panic_grid, 1, axis=1)
    upleft = np.roll(up, -1, axis=1)
    upright = np.roll(up, 1, axis=1)
    downleft = np.roll(down, -1, axis=1)
    downright = np.roll(down, 1, axis=1)

    neighbor_sum = up + down + left + right + \
        upleft + upright + downleft + downright
    return neighbor_sum / 8.0


# ---------------------------
# Initialization
# ---------------------------

def initialize():
    # Agent stocks
    stocks = np.random.normal(
        INITIAL_STOCK_MEAN, INITIAL_STOCK_STD, size=(NX, NY))
    stocks = np.clip(stocks, MIN_INITIAL_STOCK, CAPACITY)

    # Panic state: initially no one is panicking
    panic = np.zeros((NX, NY), dtype=np.int8)

    # Individual susceptibility noise (affects intercept)
    suscept = np.random.normal(0.0, SUSCEPTIBILITY_STD, size=(NX, NY))

    # Store inventory
    store = float(INITIAL_STORE)

    return stocks, panic, suscept, store


# ---------------------------
# Simulation step
# ---------------------------

def step(stocks, panic, suscept, store):
    """
    Run one time step and return updated (stocks, panic, store) plus diagnostics.
    """
    # 1) Perceived scarcity
    perceived_scarcity = 1.0 - (store / MAX_STORE)  # 0 when full, 1 when empty
    perceived_scarcity = np.clip(perceived_scarcity, 0.0, 1.0)

    # 2) Social influence
    neigh_frac = neighbors_fraction(panic)  # fraction of neighbors panicking

    # 3) Decision to panic: logistic combining scarcity, neighbors, individual susceptibility
    # panic_prob = sigmoid(INTERCEPT + ALPHA_SCARCITY*perceived_scarcity + BETA_SOCIAL*neigh_frac + suscept)
    # Note: perceived_scarcity is scalar but we broadcast
    linear_term = INTERCEPT + ALPHA_SCARCITY * \
        perceived_scarcity + BETA_SOCIAL * neigh_frac + suscept
    panic_prob = sigmoid(linear_term)

    # Agents who are currently panicking persist with high prob
    keep_panic = (np.random.rand(NX, NY) < PANIC_PERSIST_PROB) & (panic == 1)

    # New panic assignments: if random < panic_prob OR they already keep panic
    newly_panicking = (np.random.rand(NX, NY) < panic_prob) | keep_panic
    new_panic = newly_panicking.astype(np.int8)

    # 4) Consumption and purchases
    purchases = np.zeros((NX, NY))

    # baseline consumption (agents consume small amounts regardless)
    consume_mask = (np.random.rand(NX, NY) < BASE_CONSUMPTION_PROB)
    purchases[consume_mask] += BASE_CONSUMPTION_AMOUNT

    # panicking agents attempt to buy extra
    panic_mask = (new_panic == 1)
    purchases[panic_mask] += PANIC_EXTRA_BUY

    # But agents cannot exceed capacity; compute desired purchases limited by capacity
    desired = purchases
    available_capacity = CAPACITY - stocks
    desired = np.minimum(desired, available_capacity)
    desired = np.clip(desired, 0.0, None)

    # Aggregate desired vs store inventory
    total_desired = desired.sum()
    actual_purchases = np.zeros_like(desired)

    if total_desired <= store + 1e-9:
        # everyone gets desired amount
        actual_purchases = desired
        store -= total_desired
    else:
        # ration proportionally to desired amounts
        if total_desired <= 0:
            pass
        else:
            proportion = store / total_desired
            actual_purchases = desired * proportion
            store = 0.0

    # Update stocks with purchases
    stocks = stocks + actual_purchases

    # Agents may consume from stocks as well (consumption reduces stored stock)
    # Here consumption already included as a purchase from the store, but we also model
    # that agents consume some of their own stock (e.g., use it), which reduces it.
    # We'll model that baseline consumption above is actually consumption taken from store,
    # and that agents may also use existing stock slowly:
    # very small chance to use from stock
    personal_use = np.random.rand(NX, NY) < 0.001
    stocks[personal_use] = np.maximum(0.0, stocks[personal_use] - 1.0)

    # 5) Recovery from panic: some panicking agents stop panicking probabilistically
    recovery_mask = (np.random.rand(NX, NY) < (
        1.0 - PANIC_PERSIST_PROB)) & (new_panic == 1)
    new_panic[recovery_mask] = 0

    # 6) Store restocking
    store = min(MAX_STORE, store + RESTOCK_PER_STEP)

    # Diagnostics
    frac_panicking = new_panic.mean()
    total_store = store
    total_purchased = actual_purchases.sum()

    return stocks, new_panic, store, frac_panicking, total_store, total_purchased


# ---------------------------
# Run simulation
# ---------------------------

def run_simulation(steps=T_STEPS):
    stocks, panic, suscept, store = initialize()

    times = []
    frac_panic_ts = []
    store_ts = []
    purchased_ts = []

    for t in range(steps):
        stocks, panic, store, frac_panic, total_store, total_purchased = step(
            stocks, panic, suscept, store)

        times.append(t)
        frac_panic_ts.append(frac_panic)
        store_ts.append(total_store)
        purchased_ts.append(total_purchased)

    history = {
        'times': np.array(times),
        'frac_panic': np.array(frac_panic_ts),
        'store': np.array(store_ts),
        'purchased': np.array(purchased_ts),
        'final_stocks': stocks.copy(),
        'final_panic': panic.copy()
    }
    return history


# ---------------------------
# Plotting
# ---------------------------

def plot_history(history):
    times = history['times']
    frac_panic = history['frac_panic']
    store = history['store']
    purchased = history['purchased']
    final_stocks = history['final_stocks']
    final_panic = history['final_panic']

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(times, frac_panic, color='red')
    axs[0, 0].set_title('Fraction Panicking')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Fraction')

    axs[0, 1].plot(times, store, color='blue')
    axs[0, 1].set_title('Store Inventory')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Units')

    axs[1, 0].plot(times, purchased, color='orange')
    axs[1, 0].set_title('Total Purchased per Step')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Units')

    im = axs[1, 1].imshow(final_panic, cmap='Reds', vmin=0, vmax=1)
    axs[1, 1].set_title('Final Panic Map (grid)')
    plt.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # Additional figure: histogram of final stocks
    plt.figure(figsize=(6, 4))
    plt.hist(final_stocks.ravel(), bins=30, color='gray', edgecolor='black')
    plt.title('Distribution of Agent Stocks (final)')
    plt.xlabel('Stock units')
    plt.ylabel('Number of agents')
    plt.show()


# ---------------------------
# Entry point
# ---------------------------

if __name__ == '__main__':
    history = run_simulation(T_STEPS)
    plot_history(history)
