# Colonel Blotto

Castles are worth 1, 2, 3, 4, and 5. An allocation is five non-negative integers summing to 100.

- **Game 1** is head-to-head (ties split that castle).
- **Game 2** draws three opponents, gives each castle to whoever placed the most soldiers (ties split), then pays 9 / 4 / 2 / 0 by rank (tied ranks split those payouts).

## Run

```powershell
py -3 -m pip install numpy matplotlib
py -3 blotto.py
py -3 blotto.py --human
py -3 blotto.py --both
```

Flags: `--seed` (default 0), `--rounds` (default 300), `--burn-in` (default 50; forced off with `--human`), `--human`, `--both`.

Each game starts from 200 allocations and adds a best response each round (random sample, then hill-climbing by moving 1-5 soldiers). After burn-in, the initial seed rows are dropped so the pool holds only best responses. `--human` seeds multiples of 5 plus a few common patterns and leaves burn-in off. Game 2's search freezes one draw of 20,000 opponent groups per round. The script prints the most common allocations, the pool mean, a rounded mean that sums to 100, and the exploitability gap every 50 rounds.

Plots are written to `figures/`:

- `figures/blotto_<game>_<mode>.png` — mean soldiers per castle over rounds
- `figures/blotto_<game>_<mode>_gap.png` — exploitability gap over rounds

`<game>` is `game1` or `game2`. `<mode>` is `standard` or `human`.
