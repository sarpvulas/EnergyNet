<!-- ba649cf2-5584-45ad-8ca4-179a3a242750 35a1f005-1018-4f6f-97a1-2a41b1b35113 -->
# Barter Visualization: Lending, Returns, Stored Badges, and House Details

## What we'll add

- Barter arrows split into two visuals:
  - Green solid arrows for “used this hour” energy (eta × Es_kWh).
  - Purple solid arrows for “returned (claim) flows.”
- Stored portion markers at the buyer (consumer) node: colored badges with the seller’s house number on them.
- Click a house to open a details popup showing:
  - This hour: incoming barters breakdown (used% vs stored%) and outgoing (lent) details.
  - Outstanding toggle: live outstanding per-house claims as of the selected hour (computed from barters − returns up to that hour) for both “stored in me” and “my claims stored elsewhere.”

## Minimal simulator output addition

- Log physical claim redemptions so the viewer knows return flows and their origin battery.
  - New CSV: `claim_returns.csv` with columns: `t, owner, stored_on, buyer, qty_kWh`.
  - Written alongside existing outputs in `results/folder_*/`.

## Files to change

- `src/energynet/barter_sim/engine.py`
  - In `_sell_existing_claims(...)`, record each FIFO claim decrement as a return event `(t, owner_id, stored_on_id, buyer_id, qty)`.
- `src/energynet/barter_sim/io.py`
  - Add `write_claim_returns(out_dir, return_events)` to emit `claim_returns.csv`.
- `src/energynet/barter_sim/cli.py`
  - After `res = sim.run()`, call `write_claim_returns(...)` with the collected return events.
- `src/energynet/viewer/io/filesystem.py`
  - Add `read_claim_returns_csv(results_dir) -> pd.DataFrame`.
- `src/energynet/viewer/gui/controller.py`
  - Build three hourly lists for Barters mode:
    - `barters_used_by_hour` (green arrows): `{type:"barter-used", seller, buyer, amount}`.
    - `barters_stored_markers_by_hour` (badges): `{type:"barter-stored", seller, buyer, amount}`.
    - `returns_by_hour` (purple arrows): `{type:"barter-return", seller: stored_on, buyer: buyer, amount, owner}`.
  - Pass a combined list to the plotting layer and keep markers separate in `state`.
  - Add node pick handling: clicking a house circle opens a popup with two tabs/toggles: “This hour” and “Outstanding”. Compute outstanding on-the-fly by scanning barters and returns up to the selected hour.
- `src/energynet/viewer/plotting/panels.py`
  - Update `draw_trade_network(...)` to:
    - Draw green arrows for `barter-used`.
    - Draw purple arrows (e.g., `#7e22ce`) for `barter-return`.
    - Render small colored badges near the buyer node for each `barter-stored` marker; badge color comes from seller color mapping; text is the seller id.

## Key insertion points

```315:324:src/energynet/viewer/gui/controller.py
    def on_pick(event):
        art = event.artist
        # Stats tab switching
        if hasattr(art, "_tab_index"):
            state["stats_tab_index"] = int(getattr(art, "_tab_index"))
            goto(h_var.get())
            return
        # Trade arrow clicked → detailed popup
        if hasattr(art, "_trade_info"):
```

- Extend above to also handle `if hasattr(art, "_house_id"):` → open the House Details popup with this-hour/outstanding toggles and the used vs stored breakdown.
```460:474:src/energynet/viewer/plotting/panels.py
    # Arrows: peer trades (solid green, width ∝ amount)
    max_amt_house = max((t["amount"] for t in trades_for_hour), default=0)
    for t in trades_for_hour:
        s_id, b_id, amt = t["seller"], t["buyer"], float(t["amount"])
        (x0, y0), (x1, y1) = coords[s_id], coords[b_id]
        lw = 2 * amt / max_amt_house if max_amt_house else 1.5
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="->", mutation_scale=12,
            linewidth=lw, alpha=0.8, color="#2ca02c",
            picker=6, zorder=3
        )
        arrow._trade_info = {"seller": s_id, "buyer": b_id, "amount": amt}
        ax.add_patch(arrow)
        state["arrow_patches"].append(arrow)
```

- Modify to branch on `t["type"]`:
  - `barter-used` → green (keep).
  - `barter-return` → purple color; maybe different linestyle.
  - Ignore `barter-stored` here; those render as badges at the buyer node.

## Data shapes passed to plotting

- Used (green): `{type:"barter-used", seller:int, buyer:int, amount:float}`
- Stored marker (badge): `{type:"barter-stored", seller:int, buyer:int, amount:float}`
- Return (purple): `{type:"barter-return", seller:int /*stored_on*/, buyer:int /*recipient*/, owner:int, amount:float}`

## Interaction details

- Clicking a house shows a popup:
  - This hour: lists incoming barters with `used% = eta` and `stored% = claim_kWh/Es_kWh` and outgoing (lent) events, color keyed by seller.
  - Outstanding: two tables at hour h computed as cumulative barters − cumulative returns through h.
  - A small legend shows the seller color mapping.

## Deliverables

- Visuals: green used arrows, purple return arrows, colored badges with seller ids at consumers, accurate per-house popup with this-hour vs outstanding.
- No breaking changes to “Trades” mode; Barters operate only when that view is selected.

### To-dos

- [ ] Log claim return events in engine and write claim_returns.csv
- [ ] Add reader for claim_returns.csv in filesystem.py
- [ ] Build used/stored/return hourly lists and wire into goto()
- [ ] Draw green used arrows, purple return arrows, stored badges with labels
- [ ] Implement house click popup with this-hour and outstanding toggles
- [ ] Add consistent seller→color mapping for badges/legend