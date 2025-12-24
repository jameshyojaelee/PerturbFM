# Third-Party Compatibility Notes

This repo includes small compatibility patches to external code under `third_party/` so
reruns reproduce the same behavior with the current environments (scvi-tools 1.3.3,
lightning 2.x, etc.). If you reclone or refresh any external repos, reapply these changes:

- CPA (`third_party/cpa/`): scvi/lightning API updates (callbacks, device parsing, epoch hooks,
  pin_memory), optional `ray` import guard, and prediction shape handling.
- scGen (`third_party/scgen/`): expose `qzm`/`qzv` keys for scvi v1.3 inference.
- GEARS (`third_party/GEARS/`): safe GO-graph filtering for single-gene perts and loss fallback
  when `dict_filter` is missing entries.

These patches are for reproducibility only; upstream updates may supersede them.
