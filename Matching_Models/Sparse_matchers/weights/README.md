# Sparse matcher weights

Place local checkpoint files here:

- `gim_lightglue_100h.ckpt` for `SP_LG_GIM`
- `minima_lightglue.pth` for `SP_LG_MINIMA`

You can also override these paths from `run_avl.py`:

```bash
--gim_lg_ckpt /path/to/gim_lightglue_100h.ckpt
--minima_lg_ckpt /path/to/minima_lightglue.pth
```

The code does not download checkpoints automatically.
