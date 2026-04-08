# Thesis Geographic Split v1

Populate these manifests on the cluster host before launching the thesis runs:

- `train.yml`: YAML list or mapping containing training scene or tile stems.
- `val.yml`: YAML list or mapping containing validation scene or tile stems.
- `holdout.txt`: one source GeoTIFF path per line for final holdout inference.

Current cluster path for manifest generation:

- Enumerate cached training tiles from
  `/mnt/ceph-hdd/projects/mthesis_davide_mattioli/processed/folder_2_2/tiles_1024_nofeat_ps16_drop_partial_labelgrid`.
- Write either exact `.pt` stems or full source-scene stems such as
  `dop20_592000_5982000_1km_20cm`, not full paths.
- Keep entire source scenes together when assigning stems to `train.yml` and
  `val.yml`. The pipeline strips `_y..._x...` suffixes before checking for
  train/validation leakage.

Important constraints:

- `train.yml` and `val.yml` must be source-disjoint. The pipeline will fail if
  source groups overlap.
- `holdout.txt` should point to geographically separated scenes reserved for
  final reporting only.
- The tile-list parser does not ignore comments, so use plain newline-delimited
  entries only.
- `holdout.txt` may stay empty until a true geographic holdout is frozen. The
  current thesis rerun pack keeps inference disabled.
