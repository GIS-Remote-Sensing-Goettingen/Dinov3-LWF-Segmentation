# Thesis Run Configs

These standalone YAMLs are thesis-specific HPC run profiles derived from
`configs/config_hpc.yml`.

They lock the split paths to `splits/thesis_geo_v1/train.yml`,
`splits/thesis_geo_v1/val.yml`, and `splits/thesis_geo_v1/holdout.txt`,
isolate per-run weights and logs, and disable expensive plots plus inference
for metric-only reruns unless noted otherwise.

Important runtime note:

- Prior completed runs in the cluster checkout resolved the effective training
  cache to
  `/mnt/ceph-hdd/projects/mthesis_davide_mattioli/processed/folder_2_2/tiles_1024_nofeat_ps16_drop_partial_labelgrid`.
- `train.yml` and `val.yml` are already populated with source-scene stems.
  Training expands those stems to cached tiles automatically, and directory
  inference now expands the same manifests back to raw GeoTIFF scene paths via
  `inference.input_dir`.
- `holdout.txt` is staged for a later geographic holdout. It is not on the
  critical path for the current thesis rerun pack because every `R1` to `R11`
  config keeps `inference.enable: false`.
- `R13_mask2former_semantic_split_s1337.yml` expects one manually staged local
  Hugging Face checkpoint under
  `/user/davide.mattioli/u20330/Dinov3-LWF-Segmentation/weights/hf/facebook/mask2former-swin-base-ade-semantic`
  containing `config.json`, `preprocessor_config.json`, and model weights.
- The coarse-supervision baseline `C1_topo_coarse_split_s1337.yml` points
  `paths.label_path` at
  `/mnt/vast-standard/home/davide.mattioli/u20330/planet_labels_2022.tif`,
  writes a separate coarse cache under
  `/mnt/ceph-hdd/projects/mthesis_davide_mattioli/processed/thesis_runs/C1_topo_coarse_split_s1337`,
  and restricts prepare-time tiling to the union of the explicit thesis
  train/validation scene manifests so it does not scan unrelated raw scenes.

Recommended submission order:

1. `R1_topo_split_s1337.yml`
2. `R2_nanofapm_split_s1337.yml`
3. `R3_unet_split_s1337.yml`
4. `R12_deeplabv3_split_s1337.yml`
5. `R13_mask2former_semantic_split_s1337.yml`
6. `R4_maskformer_split_s1337.yml`
7. `R6_topo_split_s2027.yml`
8. `R7_topo_split_s3407.yml`
9. `R8_nanofapm_split_s2027.yml`
10. `R9_nanofapm_split_s3407.yml`
11. `R5_denseprobe_split_s1337.yml`
12. `R10_topo_no_boundary_s1337.yml`
13. `R11_topo_no_topology_s1337.yml`
14. `C1_topo_coarse_split_s1337.yml` (coarse-label supervision baseline)

Submission commands:

```bash
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/R1_topo_split_s1337.yml segmentation.sh
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/R2_nanofapm_split_s1337.yml segmentation.sh
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/R3_unet_split_s1337.yml segmentation.sh
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/R12_deeplabv3_split_s1337.yml segmentation.sh
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/R13_mask2former_semantic_split_s1337.yml segmentation.sh
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/R4_maskformer_split_s1337.yml segmentation.sh
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/R6_topo_split_s2027.yml segmentation.sh
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/R7_topo_split_s3407.yml segmentation.sh
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/R8_nanofapm_split_s2027.yml segmentation.sh
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/R9_nanofapm_split_s3407.yml segmentation.sh
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/R5_denseprobe_split_s1337.yml segmentation.sh
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/R10_topo_no_boundary_s1337.yml segmentation.sh
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/R11_topo_no_topology_s1337.yml segmentation.sh
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/C1_topo_coarse_split_s1337.yml segmentation.sh
```

Coarse-vs-refined supervision baseline:

1. Train the refined-topology baseline on the explicit thesis split:

```bash
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/R1_topo_split_s1337.yml segmentation.sh
```

2. Train the same topology head on the coarse `planet_labels_2022.tif`
   supervision with a separate cache and checkpoint root:

```bash
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/C1_topo_coarse_split_s1337.yml segmentation.sh
```

3. Export validation-scene predictions for both checkpoints against the same
   explicit validation scene manifest:

```bash
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/E1_topo_refined_val_eval_s1337.yml segmentation.sh
sbatch --export=ALL,CONFIG_PATH=configs/thesis_runs/E2_topo_coarse_val_eval_s1337.yml segmentation.sh
```

4. Score both exported prediction rasters against the current clean evaluation
   raster `crf/final_labels_1m.tif`:

```bash
python scripts/validate_prediction_rasters.py \
  /user/davide.mattioli/u20330/Dinov3-LWF-Segmentation/crf/final_labels_1m.tif \
  /user/davide.mattioli/u20330/Dinov3-LWF-Segmentation/output/thesis_runs/E1_topo_refined_val_eval_s1337/predictions.tif \
  /user/davide.mattioli/u20330/Dinov3-LWF-Segmentation/output/thesis_runs/E2_topo_coarse_val_eval_s1337/predictions.tif
```

The headline supervision-source table should therefore compare:

- `R1_topo_split_s1337` checkpoint evaluated on `val.yml` against
  `final_labels_1m.tif`
- `C1_topo_coarse_split_s1337` checkpoint evaluated on the same `val.yml`
  scenes against the same `final_labels_1m.tif`
