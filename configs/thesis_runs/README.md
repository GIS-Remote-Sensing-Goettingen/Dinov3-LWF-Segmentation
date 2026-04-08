# Thesis Run Configs

These standalone YAMLs are thesis-specific HPC run profiles derived from
`configs/config_hpc.yml`.

They lock the split paths to `splits/thesis_geo_v1/train.yml`,
`splits/thesis_geo_v1/val.yml`, and `splits/thesis_geo_v1/holdout.txt`,
isolate per-run weights and logs, and disable expensive plots plus inference
for metric-only reruns unless noted otherwise.

Important runtime note:

- Generate `train.txt` and `val.txt` on the cluster host where the cached
  training tiles are mounted.
- Prior completed runs in the cluster checkout resolved the effective training
  cache to
  `/mnt/ceph-hdd/projects/mthesis_davide_mattioli/processed/folder_2_2/tiles_1024_nofeat_ps16_drop_partial_labelgrid`.
- Build the train and validation manifests from that cache directory, using
  either exact tile stems or source-scene stems and keeping full source scenes
  disjoint across train and validation.
- `holdout.txt` is staged for a later geographic holdout. It is not on the
  critical path for the current thesis rerun pack because every `R1` to `R11`
  config keeps `inference.enable: false`.

Recommended submission order:

1. `R1_topo_split_s1337.yml`
2. `R2_nanofapm_split_s1337.yml`
3. `R3_unet_split_s1337.yml`
4. `R4_maskformer_split_s1337.yml`
5. `R6_topo_split_s2027.yml`
6. `R7_topo_split_s3407.yml`
7. `R8_nanofapm_split_s2027.yml`
8. `R9_nanofapm_split_s3407.yml`
9. `R5_denseprobe_split_s1337.yml`
10. `R10_topo_no_boundary_s1337.yml`
11. `R11_topo_no_topology_s1337.yml`

Submission commands:

```bash
CONFIG_PATH=configs/thesis_runs/R1_topo_split_s1337.yml sbatch segmentation.sh
CONFIG_PATH=configs/thesis_runs/R2_nanofapm_split_s1337.yml sbatch segmentation.sh
CONFIG_PATH=configs/thesis_runs/R3_unet_split_s1337.yml sbatch segmentation.sh
CONFIG_PATH=configs/thesis_runs/R4_maskformer_split_s1337.yml sbatch segmentation.sh
CONFIG_PATH=configs/thesis_runs/R6_topo_split_s2027.yml sbatch segmentation.sh
CONFIG_PATH=configs/thesis_runs/R7_topo_split_s3407.yml sbatch segmentation.sh
CONFIG_PATH=configs/thesis_runs/R8_nanofapm_split_s2027.yml sbatch segmentation.sh
CONFIG_PATH=configs/thesis_runs/R9_nanofapm_split_s3407.yml sbatch segmentation.sh
CONFIG_PATH=configs/thesis_runs/R5_denseprobe_split_s1337.yml sbatch segmentation.sh
CONFIG_PATH=configs/thesis_runs/R10_topo_no_boundary_s1337.yml sbatch segmentation.sh
CONFIG_PATH=configs/thesis_runs/R11_topo_no_topology_s1337.yml sbatch segmentation.sh
```
