# MODELS

## Goal
This document explains:
- how DINO layers are extracted and passed into segmentation heads,
- the exact formulas used for feature conversion,
- how FAPM works in both the classic and Nano variants.

## Notation
- `I`: input image tensor, shape `(B, 3, H, W)`
- `p`: DINO patch size (`14` or `16`, depending on backbone)
- `L = [l0, l1, ..., lk-1]`: selected DINO layer indices from `model.layers`
- `R`: number of register tokens (`backbone.config.num_register_tokens`)
- `C`: DINO channel width (`model.dino_channels`)

## 1) From DINO hidden states to feature maps

Code paths:
- single tile: `utils/data.py` -> `extract_multiscale_features(...)`
- batch: `pipeline/train_utils.py` -> `extract_multiscale_features_batch(...)`

### Step 1: run backbone and collect hidden states
The backbone is called with `output_hidden_states=True`:

`H^(l) in R^(B x (1 + R + N) x C)`

for each selected transformer layer `l`.

- `1` is the CLS token.
- `R` are optional register tokens.
- `N` are patch tokens.

### Step 2: remove CLS/register tokens

`T^(l) = H^(l)[:, 1 + R :, :]`

Now:

`T^(l) in R^(B x N x C)`

### Step 3: rebuild 2D patch grid
If processed image size is `(H_proc, W_proc)`, then:

`G_h = H_proc // p`
`G_w = W_proc // p`
`N = G_h * G_w`

Reshape and permute:

`F^(l) = permute(reshape(T^(l), B, G_h, G_w, C), 0, 3, 1, 2)`

So:

`F^(l) in R^(B x C x G_h x G_w)`

### Step 4: pass features into the head
For `model.layers = [5, 11, 17, 23]`, the code builds:

`features = [F^(5), F^(11), F^(17), F^(23)]`

Heads then use fixed index roles, for example in `models/unet_nano_fapm.py`:
- `features[0]` -> shallow path
- `features[1]` -> mid path 1
- `features[2]` -> mid path 2
- `features[3]` -> deep path

Important: ViT layers keep the same patch-grid resolution. The "multiscale" effect
comes from layer depth (semantic level), while decoder up/down sampling creates the
spatial pyramid behavior.

## 2) Classic FAPM (channel-attention projection)

Code path:
- `models/unet_v2.py` -> `FidelityAwareProjection`

Given input feature map `x`:

1. 1x1 projection + normalization:

`z = BN(Conv1x1(x))`

2. Global channel descriptor:

`q = GAP(z)`

3. Two-layer squeeze/excitation gate:

`a = sigmoid(W2(ReLU(W1(q))))`

4. Channel-wise modulation:

`y = z * a`

This keeps spatial structure and reweights channels by global context.

## 3) NanoFAPM (split-and-modulate)

Code path:
- `models/unet_nano_fapm.py` -> `NanoFAPM`

Given input `x`, NanoFAPM separates "specific" and "context" branches:

1. Specific branch:

`s = GELU(GN(W_s * x))`

2. Low-rank context branch:

`c = W_c * x`

3. FiLM-style modulation parameters:

`[gamma, beta] = split(sigmoid(W_m(GAP(c))))`

4. Modulation:

`m = s * (1 + gamma) + beta`

5. Depthwise separable refinement:

`r = GN(PW(GELU(GN(DW(m)))))`
`g = sigmoid(r)`

6. Residual output:

`shortcut = x if channels match else W_sc * x`
`y = g * m + shortcut`

Interpretation:
- The specific path preserves discriminative channels.
- The low-rank context path injects scene-level conditioning.
- The gate and residual keep the projection stable and lightweight.

## 4) Practical config hook

You control which DINO blocks are used through:

`model.layers` in `config.example.yml` (and local/hpc configs).

Example:

```yaml
model:
  backbone: facebook/dinov3-vitl16-pretrain-sat493m
  layers: [5, 11, 17, 23]
  dino_channels: 1024
  head: unet_nano_fapm
```

## 5) Head parameter counts (KB snapshot)

Counts below were computed from the current registry (`models.available_heads`)
using:
- `num_classes = 2`
- `dino_channels = 1024`
- `model.layers = [5, 11, 17, 23]`
- current default head options from `models/__init__.py` and `config*.yml`

| head | total params | trainable params | frozen params |
| --- | ---: | ---: | ---: |
| `dino_dense_probe` | 4,098 | 4,098 | 0 |
| `dino_segdino_light` | 591,362 | 591,362 | 0 |
| `maskformer` | 4,768,512 | 4,768,512 | 0 |
| `unet` | 13,512,994 | 13,512,994 | 0 |
| `unet_lite` | 778,500 | 778,500 | 0 |
| `unet_lite_plus` | 921,189 | 921,189 | 0 |
| `unet_nano` | 530,132 | 530,132 | 0 |
| `unet_nano_fapm` | 761,893 | 761,893 | 0 |
| `unet_topo_fusion` | 600,317 | 534,781 | 65,536 |
| `unet_v2` | 9,142,308 | 9,142,308 | 0 |

## 6) `unet_topo_fusion` formulas

Code path:
- `models/unet_topo_fusion.py`

### Backbone-aligned patch grid
For DINOv3-SAT (`vitl16`):
- `p = 16`
- `R = 4` register tokens

Effective size used for tokenization:

`H_eff = floor(H / p) * p`
`W_eff = floor(W / p) * p`

`G_h = H_eff / p`
`G_w = W_eff / p`

### Learned layer fusion
Project each selected layer feature `F_i` to a shared width and compute
spatially varying weights:

`alpha_i(x,y) = softmax_i(score(F_i)(x,y))`

`F_fused(x,y) = sum_i alpha_i(x,y) * F_i(x,y)`

### LoRA-style projection adapter
Each 1x1 projection uses:

`Y = W_0 * X + (alpha/r) * (B * (A * X))`

where `A` and `B` are low-rank 1x1 convolutions (`rank = r`).

### Boundary gating refinement
Boundary stream outputs gate map `g = sigmoid(gate_conv(E))`.
Decoder features are modulated before final mask logits:

`X_refined = X * (1 + s * g)`

with default gate scale `s = 0.1` and optional clamp to `[1, 1+s]`.

### Topology supervision
Foreground probability for class `k`:

`P_fg = softmax(logits)[:, k]` (multiclass)

Soft-clDice term:

`L_cldice = 1 - clDice(P_fg, Y_fg)`

Skeleton branch term:

`L_skel = BCEWithLogits(S_logits, soft_skeletonize(Y_fg))`

Combined objective contribution:

`L_topo = lambda_topo * L_cldice + lambda_skel * L_skel`

By default, topology terms are computed on aux-resolution outputs for stability
and lower compute cost.
