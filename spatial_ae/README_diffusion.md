# Conditional diffusion with the spatial AE

This pipeline uses the updated mask-BCE spatial AE described in
[README.md](README.md). A compatible conditional
denoiser is trained on its appearance grids. The AE weights stay frozen while
gradients pass through its decoder during denoiser training and guidance.

## Files and entry points

| File | Purpose |
| --- | --- |
| `spatial_diffusion_data.py` | Parse episodes, action/state alignment and terminal boundaries |
| `spatial_ae_diffusion.py` | Prepare latent pairs, train the denoiser, evaluate initial generation |
| `spatial_guided_diffusion.py` | Run matched guidance-off/on refinement and per-pair selection |
| `train_spatial_diffusion_mask.slurm` | Prepare, train for 100 epochs, evaluate 400 test pairs |
| `run_spatial_guided_mask.slurm` | Evaluate 50-step generation and 10 refinement rounds on 400 pairs |

Run these scripts from the repository root. They use the architecture
and calibrated `constraint_checker.py` in this folder. The checker contains
the calibrated x/y coefficients used in the reported runs.

## Training data and objective

One pair associates a source frame at i with its future frame at i+2. Conditioning
includes the source grid, source and requested future poses, two observed raw
states, and actions. Raw states require x/y/theta in columns 0/1/4. The default
conversion is x = 0.1 * raw_x - 1, y = 0.15 * raw_y - 0.59, with theta in radians.
Images have shape 96 x 128 x 3. Archives contain `image` or `frame`, `lander_raw`,
and scalar discrete `action` arrays. Pairs stay within episodes and terminal
boundaries.

Preparation selects up to 400 training archives, reserves ten percent for
validation, and excludes shard_0029 from training. It caps training at 20,000
pairs and derives normalization from the training split. The first 50 archives
of shard_0029 supply up to 400 test pairs.

The denoiser trains with 1,000 diffusion timesteps, batch size 64, learning rate
2e-4, and 100 epochs. An EMA checkpoint is selected using validation loss.

```text
L_train = noise_MSE + 0.10 * shape + 0.05 * RGB
shape = Dice + 0.25 * silhouette_edge_L1
RGB = local_RGB_L1 + 0.10 * full_image_RGB_L1
```

Shape and RGB terms use decoded predictions and real training future images
at timesteps t < 200. Local RGB covers a dilated target silhouette. Requested
x/y/theta condition the denoiser and decoder.

## Generation and guidance

The evaluated dynamics model predicts the future pose from two observed states
and the intervening action. A new future appearance grid is generated with
50 DDIM steps. Decoding it gives iteration 0. Each of ten additional refinement
rounds adds noise at t=20 and denoises through 20,19,...,0.

OFF and ON share the same initial latent and per-pair noise draws. ON adjusts
the predicted clean appearance grid through the decoder with

```text
L_guidance = ((x - requested_x) / 0.06)^2
           + ((y - requested_y) / 0.10)^2
           + 0.02 * theta_axis_loss + 0.10 * shape_guidance
shape_guidance = Dice + missing_mask + 0.35 * excess_mask + 0.25 * edge_L1
```

Theta guidance aligns the unoriented axis, treating theta and theta + pi
equivalently. Evaluation uses the existing directed-heading checker. The shape
target is the observed source silhouette rotated and translated to the requested
future pose. Clipped or absent source silhouettes disable the shape term.
True future images and poses are evaluation references during sampling.

For each branch, selected output chooses among iterations 0 through 10 using
the requested-pose hard checker. Candidates satisfying all constraints rank
first, then the sum of normalized x/y/theta errors breaks ties. Future-image
IoU and true future pose do not select candidates.

## Run

Python 3.10 or newer, PyTorch, NumPy, SciPy and Matplotlib are required. Run
commands from the repository root with a configured virtual environment.

```bash
python spatial_ae/spatial_diffusion_data.py
python spatial_ae/spatial_ae_diffusion.py self-test
python spatial_ae/spatial_guided_diffusion.py --self-test
```

The launchers default to these paths relative to `PIWM_PROJECT_DIR`, which
itself defaults to the repository root.

| Artifact | Default path |
| --- | --- |
| Virtual environment | `.venv` |
| AE | `checkpoints/spatial_ae/best.pt` |
| Denoiser | `checkpoints/spatial_diffusion/model/best.pt` |
| Cache | `checkpoints/spatial_diffusion/data` |
| Dynamics weights | `dynamic/checkpoints/dynamics_discrete/best.pt` |
| Trajectories | `dynamic/data/lunar/lunar_discrete/train` |

Set `PIWM_PROJECT_DIR`, `PIWM_CHECKPOINT_ROOT`, `PIWM_VENV`, or `PIWM_REPO_DIR`
to configure the layout. Override `PIWM_AE_CHECKPOINT`, `PIWM_DIFFUSION_CACHE`,
`PIWM_DIFFUSION_CHECKPOINT`, and `PIWM_DYNAMICS_CHECKPOINT` for another compatible
artifact set. All artifact paths passed as overrides should be absolute.

On a Slurm cluster, load the Python module needed by your virtual environment,
then submit from the repository root with your account and partition options.

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  spatial_ae/run_spatial_guided_mask.slurm
```

The refinement launcher evaluates 400 pairs by default. Set `PIWM_GUIDED_PAIRS=40`
for a smaller evaluation subset. The dynamics loader prefers
`old_discrete_dynamics.py` under the configured project when available and
supports `dynamic/lunar_discrete.py` in a repository checkout. It logs the
implementation used. Reproducing a run requires matching dynamics code and weights.

To prepare data and train a new denoiser, set `PIWM_AE_CHECKPOINT` to the AE
you want to use, then submit the training launcher. `PIWM_TRAJECTORY_ROOT` can
override the trajectory directory.

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  --export=ALL,PIWM_AE_CHECKPOINT=/absolute/path/to/ae/best.pt \
  spatial_ae/train_spatial_diffusion_mask.slurm
```

After training, set the refinement launcher's cache and denoiser paths to the
new job's `data` and `model/best.pt`. AE, cache and denoiser hashes are checked
for compatibility. Model weights, latent caches and datasets are external inputs.

Both launchers print the summary and image ZIP paths. Refinement saves
`config.json` and `measurements.json` with per-pair results, model hashes,
requested-pose scores and true-future scores. Joint x/y thresholds are 0.06 and
0.10, and theta must be within 0.30 radians. Missing generated detections count
as failures. Logs are written in the submission directory as
`spatial_guided_mask_JOBID.log` and `.err`.
