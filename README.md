# ACT-WAM

ACT (Action Chunking with Transformers) with visuo-tactile input, trained on UniVTAC demonstrations, and
an optional world action model head: ACT additionally predicts the tactile latents of future frames.

This repository covers training only: data conversion, the model and the configs. Evaluation runs in the
UniVTAC simulator, whose ACT deploy wrapper loads the checkpoints written here.

## Model

ACT as in UniVTAC: a ResNet18 per camera (ImageNet, fine-tuned), the UniVTAC tactile ResNet18 per tactile
stream (pretrained on simulated contact, fine-tuned), a transformer encoder-decoder and a CVAE over the
50-action chunk, trained with L1 + KL.

World action model (`wam_enabled: true`):

- The decoder gets one extra query per future frame and tactile stream (default 6 frames, t+5 ... t+30,
  2 streams: 12 queries). A linear head regresses the 512-dim UniVTAC latent of that frame.
- The decoder self-attention is bidirectional, so the action queries attend to the predicted future touch.
- Targets: the future tactile images encoded by an EMA copy of the fine-tuned tactile backbone
  (`wam_tactile_ema_decay`), layer-normalized. A target encoder trained by its own prediction loss could
  collapse to a constant.
- Loss: ACT's loss + `wam_tactile_loss_weight` x L1 of the predicted latents; frames past the episode end
  are left out.
- Inference only uses the action queries; the extra queries cost a few tokens per forward pass.

## Install

```bash
uv venv .venv --python 3.12
VIRTUAL_ENV=$PWD/.venv uv pip install --torch-backend=cu128 torch torchvision
VIRTUAL_ENV=$PWD/.venv uv pip install -r requirements.txt
```

The tactile encoder checkpoint (UniVTAC `encoder/checkpoints/resnet18/20251128-125750/best.pth`) goes to
`checkpoints/univtac_resnet18_20251128-125750.pth`. Training stops if `tactile_ckpt` is missing.

## Data

The raw UniVTAC episodes (`<raw root>/<task>/<config>/<episode>.hdf5`) are converted into ACT episodes
(head / wrist cameras, two marker tactile images, all 256x256; joint state and action):

```bash
python process_data.py put_bottle_in_shelf clean 100 --raw-root /path/to/UniVTAC/data
```

This writes `data/sim-put_bottle_in_shelf/clean-100/` and registers it in `SIM_TASK_CONFIGS.json`.
The cameras follow `task_settings.json` (`camera_type`: `head`, or `all` = head + wrist, as chosen per task
by the UniVTAC authors); `--visual-cameras head wrist` exports the wrist camera too, which the `*_all*`
configs need.

## Train

Run from the repo root. `<config>.yml` holds every hyperparameter:

| Config | Cameras | Tactile | Steps |
|---|---|---|---|
| `train_config` | head | UniVTAC encoder, fine-tuned | 4000 |
| `train_config_all` | head + wrist | UniVTAC encoder, fine-tuned | 4000 |
| `train_config_wam` | head | same, + future tactile latent prediction | 20000 |
| `train_config_all_wam` | head + wrist | same, + future tactile latent prediction | 20000 |
| `train_config_freeze`, `_scrach`, `_tactile_full`, `_vision`, `_vision_all` | | UniVTAC ablations | 4000 |

```bash
bash train.sh put_bottle_in_shelf clean 100 0 0 train_config_wam   # task config num seed gpu train_config
```

Checkpoints go to `act_ckpt/act-<task>/<config>-<num>/<train_config>/` (`policy_last.ckpt`,
`dataset_stats.pkl`); TensorBoard logs to its `tb/`. With `WANDB_PROJECT` set, the metrics also go to
wandb (`WANDB_ENTITY`, `WANDB_MODE`, `WANDB_NAME`).

The state and action are the first `state_dim` (8) joint dimensions: 7 arm joints and one gripper finger;
the converted data holds both fingers.

### NHR cluster (alex)

`train_cluster.sbatch` converts the task if needed and trains on one A40, with the data, checkpoints and
virtualenv on the NVMe workspace (`$WS/act/{data,ckpt,venv}`) and wandb online through the proxy:

```bash
sbatch --export=NONE,TRAIN_CONFIG=train_config_wam train_cluster.sbatch put_bottle_in_shelf
```

## Evaluate

Copy (or link) the checkpoint directory to `policy/ACT/act_ckpt/act-<task>/<config>-<num>/<train_config>/`
of UniVTAC and run its ACT deploy with `TRAIN_CONFIG=<train_config>`. The deploy builds the model from the
same `<train_config>.yml`, so the WAM keys must be in its copy too.

## License

ACT and DETR: see `LICENSE` and `detr/LICENSE`.
