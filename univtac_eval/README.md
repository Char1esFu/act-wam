# Evaluating in UniVTAC

The UniVTAC ACT deploy (`policy/ACT/deploy_policy.py` of the UniVTAC repo) builds the model with UniVTAC's
own `act_policy.py` and `detr/models/detr_vae.py`. Upstream, those cannot build the model this repo trains
(the future-tactile head and its EMA target encoder) or handle its extra output. This directory holds the
three files to put over `policy/ACT` of UniVTAC before evaluating:

| File | Change against UniVTAC `main` (054528f) |
|---|---|
| `act_policy.py` | WAM tactile head: extra model output, tactile loss and EMA update (the same file as `../act_policy.py`) |
| `detr/models/detr_vae.py` | WAM tactile head and target encoder (the same file as `../detr/models/detr_vae.py`) |
| `deploy_policy.py` | feeds the wrist camera whenever the train config lists `cam_wrist` (also for `camera_type: head` tasks) |

`univtac_main.diff` is the same change as a diff against UniVTAC `main`.

UniVTAC's `detr/models/backbone.py` stays as it is: unlike this repo's, it only warns when `tactile_ckpt`
is missing, and at evaluation the tactile encoder weights come from the checkpoint anyway.

```bash
cp univtac_eval/act_policy.py univtac_eval/deploy_policy.py <UniVTAC>/policy/ACT/
cp univtac_eval/detr/models/detr_vae.py <UniVTAC>/policy/ACT/detr/models/
# or: git -C <UniVTAC> apply <act-wam>/univtac_eval/univtac_main.diff
```

Then place the checkpoint and its config as described in the main README (Evaluate).
