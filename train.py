import torch
from torch.utils.data import DataLoader

import accelerate
import time, os, random
import numpy as np
from tqdm import tqdm
import gc  # garbage collection

from internal import configs
from internal import train_utils
from internal import utils
from internal import datasets
from internal import checkpoints

from absl import app

TIME_PRECISION = 1000


def main():
    # configs
    # rng = random.PRNGKey(20200823)
    # rng = torch.manual_seed(20200823) # possible to skip this to generate random nums
    config = configs.load_config()

    # accelerator for DDP - Distributed Data Parallalism
    accelerator = accelerate.Accelerator()
    # accelerator configs

    # load Dataset
    dataset = datasets.load_dataset("train", config.data_dir, config)
    test_dataset = datasets.load_dataset("test", config.data_dir, config)

    dataloader = DataLoader(
        np.arange(len(dataset)),
        num_workers=8,
        shuffle=True,
        batch_size=1,
        collate_fn=dataset.collate_fn,
        persist_workers=True,
    )

    test_dataloader = DataLoader(
        np.arange(len(test_dataset)),
        num_workers=4,
        shuffle=False,
        batch_size=1,
        collate_fn=dataset.collate_fn,
    )

    # camera poses
    to_array = (
        lambda x: x if isinstance(x, np.ndarray) else x
    )  # probably dont need this
    cameras = tuple(to_array(x) for x in dataset.cameras)

    # if: to check for rawnerf mode

    # rng, key = random.split(rng)

    # model & optimizer
    # model = models.Model(config=config)

    setup = train_utils.setup_model(config, dataset=dataset)
    model, state, render_eval_pfn, train_pstep, lr_fn = setup
    # HERE STATE IS THE OPTIMIZER

    model, state = accelerator.prepare(model, state)
    model.eval()
    module = accelerator.unwrap_model(model)

    dataiter = iter(dataloader)
    test_dataiter = iter(test_dataloader)

    # multi-nerf pytorch
    num_params = train_utils.tree_len(list(model.parameters()))
    accelerator.print(f"Number of parameters being optimized {num_params}")

    # standard
    if dataset.size > module.num_glo_embeddings and module.num_glo_features > 0:
        raise ValueError(
            f"Number of glo embeddings {module.num_glo_embeddings}"
            f"ust be atleast equal to num of train images"
            f"{dataset.size}"
        )

    # metric handler
    # metric_harness = image.MetricHarness()

    # CHECKPOINTS
    # sea-thru
    state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
    init_step = state.step + 1
    # state = flas.jax_utils.replicate(state)

    #  multi-nerf pytorch
    # init_step = checkpoints.restore_checkpoints(config,exp_path, model, optimizer) + 1

    accelerator.print("Begin training...")
    total_time = 0
    total_steps = 0
    reset_stats = True
    if config.early_exit_steps is not None:
        num_steps = config.early_exit_steps
    else:
        num_steps = config.max_steps

    if accelerator.is_local_main_process:
        tbar = tqdm(range(init_step, num_steps + 1))
    else:
        tbar = range(init_step, num_steps + 1)

    for step in tbar:
        try:
            batch = next(dataloader)
        except StopIteration:
            dataiter = iter(dataloader)
            batch = next(dataiter)

        batch = accelerate.utils.send_to_device(batch, accelerator.device)
        if reset_stats and accelerator.is_local_main_process:
            stats_buffer = []
            train_start_time = time.time()
            reset_stats = False

        learning_rate = lr_fn(step)

        # sea-thru
        if step >= config.uw_decay_acc:
            sig_mult = config.uw_final_acc_trans_loss_mult
            bs_mult = config.uw_final_acc_weights_loss_mult
        else:
            sig_mult = config.uw_initial_acc_trans_loss_mult
            bs_mult = config.uw_initial_acc_weights_loss_mult

        # fraction of training period
        train_frac = np.clip((step - 1) / (config.max_steps - 1), 0, 1)

        ## sea-thru - with pmap
        with accelerate.autocast():
            state, stats, rngs = train_pstep(
                # rngs,
                state,
                batch,
                cameras,
                train_frac,
                bs_mult,
                sig_mult,
            )

        # disable garbage collection
        if step % config.gc_every == 0:
            gc.collect()

        # multinerf-pytorch
        # losses = {}
        # data_loss, stats = train_utils.compute_data_loss(batch, renderings, config)

        # Log training sumaries
        if accelerator.is_local_main_process:
            stats_buffer.append(stats)
            if step == init_step or step % config.print_every == 0:
                elapsed_time = time.time()
                steps_per_sec = config.print_every / elapsed_time
                rays_per_sec = config.batch_size * steps_per_sec

                # approx of total time
                total_time += int(round(TIME_PRECISION * elapsed_time))
                total_steps += config.print_every
                approx_total_time = int(round(step * total_time / total_steps))

                # Transpose and stack stats_buffer along axis 0

                fs = [utils.flatten_dict(s, sep="/") for s in stats_buffer]
                stats_stacked = {k: np.stack([f[k] for f in fs]) for k in fs[0].keys()}

                # Split statistic thats not a vector, into a set of statistics
                stats_split = {}
                for k, v in stats_stacked.items():
                    if v.ndim not in [1, 2] and v.shape[0] != len(stats_buffer):
                        raise ValueError("statistics must be of size [n], or [n,k].")
                    if v.ndim == 1:
                        stats_split[k] = v
                    elif v.ndim == 2:
                        for i, vi in enumerate(tuple(v.T)):
                            stats_split[f"{k}/{i}"] = vi

                # SUMMARY WRITER N STUFF

                # Take the mean and max of each statistic since the last summary.
                avg_stats = {k: np.mean(v) for k, v in stats_split.items()}
                max_stats = {k: np.max(v) for k, v in stats_split.items()}

                precision = int(np.ceil(np.log10(config.max_steps))) + 1
                avg_loss = avg_stats["loss"]
                avg_psnr = avg_stats["psnr"]
                str_losses = {  # Grab each "losses_{x}" field and print it as "x[:4]".
                    k[7:11]: (f"{v:0.5f}" if v >= 1e-4 and v < 10 else f"{v:0.1e}")
                    for k, v in avg_stats.items()
                    if k.startswith("losses/")
                }
                tbar.write(
                    f"{step:{precision}d}"
                    + f"/{config.max_steps:d}: "
                    + f"loss={avg_loss:0.5f}, "
                    + f"psnr={avg_psnr:6.3f}, "
                    + f"lr={learning_rate:0.2e} | "
                    + ", ".join([f"{k}={s}" for k, s in str_losses.items()])
                    + f", {rays_per_sec:0.0f} r/s"
                )

                # Reset everything we are tracking between summarizations.
                reset_stats = True

            if step>0 and step%config.checkpoint_every==0:
                state_to_save = state # recheck - replicate vs unreplicate
                checkpoints.save_checkpoint(
                    config.checkpoint_dir, state_to_save, int(step), keep=100
                )

        # Test-set Evaluation

            

    pass


if __name__ == "__main__":
    app.run(main)
