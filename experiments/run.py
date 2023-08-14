import copy
import os
from argparse import Namespace

import haiku as hk
import jax.numpy as jnp
import jmp
import numpy as np
import wandb
import yaml

from experiments.utils import setup_data, setup_model
from lagrangebench import Trainer, infer
from lagrangebench.case_setup import case_builder
from lagrangebench.evaluate import averaged_metrics
from lagrangebench.utils import PushforwardConfig


def train_or_infer(args: Namespace):
    data_train, data_eval = setup_data(args)

    # neighbors search
    bounds = np.array(data_train.metadata["bounds"])
    args.box = bounds[:, 1] - bounds[:, 0]

    args.info.len_train = len(data_train)
    args.info.len_eval = len(data_eval)

    # setup core functions
    case = case_builder(
        box=args.box,
        metadata=data_train.metadata,
        input_seq_length=args.config.input_seq_length,
        isotropic_norm=args.config.isotropic_norm,
        noise_std=args.config.noise_std,
        magnitude_features=args.config.magnitudes,
        external_force_fn=data_train.external_force_fn,
        neighbor_list_backend=args.config.neighbor_list_backend,
        neighbor_list_multiplier=args.config.neighbor_list_capacity_multiplier,
        dtype=(jnp.float64 if args.config.f64 else jnp.float32),
    )

    _, particle_type = data_train[0]

    args.info.homogeneous_particles = particle_type.max() == particle_type.min()
    args.metadata = data_train.metadata
    args.normalization_stats = case.normalization_stats
    args.config.has_external_force = data_train.external_force_fn is not None

    # setup model from configs
    model, MODEL = setup_model(args)
    model = hk.without_apply_rng(hk.transform_with_state(model))

    # mixed precision training based on this reference:
    # https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/train.py
    policy = jmp.get_policy("params=float32,compute=float32,output=float32")
    hk.mixed_precision.set_policy(MODEL, policy)

    if args.config.mode == "train":
        # save config file
        run_prefix = f"{args.config.model}_{data_train.name}"
        i = 0
        while os.path.isdir(os.path.join(args.config.ckp_dir, f"{run_prefix}_{i}")):
            i += 1
        args.info.run_name = f"{run_prefix}_{i}"

        args.config.new_checkpoint = os.path.join(
            args.config.ckp_dir, args.info.run_name
        )
        os.makedirs(args.config.new_checkpoint, exist_ok=True)
        os.makedirs(os.path.join(args.config.new_checkpoint, "best"), exist_ok=True)
        with open(os.path.join(args.config.new_checkpoint, "config.yaml"), "w") as f:
            yaml.dump(vars(args.config), f)
        with open(
            os.path.join(args.config.new_checkpoint, "best", "config.yaml"), "w"
        ) as f:
            yaml.dump(vars(args.config), f)

        if args.config.wandb:
            # wandb doesn't like Namespace objects
            args_dict = copy.copy(args)
            args_dict.config = vars(args.config)
            args_dict.info = vars(args.info)

            wandb_run = wandb.init(
                project=args.config.wandb_project,
                entity=args.config.wandb_entity,
                name=args.info.run_name,
                config=args_dict,
                save_code=True,
            )
        else:
            wandb_run = None

        pf_config = PushforwardConfig(
            steps=args.config.pushforward["steps"],
            unrolls=args.config.pushforward["unrolls"],
            probs=args.config.pushforward["probs"],
        )

        trainer = Trainer(
            model,
            case,
            data_train,
            data_eval,
            metrics=args.config.metrics,
            seed=args.config.seed,
            batch_size=args.config.batch_size,
            noise_std=args.config.noise_std,
            pushforward=pf_config,
            lr_start=args.config.lr_start,
            lr_end=args.config.lr_final,
            lr_steps=args.config.lr_decay_steps,
            lr_decay_rate=args.config.lr_decay_rate,
            log_steps=args.config.log_steps,
            eval_steps=args.config.eval_steps,
        )
        trainer(
            step_max=args.config.step_max,
            load_checkpoint=args.config.model_dir,
            store_checkpoint=args.config.new_checkpoint,
            wandb_run=wandb_run,
        )
    elif args.config.mode == "infer":
        metrics = infer(
            model,
            case,
            data_eval,
            load_checkpoint=args.config.model_dir,
            metrics=args.config.metrics,
            rollout_dir=args.config.rollout_dir,
            eval_n_trajs=args.config.eval_n_trajs,
            n_rollout_steps=args.config.n_rollout_steps,
            out_type=args.config.out_type,
            n_extrap_steps=args.config.n_extrap_steps,
            seed=args.config.seed,
        )

        print(averaged_metrics(metrics))
