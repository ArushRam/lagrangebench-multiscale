import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import haiku as hk
import lagrangebench
from omegaconf import DictConfig, OmegaConf

def load_embedded_configs(config_path: str, cli_args: DictConfig) -> DictConfig:
    """Loads all 'extends' embedded configs and merge them with the cli overwrites."""
    cfgs = [OmegaConf.load(config_path)]
    while "extends" in cfgs[0]:
        extends_path = cfgs[0]["extends"]
        del cfgs[0]["extends"]

        # go to parents configs until the defaults are reached
        if extends_path != "LAGRANGEBENCH_DEFAULTS":
            cfgs = [OmegaConf.load(extends_path)] + cfgs
        else:
            from lagrangebench.defaults import defaults
            cfgs = [defaults] + cfgs
            break

    # merge all embedded configs and give highest priority to cli_args
    cfg = OmegaConf.merge(*cfgs, cli_args)
    return cfg

def create_model(cfg, metadata):
    """Create the model with specified architecture."""
    def model_fn(x):
        if cfg.model.name.lower() == "ms_gns":
            return lagrangebench.models.MultiScaleGNS(
                metadata=metadata,
                particle_dimension=2,  # 2D dataset
                latent_size=cfg.model.latent_dim,
                blocks_per_step=cfg.model.num_mlp_layers,
                num_scales=cfg.model.num_scales,
                particle_type_embedding_size=16,
                mp_steps_per_scale=cfg.model.mp_steps_per_scale,
                clustering_type=cfg.model.clustering_type,
            )(x)
        elif cfg.model.name.lower() == "gns":
            return lagrangebench.models.GNS(
                particle_dimension=2,  # 2D dataset
                latent_size=cfg.model.latent_dim,
                blocks_per_step=cfg.model.num_mlp_layers,
                num_mp_steps=cfg.model.num_mp_steps,
                particle_type_embedding_size=16,
            )(x)
        else:
            raise ValueError(f"Unknown model type: {cfg.model.name}")
            
    # return hk.without_apply_rng(hk.transform_with_state(model_fn))
    return hk.transform_with_state(model_fn)

def load_checkpoint(ckpt_path):
    """Load model parameters and state from checkpoint."""
    from lagrangebench.utils import load_haiku
    params, state, _, _ = load_haiku(ckpt_path)
    return params, state

def evaluate_model(model, params, state, test_data, case_setup, cfg_eval_infer):
    """Evaluate model and generate rollout."""
    metrics = lagrangebench.infer(
        model,
        case=case_setup,
        data_test=test_data,
        params=params,
        state=state,
        cfg_eval_infer=cfg_eval_infer,
        n_rollout_steps=cfg_eval_infer.n_rollout_steps,
        rollout_dir="rollouts/",
    )["rollout_0"]
    
    return metrics, pickle.load(open("rollouts/rollout_0.pkl", "rb"))
    
def plot_metrics(metrics, save_path="metrics.png"):
    """Plot evaluation metrics."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(metrics["mse"])
    ax.set_title("MSE over time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("MSE")
    plt.savefig(save_path)
    plt.close()

def create_animation(rollout, save_path="scatter.gif"):
    """Create and save animation of the rollout."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_xlim([0, 1.0])
    ax[0].set_ylim([0, 1.0])
    ax[1].set_xlim([0, 1.0])
    ax[1].set_ylim([0, 1.0])
    ax[0].set_title("Prediction")
    ax[1].set_title("Ground Truth")

    rollout_len = rollout["predicted_rollout"].shape[0] - 1
    num_particles = rollout["predicted_rollout"].shape[1]

    # Create a color map based on initial x,y coordinates
    initial_pos = rollout["predicted_rollout"][0]
    # Sort particles by x coordinate first, then y coordinate
    sort_idx = np.lexsort((initial_pos[:,1], initial_pos[:,0]))
    
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num_particles))

    scat0 = ax[0].scatter(
        rollout["predicted_rollout"][0, sort_idx, 0],
        rollout["predicted_rollout"][0, sort_idx, 1],
        c=colors
    )
    scat1 = ax[1].scatter(
        rollout["ground_truth_rollout"][0, sort_idx, 0],
        rollout["ground_truth_rollout"][0, sort_idx, 1],
        c=colors
    )

    def animate(i):
        scat0.set_offsets(rollout["predicted_rollout"][i, sort_idx])
        scat1.set_offsets(rollout["ground_truth_rollout"][i, sort_idx])
        return scat0, scat1

    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=rollout_len, interval=50
    )
    writer = animation.PillowWriter(fps=10, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(save_path, writer=writer)
    plt.close()

def main():
    # Load and merge configs
    cli_args = OmegaConf.from_cli()
    assert ("config" in cli_args) != ("load_ckp" in cli_args), \
        "You must specify one of 'config' or 'load_ckp'."

    if "config" in cli_args:
        config_path = cli_args.config
        name = config_path.split("/")[-1].split(".")[0]
    elif "load_ckp" in cli_args:
        config_path = os.path.join(cli_args.load_ckp, "config.yaml")
        name = cli_args.load_ckp.split("/")[-1].split(".")[0]

    # Set GPU/CPU device
    cli_args.gpu = cli_args.get("gpu", -1)
    cli_args.xla_mem_fraction = cli_args.get("xla_mem_fraction", 0.75)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu)
    if cli_args.gpu == -1:
        os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cli_args.xla_mem_fraction)

    cfg = load_embedded_configs(config_path, cli_args)

    # Create output directories
    os.makedirs("rollouts", exist_ok=True)
    os.makedirs(f"media/{name}", exist_ok=True)

    # Load datasets using setup_data
    from lagrangebench.runner import setup_data
    _, _, data_test = setup_data(cfg)

    # Setup case
    bounds = np.array(data_test.metadata["bounds"])
    box = bounds[:, 1] - bounds[:, 0]
    
    rpf2d_case = lagrangebench.case_builder(
        box=box,
        metadata=data_test.metadata,
        input_seq_length=cfg.model.input_seq_length,
        cfg_neighbors=cfg.neighbors,
        cfg_model=cfg.model,
        noise_std=cfg.train.noise_std,
        external_force_fn=data_test.external_force_fn,
    )

    # Create and load model
    model = create_model(cfg, data_test.metadata)
    params, state = load_checkpoint(cfg.load_ckp)

    # Setup evaluation config
    cfg_eval_infer = OmegaConf.create({
        "metrics": ["mse"],
        "n_trajs": 1,
        "batch_size": 1,
        "out_type": "pkl",
        "n_rollout_steps": cfg.eval.n_rollout_steps
    })

    # Evaluate model
    metrics, rollout = evaluate_model(
        model, params, state, data_test, rpf2d_case, cfg_eval_infer
    )

    # Generate visualizations
    plot_metrics(metrics, save_path=f"media/{name}/metrics.png")
    create_animation(rollout, save_path=f"media/{name}/scatter.gif")

    print(f"Testing completed. Check media/{name}/metrics.png and media/{name}/scatter.gif for results.")

if __name__ == "__main__":
    main()