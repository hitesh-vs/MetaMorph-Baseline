import argparse
import os
import sys

import torch

from gym.envs.registration import register

from metamorph.algos.ppo.ppo import PPO
from metamorph.config import cfg
from metamorph.config import dump_cfg
from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu
from metamorph.utils.xml_utils import strip_unsupported_attrs

import matplotlib
matplotlib.use('Agg')

def set_cfg_options():

    # Hard override: multi-robot always needs consistent padding (for baseline)
    cfg.MODEL.MLP.CONSISTENT_PADDING = True
    
    calculate_max_iters()
    maybe_infer_walkers()
    calculate_max_limbs_joints()


def calculate_max_limbs_joints():
    """
    Calculate MAX_LIMBS and MAX_JOINTS based on robot type.
    
    LIMITATION FOR REAL URDFs: This function uses hardcoded string matching
    on cfg.ENV.WALKER_DIR to determine padding sizes. This breaks when:
    1. New robot URDFs don't match the hardcoded patterns
    2. Robot names use different conventions
    3. Multiple robot types are mixed
    
    RESEARCH NOTE: This is a key limitation you're addressing with learned
    functional groupings - no more manual specification needed.
    """
    if cfg.ENV_NAME == "Unimal-v0":

        num_joints, num_limbs = [], []

        metadata_paths = []
        for agent in cfg.ENV.WALKERS:
            metadata_paths.append(os.path.join(
                cfg.ENV.WALKER_DIR, "metadata", "{}.json".format(agent)
            ))

        for metadata_path in metadata_paths:
            metadata = fu.load_json(metadata_path)
            num_joints.append(metadata["dof"])
            num_limbs.append(metadata["num_limbs"] + 1)

        # Add extra 1 for max_joints; needed for adding edge padding
        cfg.MODEL.MAX_JOINTS = max(num_joints) + 1
        cfg.MODEL.MAX_LIMBS = max(num_limbs) + 1
        cfg.MODEL.MAX_JOINTS = 16
        cfg.MODEL.MAX_LIMBS = 12
        print (cfg.MODEL.MAX_JOINTS, cfg.MODEL.MAX_LIMBS)
    
    elif cfg.ENV_NAME == 'Modular-v0':
        # HARDCODED ROBOT CONFIGURATIONS
        # This is a key limitation: each new real URDF requires manual configuration
        
        if 'hopper' in cfg.ENV.WALKER_DIR:
            cfg.MODEL.MAX_LIMBS = 5
            cfg.MODEL.MAX_JOINTS = 5
            
        if 'walker' in cfg.ENV.WALKER_DIR:
            cfg.MODEL.MAX_LIMBS = 7
            cfg.MODEL.MAX_JOINTS = 7
            
        if 'humanoid' in cfg.ENV.WALKER_DIR:
            cfg.MODEL.MAX_LIMBS = 9
            cfg.MODEL.MAX_JOINTS = 9
        
        # REAL URDF SUPPORT: Unitree G1 (12-DOF humanoid)
        # Manual configuration required - demonstrates need for learned groupings
        if 'unitree_g1' in cfg.ENV.WALKER_DIR or 'g1_12dof' in cfg.ENV.WALKER_DIR:
            cfg.MODEL.MAX_LIMBS = 13  # pelvis + 12 actuated limbs
            cfg.MODEL.MAX_JOINTS = 12  # 12 actuated joints (free joint not counted)
            print(f"[Real URDF] Configured for Unitree G1: MAX_LIMBS=13, MAX_JOINTS=12")
        
        # Multi-robot training (requires consistent padding)
        if 'all' in cfg.ENV.WALKER_DIR:
            if cfg.MODEL.MLP.CONSISTENT_PADDING:
                cfg.MODEL.MAX_LIMBS = 19
                cfg.MODEL.MAX_JOINTS = 19
            else:
                cfg.MODEL.MAX_LIMBS = 9
                cfg.MODEL.MAX_JOINTS = 9
        
        # Debug output to verify configuration
        print(f"[Config] ENV.WALKER_DIR: {cfg.ENV.WALKER_DIR}")
        print(f"[Config] MAX_LIMBS: {cfg.MODEL.MAX_LIMBS}, MAX_JOINTS: {cfg.MODEL.MAX_JOINTS}")
        print(f"[Config] WALKERS: {cfg.ENV.WALKERS}")


def calculate_max_iters():
    # Iter here refers to 1 cycle of experience collection and policy update.
    cfg.PPO.MAX_ITERS = (
        int(cfg.PPO.MAX_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )
    cfg.PPO.EARLY_EXIT_MAX_ITERS = (
        int(cfg.PPO.EARLY_EXIT_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )


def maybe_infer_walkers():
    if cfg.ENV_NAME not in ["Unimal-v0", "Modular-v0"]:
        return

    # Only infer the walkers if this option was not specified
    if len(cfg.ENV.WALKERS):
        return
    
    xml_dir = os.path.join(cfg.ENV.WALKER_DIR, "xml")
    cfg.ENV.WALKERS = [
        f[:-4]                              # strip .xml
        for f in os.listdir(xml_dir)
        if f.endswith(".xml") and not f.endswith("_stripped.xml")
    ]

    if cfg.ENV_NAME == 'Modular-v0':
        register_modular_envs()


def register_modular_envs():
    """register the MuJoCo envs with Gym and return the per-agent observation size and max action value (for modular policy training)"""
    # register each env
    for agent in cfg.ENV.WALKERS:
        xml = os.path.join(cfg.ENV.WALKER_DIR, 'xml', agent + '.xml')
        xml_abs = os.path.abspath(xml)
        
        # Strip incompatible attrs once here, before any subprocess sees it
        clean_xml = strip_unsupported_attrs(xml_abs)
        
        params = {"xml": clean_xml}   # ← pass the stripped path directly
        try:
            register(
                id=f"{agent}-v0",
                max_episode_steps=1000,
                entry_point=f"modular.{agent}:make_env",
                kwargs=params,
            )
        except Exception as e:
            print(f"[Gym Registration] Failed to register {agent}-v0: {e}")
            continue


def get_hparams():
    hparam_path = os.path.join(cfg.OUT_DIR, "hparam.json")
    # For local sweep return
    if not os.path.exists(hparam_path):
        return {}

    hparams = {}
    varying_args = fu.load_json(hparam_path)
    flatten_cfg = swu.flatten(cfg)

    for k in varying_args:
        hparams[k] = flatten_cfg[k]

    return hparams


def cleanup_tensorboard():
    tb_dir = os.path.join(cfg.OUT_DIR, "tensorboard")

    # Assume there is only one sub_dir and break when it's found
    for content in os.listdir(tb_dir):
        content = os.path.join(tb_dir, content)
        if os.path.isdir(content):
            break

    # Return if no dir found
    if not os.path.isdir(content):
        return

    # Move all the event files from sub_dir to tb_idr
    for event_file in os.listdir(content):
        src = os.path.join(content, event_file)
        dst = os.path.join(tb_dir, event_file)
        fu.move_file(src, dst)

    # Delete the sub_dir
    os.rmdir(content)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument(
        "--no_context_in_state", action="store_true"
    )
    parser.add_argument(
        "opts",
        help="See morphology/core/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def ppo_train():
    su.set_seed(cfg.RNG_SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.set_num_threads(1)

    if cfg.get("TSNE_MODE", False):
        run_tsne_collection()
    else:
        print("="*80)
        print("STARTING PPO TRAINING")
        print("="*80)
        PPOTrainer = PPO()
        PPOTrainer.train()
        hparams = get_hparams()
        PPOTrainer.save_rewards(hparams=hparams)
        PPOTrainer.save_model(-1)
        cleanup_tensorboard()

def get_limb_labels_from_xml(xml_path, robot_name):
    """
    Build limb labels directly from XML — no env instance needed.
    Works with SubprocVecEnv since we never touch the subprocess.
    """
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    
    # DFS to collect bodies + build parent map (mirrors MujocoGraphParser)
    nodes   = []   # body names in DFS order
    parents = {}   # name -> parent_name or None

    def traverse(body, parent_name):
        name = body.attrib.get("name", f"body_{len(nodes)}")
        nodes.append(name)
        parents[name] = parent_name
        for child in body.findall("body"):
            traverse(child, name)

    for body in worldbody.findall("body"):
        traverse(body, None)

    # Depth from root
    depth = {}
    def get_depth(name):
        if name not in depth:
            p = parents[name]
            depth[name] = 0 if p is None else get_depth(p) + 1
        return depth[name]
    for n in nodes:
        get_depth(n)

    labels = []
    for bname in nodes:
        n = bname.lower()
        side  = ("left"   if ("left"  in n or "_l_" in n or n.startswith("l_")) else
                 "right"  if ("right" in n or "_r_" in n or n.startswith("r_")) else
                 "center")
        jtype = ("hip_pitch" if "hip_pitch" in n else
                 "hip_roll"  if "hip_roll"  in n else
                 "hip_yaw"   if "hip_yaw"   in n else
                 "knee"      if "knee"      in n else
                 "ankle"     if "ankle"     in n else
                 "root"      if ("pelvis"   in n or "torso" in n) else
                 "other")
        labels.append({
            "robot":    robot_name,
            "body":     bname,
            "semantic": f"{side}_{jtype}",
            "depth":    depth[bname],
        })
    return labels


def run_tsne_collection():
    import pickle
    import numpy as np

    # How many PPO updates before collecting
    WARMUP_ITERS = cfg.TSNE_WARMUP_ITERS 

    trainer = PPO()
    ac      = trainer.actor_critic
    mu_net  = ac.mu_net
    envs    = trainer.envs

    # ── Fix obs_space / buffer shape mismatch ────────────────────────────
    # The gym obs_space is registered before cmdline overrides fully propagate
    # to subprocesses. Get the real shape from an actual reset and rebuild buffer.
    real_obs = envs.reset()
    real_obs_space = {}
    for k, v in real_obs.items():
        import gym
        real_obs_space[k] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=v.shape[1:],   # drop batch dim
            dtype=np.float32
        )
    real_obs_space = gym.spaces.Dict(real_obs_space)

    # Rebuild buffer with correct shapes
    from metamorph.algos.ppo.buffer import Buffer
    trainer.buffer = Buffer(real_obs_space, envs.action_space.shape)
    trainer.buffer.to(trainer.device)

    print(f"[t-SNE] Buffer rebuilt from real obs shapes")
    for k, v in real_obs.items():
        print(f"  {k}: {v.shape[1:]}")

    # Optionally load checkpoint
    ckpt = cfg.TSNE_CHECKPOINT
    if ckpt and os.path.exists(ckpt):
        loaded = torch.load(ckpt, map_location="cpu")
        if isinstance(loaded, list):
            ac.load_state_dict(loaded[0].state_dict())
            from metamorph.algos.ppo.envs import set_ob_rms
            set_ob_rms(envs, loaded[1])
        else:
            ac.load_state_dict(loaded)
        print(f"[t-SNE] Loaded checkpoint: {ckpt}")
        WARMUP_ITERS = 0  # checkpoint already trained, collect immediately
    else:
        print(f"[t-SNE] No checkpoint — running {WARMUP_ITERS} PPO iters first")

    # ── Warmup: run real PPO updates so GCN has seen gradients ───────────
    if WARMUP_ITERS > 0:
        obs = envs.reset()
        trainer.buffer.to(trainer.device)
        trainer.start = __import__('time').time()

        for cur_iter in range(WARMUP_ITERS):
            for step in range(cfg.PPO.TIMESTEPS):
                unimal_ids = [0 for _ in range(cfg.PPO.NUM_ENVS)]
                val, act, logp, dmv, dmmu = trainer.agent.act(obs, unimal_ids=unimal_ids)
                next_obs, reward, done, infos = envs.step(act)
                masks = torch.tensor(
                    [[0.0] if d else [1.0] for d in done],
                    dtype=torch.float32, device=trainer.device)
                timeouts = torch.tensor(
                    [[0.0] if "timeout" in info else [1.0] for info in infos],
                    dtype=torch.float32, device=trainer.device)
                trainer.buffer.insert(obs, act, logp, val, reward, masks, timeouts, dmv, dmmu, unimal_ids)
                obs = next_obs

            unimal_ids = [0 for _ in range(cfg.PPO.NUM_ENVS)]
            next_val = trainer.agent.get_value(obs, unimal_ids=unimal_ids)
            trainer.buffer.compute_returns(next_val)
            trainer.train_on_batch(cur_iter)
            print(f"[t-SNE] Warmup iter {cur_iter+1}/{WARMUP_ITERS} done")

    # ── Now collect embeddings ────────────────────────────────────────────
    ac.eval()
    mu_net._tsne_buffer["active"] = True
    obs = envs.reset()

    COLLECT_STEPS = 30
    with torch.no_grad():
        for step in range(COLLECT_STEPS):
            unimal_ids = [0 for _ in range(cfg.PPO.NUM_ENVS)]
            _, act, _, _, _ = trainer.agent.act(obs, unimal_ids=unimal_ids)
            obs, _, _, _    = envs.step(act)

    mu_net._tsne_buffer["active"] = False
    ac.train()

    # ── Flatten embeddings + labels ───────────────────────────────────────
    raw      = mu_net._tsne_buffer["embeds"]
    combined = torch.cat(raw, dim=1)
    padded_seq_len, N_total, d = combined.shape
    flat_embeds = combined.permute(1, 0, 2).reshape(-1, d).numpy()

    # build labels from XML
    import os
    from metamorph.utils.xml_utils import strip_unsupported_attrs
    walker_labels = {}
    for walker in cfg.ENV.WALKERS:
        xml_path = os.path.join(cfg.ENV.WALKER_DIR, "xml", walker + ".xml")
        xml_path = strip_unsupported_attrs(xml_path)
        walker_labels[walker] = get_limb_labels_from_xml(xml_path, walker)

    n_envs     = cfg.PPO.NUM_ENVS
    env_labels = [
        walker_labels[cfg.ENV.WALKERS[i % len(cfg.ENV.WALKERS)]]
        for i in range(n_envs)
    ]

    n_steps = N_total // n_envs
    flat_labels = []
    for _ in range(n_steps):
        for lbl_list in env_labels:
            flat_labels.extend(lbl_list)
            n_pad = padded_seq_len - len(lbl_list)
            flat_labels.extend([{
                "robot": "pad", "body": "pad",
                "semantic": "pad", "depth": -1
            }] * n_pad)

    min_len     = min(len(flat_embeds), len(flat_labels))
    flat_embeds = flat_embeds[:min_len]
    flat_labels = flat_labels[:min_len]

    keep        = [i for i, l in enumerate(flat_labels) if l["semantic"] != "pad"]
    flat_embeds = flat_embeds[keep]
    flat_labels = [flat_labels[i] for i in keep]

    print(f"[t-SNE] Collected {len(flat_embeds)} embeddings (dim={d}) after {WARMUP_ITERS} PPO iters")

    out_prefix = os.path.join(cfg.OUT_DIR, "tsne_data")
    np.save(f"{out_prefix}_embeds.npy", flat_embeds)
    with open(f"{out_prefix}_labels.pkl", "wb") as f:
        pickle.dump(flat_labels, f)
    print(f"[t-SNE] Saved → {out_prefix}_embeds.npy / _labels.pkl")

    plot_tsne(flat_embeds, flat_labels, out_prefix)


def plot_tsne(X, labels, prefix):
    import pickle
    import numpy as np
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    semantic = np.array([l["semantic"] for l in labels])
    robots   = np.array([l["robot"]   for l in labels])
    depths   = np.array([str(l["depth"]) for l in labels])

    # L2 normalise
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    print(f"[t-SNE] Running on {X.shape[0]} vectors of dim {X.shape[1]} ...")
    Z = TSNE(n_components=2, perplexity=40, metric="cosine",
             n_iter=2000, random_state=42).fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    _tsne_scatter(axes[0], Z, semantic, "Joint type (semantic)")
    _tsne_scatter(axes[1], Z, robots,   "Robot identity  ← want MIXED")
    _tsne_scatter(axes[2], Z, depths,   "Topological depth")
    plt.suptitle("t-SNE of transformer input embedding (obs_embed after GCN)")
    plt.tight_layout()
    out = f"{prefix}_tsne.png"
    plt.savefig(out, dpi=150)
    print(f"[t-SNE] Plot saved → {out}")
    #plt.show()

    # Purity scores
    for lbl_arr, name in [(semantic, "semantic"), (robots, "robot"), (depths, "depth")]:
        nn  = NearestNeighbors(n_neighbors=16, metric="euclidean").fit(Z)
        _, idx = nn.kneighbors(Z)
        idx = idx[:, 1:]
        purity = np.mean([np.mean(lbl_arr[idx[i]] == lbl_arr[i])
                          for i in range(len(lbl_arr))])
        print(f"  KNN purity [{name:>10s}]: {purity:.3f}  "
              f"(chance={1/len(set(lbl_arr)):.3f})")


def _tsne_scatter(ax, Z, labels, title):
    import matplotlib.cm as cm
    uniq = sorted(set(labels))
    cmap = cm.get_cmap("tab20", len(uniq))
    for i, u in enumerate(uniq):
        m = labels == u
        ax.scatter(Z[m, 0], Z[m, 1], c=[cmap(i)], label=u, alpha=0.55, s=14)
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title(title)


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    if args.no_context_in_state:
        obs_type = [
            "body_xpos", "body_xvelp", "body_xvelr", "body_xquat", # limb
            "qpos", "qvel", # joint
        ]
        ob_opts = ["MODEL.PROPRIOCEPTIVE_OBS_TYPES", obs_type]
        cfg.merge_from_list(ob_opts)

    # Set cfg options which are inferred
    set_cfg_options()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # Save the config
    dump_cfg()
    ppo_train()


if __name__ == "__main__":
    main()