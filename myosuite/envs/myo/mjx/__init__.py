from typing import Callable
from ml_collections import config_dict
import copy
from etils import epath
from jax import numpy as jp
import myosuite.envs.myo.mjx.myo_registry as registry
from mujoco_playground._src import mjx_env

from myosuite.envs.myo.mjx.playground_pose_v0 import MjxPoseEnvV0
from myosuite.envs.myo.mjx.playground_reach_v0 import MjxReachEnvV0
from myosuite.envs.myo.mjx.playground_keyboard_v0 import MjxKeyboardEnvV0

base_config = config_dict.create(
    ctrl_dt=0.02,
    sim_dt=0.002,
    num_envs=4_096,
    max_episode_steps=100,
    model_path=epath.Path("/tmp/dummy.xml"),
    impl="jax",
    norm_actions=True,
)

pose_env_config = config_dict.ConfigDict({**base_config, **config_dict.create(
    reward_config=config_dict.create(
        angle_reward_weight=1.0,
        ctrl_cost_weight=1.0,
        pose_thd=0.35,
        far_th=4 * jp.pi / 2,
        bonus_weight=4.0,
    ),
    target_jnt_range=config_dict.ConfigDict(),
)})

reach_env_config = config_dict.ConfigDict({**base_config, **config_dict.create(
    reward_config=config_dict.create(
        reach_weight=1.0,
        bonus_scale=4.0,
        penalty_scale=50.0,
    ),
    target_reach_range=config_dict.ConfigDict(),
    far_th=0.35,
)})

keyboard_env_config = config_dict.ConfigDict({**base_config, **config_dict.create(
    # single-key press task (milestone 1). Small right-hand board.
    num_envs=2_048,
    max_episode_steps=100,
    press_th=0.6,          # fraction of key travel counted as "pressed"
    release_th=0.2,        # travel below which a key counts as released (debounce)
    reach_th=0.015,        # fingertip-key distance counted as "on key" (m)
    far_th=0.30,           # episode-fail distance (m)
    sequence_length=1,     # 1 -> single-key hold; >1 -> type a key sequence in order
    base_init_noise=0.15,  # randomize the hand start over [-1,1]*noise*half-range
    pronate=1.4,           # forearm pronation (palm down)
    knuckle_arch=0.15,     # MCP flexion so fingers arch downward
    finger_spread=0.3,     # MCP abduction to fan fingers across columns
    hover_z=0.03,          # base_z init so fingertips hover above keys
    home_offset=(0.04, 0.04),  # (dx,dy) base shift so fingers rest on keys
    wrist_stiffness=8.0,   # passive stiffness holding the wrist pose
    naconmax_per_env=48,   # contact-buffer budget/env (broadphase candidates, pre-filter)
    target_keys=(),        # () -> every key on the board is targetable
    obs_released_seen=False,  # P5: expose the debounced release latch in the obs
    # multi-key WORD typing (WACV). Inert at these defaults -> the single-key
    # task (and the archived checkpoints) are byte-identical; the words configs
    # below switch them on.
    obs_lookahead=0,            # >0: expose N upcoming keys (+ latch) -> Markov obs
    grace_steps=12,             # steps after a keystroke before a far-fail can fire
    switch_done_mode="target",  # "target" (old, dist-to-current) | "board" (off-home)
    board_far_th=0.12,          # home-region radius for the "board" fail test (m)
    home_relax_on_handoff=True, # exempt the just-used finger from the home penalty
    word_table_path="",         # npz of padded real words; "" -> uniform-random keys
    # human muscle-synergy action space (Task 1). When synergy=True the policy
    # acts in K human-synergy latents per hand (decoded to muscle activations by
    # a FIXED linear map) + the base servos, instead of raw per-muscle commands.
    synergy=False,
    synergy_k=12,
    synergy_span=4.0,
    synergy_basis_path="",         # set below to synergy_basis_right.npz
    synergy_decoder_path="",       # if set: generative MLP decoder g(z)->act (not linear)
    synergy_participant="",         # per-participant npz: which participant id key
    synergy_scale_template="",     # human template used for the latent scale
    muscle_template_path_left="",  # left-hand template (bimanual muscle_match)
    reward_config=config_dict.create(
        reach_weight=1.5,
        press_weight=5.0,
        bonus_weight=3.0,
        home_weight=1.0,
        act_reg_weight=1.0,
        penalty_weight=10.0,
        keystroke_weight=5.0,   # sparse bonus per completed keystroke (seq mode)
        progress_weight=0.0,    # per-keystroke progress bonus scaled by word depth
        approach_weight=0.0,    # shaping toward the NEXT key (pre-positioning)
    ),
)})
model_path = "envs/myo/assets/hand/"
model_filename = "myohand_keyboard_mini.xml"
keyboard_env_config["model_path"] = (
    epath.Path(epath.resource_path("myosuite")) / model_path / model_filename
)
# repo root (emg2qwerty_myo/) for the How-We-Type datasets.
_repo_root = epath.Path(__file__).parent.parent.parent.parent.parent.parent
_synergy_basis = (_repo_root / "datasets/how_we_type/synergy_basis_right.npz").as_posix()
_muscle_tmpl_r = (_repo_root / "datasets/how_we_type/muscle_template_right.npz").as_posix()
_muscle_tmpl_l = (_repo_root / "datasets/how_we_type/muscle_template_left.npz").as_posix()
keyboard_env_config["synergy_basis_path"] = _synergy_basis
keyboard_env_config["synergy_scale_template"] = _muscle_tmpl_r

# Sequence typing task (milestone 2): type a short key sequence in order, each a
# debounced keystroke (press-after-release), episode ends on word completion.
keyboard_seq_config = copy.deepcopy(keyboard_env_config)
keyboard_seq_config["sequence_length"] = 4
keyboard_seq_config["max_episode_steps"] = 180   # room to hit 4 keys / word
keyboard_seq_config["num_envs"] = 1_024

# Right-hand-reachable keys (sanitized names) for the single-hand FULL board.
_RIGHT_HAND_KEYS = (
    "6", "7", "8", "9", "0", "y", "u", "i", "o", "p",
    "h", "j", "k", "l", "semicolon", "n", "m", "comma", "period", "space",
)

# FULL 61-key board, single right hand (milestone 2b). Same scene as the CPU env;
# touch sensors are stripped at load. Only the right-reachable keys are targeted.
keyboard_full_config = copy.deepcopy(keyboard_env_config)
keyboard_full_config["target_keys"] = _RIGHT_HAND_KEYS
keyboard_full_config["num_envs"] = 1_024
keyboard_full_config["naconmax_per_env"] = 64
keyboard_full_config["model_path"] = (
    epath.Path(epath.resource_path("myosuite")) / model_path / "myohand_keyboard.xml"
)

# MUSCLE-TEMPLATE imitation (milestone 2c): full right-hand board, but the reward
# adds a term pulling the current target key's muscle activation toward the human
# per-key template (datasets/how_we_type/muscle_template_right.npz), gated to keys
# with >= template_min_count human presses. Verifies a GPU typist still types AND
# its muscle activations become more human-like (the CPU A/B: +57% similarity).
keyboard_template_config = copy.deepcopy(keyboard_full_config)
keyboard_template_config["muscle_template_path"] = _muscle_tmpl_r
keyboard_template_config["template_min_count"] = 50
keyboard_template_config["reward_config"]["muscle_match_weight"] = 0.5

# A/B baseline: SAME board + template loaded (so muscle_match is measured) but
# reward weight 0 (no shaping) -> the "types normally, human-dissimilar" control.
keyboard_template_base_config = copy.deepcopy(keyboard_template_config)
keyboard_template_base_config["reward_config"]["muscle_match_weight"] = 0.0

# Stronger-weight variant: on GPU (brax PPO, large returns) weight 0.5 is only
# ~2% of the return and gets washed out; weight 2.0 makes the human-template pull
# a visible fraction of the reward -> muscle_match rises with typing preserved.
keyboard_template_hi_config = copy.deepcopy(keyboard_template_config)
keyboard_template_hi_config["reward_config"]["muscle_match_weight"] = 2.0

# BIMANUAL full board (milestone 2b): two MyoHands (78 muscles, 10 fingertips);
# with both hands the whole keyboard is reachable so every key is targetable.
keyboard_bimanual_config = copy.deepcopy(keyboard_env_config)
keyboard_bimanual_config["target_keys"] = ()
keyboard_bimanual_config["num_envs"] = 512
keyboard_bimanual_config["naconmax_per_env"] = 110
keyboard_bimanual_config["model_path"] = (
    epath.Path(epath.resource_path("myosuite")) / model_path / "myohand_keyboard_bimanual.xml"
)

# ============================ Task 1: SYNERGY action space =====================
# Full right-hand board with the K=12 human muscle-synergy action space. The
# template is loaded (weight 0) so `muscle_match` (human-similarity) is MEASURED
# -> this is the "raw synergy" A/B arm vs raw-muscle (MjxKeyboardTemplateBase-v0).
keyboard_synergy_config = copy.deepcopy(keyboard_template_base_config)  # full board, w=0
keyboard_synergy_config["synergy"] = True
keyboard_synergy_config["synergy_k"] = 12

# Synergy action space + template regularizer (weight 2.0): the third A/B arm.
keyboard_synergy_hi_config = copy.deepcopy(keyboard_synergy_config)
keyboard_synergy_hi_config["reward_config"]["muscle_match_weight"] = 2.0

# ================= Task 2: BIMANUAL human-muscle imitation =====================
# Bimanual board + stacked right(39)+left(39)=78 muscle template, muscle_match
# reward weight 2.0. Each key is matched on the 39 muscles of its pressing hand.
keyboard_bimanual_template_config = copy.deepcopy(keyboard_bimanual_config)
keyboard_bimanual_template_config["muscle_template_path"] = _muscle_tmpl_r
keyboard_bimanual_template_config["muscle_template_path_left"] = _muscle_tmpl_l
keyboard_bimanual_template_config["synergy_scale_template"] = _muscle_tmpl_r
keyboard_bimanual_template_config["template_min_count"] = 50
keyboard_bimanual_template_config["reward_config"]["muscle_match_weight"] = 2.0

# A/B baseline for the bimanual imitation: template loaded (muscle_match measured)
# but reward weight 0 (no shaping).
keyboard_bimanual_template_base_config = copy.deepcopy(keyboard_bimanual_template_config)
keyboard_bimanual_template_base_config["reward_config"]["muscle_match_weight"] = 0.0

# FLAGSHIP: bimanual muscle-template imitation IN the human-synergy action space
# (per-hand K=12 basis applied block-wise -> 24 latents + 6 base = 30-D action).
keyboard_bimanual_synergy_config = copy.deepcopy(keyboard_bimanual_template_config)
keyboard_bimanual_synergy_config["synergy"] = True
keyboard_bimanual_synergy_config["synergy_k"] = 12

# ===================== WACV: multi-key WORD typing ===========================
# Long-duration real-word typing. The obs layout is FIXED (lookahead L=2 + the
# release latch) and independent of sequence_length, so curriculum stages
# (seq 1->2->4->8) warm-start each other. `word_table_path`, `sequence_length`
# and `max_episode_steps` are overridden per curriculum stage by the driver
# (training/curriculum_words.py); the defaults here give a runnable 4-key env.
def _make_words_config(base):
    cfg = copy.deepcopy(base)
    cfg["obs_lookahead"] = 2
    cfg["switch_done_mode"] = "board"
    cfg["grace_steps"] = 12
    cfg["board_far_th"] = 0.14
    cfg["sequence_length"] = 4
    cfg["max_episode_steps"] = 4 * 45          # ~45 ctrl steps (0.9 s) per key
    cfg["reward_config"]["progress_weight"] = 5.0
    cfg["reward_config"]["approach_weight"] = 0.5
    return cfg

# Single right hand, full board (cheap dev path) -> right-reachable words.
keyboard_words_config = _make_words_config(keyboard_template_base_config)
# Bimanual full board (the real target) -> whole-keyboard words, two hands.
keyboard_words_bimanual_config = _make_words_config(keyboard_bimanual_template_base_config)

ppo_config = config_dict.create(
    num_timesteps=50_000_000,
    learning_rate=3e-4,
    discounting=0.97,
    gae_lambda=0.95,
    entropy_cost=0.001,
    clipping_epsilon=0.3,
    max_grad_norm=1.0,
    action_repeat=1,
    num_minibatches=32,
    num_updates_per_batch=8,
    batch_size=256,
    unroll_length=10,
    reward_scaling=1.0,
    normalize_observations=True,
    num_evals=16,
    num_eval_envs=128,
    num_resets_per_eval=1,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(64, 64, 64),
        value_hidden_layer_sizes=(64, 64, 64),
        policy_obs_key="state",
        value_obs_key="state",
    ),
)

# Elbow posing ==============================
elbow_pose_env_config = copy.deepcopy(pose_env_config)
model_path = "envs/myo/assets/elbow/"
model_filename = "myoelbow_1dof6muscles.xml"
elbow_pose_env_config["model_path"] = (
    epath.Path(epath.resource_path("myosuite")) / model_path / model_filename
)

# Finger joint posing ==============================
finger_pose_env_config = copy.deepcopy(pose_env_config)
model_path = "simhive/myo_sim/finger/"
model_filename = "myofinger_v0.xml"
finger_pose_env_config["model_path"] = (
    epath.Path(epath.resource_path("myosuite")) / model_path / model_filename
)

# Hand tips reaching ==============================
hand_reach_env_config = copy.deepcopy(reach_env_config)
model_path = "envs/myo/assets/hand/"
model_filename = "myohand_pose.xml"
hand_reach_env_config["model_path"] = (
    epath.Path(epath.resource_path("myosuite")) / model_path / model_filename
)


def wrap_class(wrapper_cls, wrapped_env_cls, wrapper_config=None):
    def _get_wrapped_class(*args, **kwargs):
        return wrapper_cls(wrapped_env_cls(*args, **kwargs), **(wrapper_config if wrapper_config is not None else {}))
    return _get_wrapped_class


def config_callable(env_config) -> Callable[[], config_dict.ConfigDict]:
    fn = lambda: env_config
    return fn


def get_default_config(env_name) -> config_dict.ConfigDict:
    return registry.get_default_config(env_name)

# TODO: is there a reason these are not registered on import?
def make(env_name: str, config_overrides=None) -> mjx_env.MjxEnv:

    env_name_base = registry.get_base_env_name(env_name)
    if "MjxElbowPose" in env_name_base:

        if env_name_base == "MjxElbowPoseFixed-v0":
            elbow_pose_env_config["target_jnt_range"] = config_dict.create(
                    r_elbow_flex=jp.array(((2), (2)))
                )
        elif env_name_base == "MjxElbowPoseRandom-v0":
            elbow_pose_env_config["target_jnt_range"] = config_dict.create(
                    r_elbow_flex=jp.array(((0), (2.27)))
                )
        registry.register_environment_with_variants(env_name_base,
                                      MjxPoseEnvV0,
                                      config_callable(elbow_pose_env_config))
        env = registry.load(env_name, config_overrides=config_overrides)

        return env

    if "MjxFingerPose" in env_name_base:

        if env_name_base == "MjxFingerPoseFixed-v0":
            finger_pose_env_config["target_jnt_range"] = config_dict.create(
                IFadb=jp.array(((0), (0))),
                IFmcp=jp.array(((0), (0))),
                IFpip=jp.array(((0.75), (0.75))),
                IFdip=jp.array(((0.75), (0.75))),
            )
        elif env_name_base == "MjxFingerPoseRandom-v0":
            finger_pose_env_config["target_jnt_range"] = config_dict.create(
                IFadb=jp.array(((-0.2), (0.2))),
                IFmcp=jp.array(((-0.4), (1))),
                IFpip=jp.array(((0.1), (1))),
                IFdip=jp.array(((0.1), (1))),
            )
        registry.register_environment_with_variants(env_name_base,
                                      MjxPoseEnvV0,
                                      config_callable(finger_pose_env_config))
        env = registry.load(env_name, config_overrides=config_overrides)

        return env

    if "MjxHandReach" in env_name_base:

        if env_name_base == "MjxHandReachFixed-v0":
            hand_reach_env_config["far_th"] = 0.044
            hand_reach_env_config["target_reach_range"] = config_dict.create(
                THtip=jp.array(((-0.165, -0.537, 1.495), (-0.165, -0.537, 1.495))),
                IFtip=jp.array(((-0.151, -0.547, 1.455), (-0.151, -0.547, 1.455))),
                MFtip=jp.array(((-0.146, -0.547, 1.447), (-0.146, -0.547, 1.447))),
                RFtip=jp.array(((-0.148, -0.543, 1.445), (-0.148, -0.543, 1.445))),
                LFtip=jp.array(((-0.148, -0.528, 1.434), (-0.148, -0.528, 1.434))),
            )
        elif env_name_base == "MjxHandReachRandom-v0":
            hand_reach_env_config["far_th"] = 0.034
            hand_reach_env_config["target_reach_range"] = config_dict.create(
                THtip=jp.array(
                    (
                        (-0.165 - 0.020, -0.537 - 0.040, 1.495 - 0.040),
                        (-0.165 + 0.040, -0.537 + 0.020, 1.495 + 0.040),
                    )
                ),
                IFtip=jp.array(
                    (
                        (-0.151 - 0.040, -0.547 - 0.020, 1.455 - 0.010),
                        (-0.151 + 0.040, -0.547 + 0.020, 1.455 + 0.010),
                    )
                ),
                MFtip=jp.array(
                    (
                        (-0.146 - 0.040, -0.547 - 0.020, 1.447 - 0.010),
                        (-0.146 + 0.040, -0.547 + 0.020, 1.447 + 0.010),
                    )
                ),
                RFtip=jp.array(
                    (
                        (-0.148 - 0.040, -0.543 - 0.020, 1.445 - 0.010),
                        (-0.148 + 0.040, -0.543 + 0.020, 1.445 + 0.010),
                    )
                ),
                LFtip=jp.array(
                    (
                        (-0.148 - 0.040, -0.528 - 0.020, 1.434 - 0.010),
                        (-0.148 + 0.040, -0.528 + 0.020, 1.434 + 0.010),
                    )
                ),
            )
        registry.register_environment(
            env_name, MjxReachEnvV0, config_callable(hand_reach_env_config)
        )
        env = registry.load(env_name, config_overrides=config_overrides)

        return env

    if "MjxKeyboard" in env_name_base:
        # NOTE: order matters (substring match) -> most specific names first.
        # (WordsBimanual before Bimanual, and both words names before the rest.)
        if "WordsBimanual" in env_name_base:
            cfg = keyboard_words_bimanual_config
        elif "Words" in env_name_base:
            cfg = keyboard_words_config
        elif "BimanualSynergy" in env_name_base:
            cfg = keyboard_bimanual_synergy_config
        elif "BimanualTemplateBase" in env_name_base:
            cfg = keyboard_bimanual_template_base_config
        elif "BimanualTemplate" in env_name_base:
            cfg = keyboard_bimanual_template_config
        elif "Bimanual" in env_name_base:
            cfg = keyboard_bimanual_config
        elif "SynergyHi" in env_name_base:
            cfg = keyboard_synergy_hi_config
        elif "Synergy" in env_name_base:
            cfg = keyboard_synergy_config
        elif "TemplateBase" in env_name_base:
            cfg = keyboard_template_base_config
        elif "TemplateHi" in env_name_base:
            cfg = keyboard_template_hi_config
        elif "Template" in env_name_base:
            cfg = keyboard_template_config
        elif "Full" in env_name_base:
            cfg = keyboard_full_config
        elif "TypeSeq" in env_name_base:
            cfg = keyboard_seq_config
        else:
            cfg = keyboard_env_config
        registry.register_environment_with_variants(
            env_name_base, MjxKeyboardEnvV0, config_callable(cfg)
        )
        env = registry.load(env_name, config_overrides=config_overrides)

        return env


env_names = [
    "MjxElbowPoseFixed-v0",
    "MjxElbowPoseRandom-v0",
    "MjxFingerPoseFixed-v0",
    "MjxFingerPoseRandom-v0",
    "MjxHandReachRandom-v0",
    "MjxHandReachFixed-v0",
    "MjxKeyboardKeyPress-v0",
    "MjxKeyboardTypeSeq-v0",
    "MjxKeyboardFullKeyPress-v0",
    "MjxKeyboardBimanual-v0",
    "MjxKeyboardTemplate-v0",
    "MjxKeyboardTemplateBase-v0",
    "MjxKeyboardTemplateHi-v0",
    "MjxKeyboardSynergy-v0",
    "MjxKeyboardSynergyHi-v0",
    "MjxKeyboardBimanualTemplate-v0",
    "MjxKeyboardBimanualTemplateBase-v0",
    "MjxKeyboardBimanualSynergy-v0",
    "MjxKeyboardWords-v0",
    "MjxKeyboardWordsBimanual-v0",
]
