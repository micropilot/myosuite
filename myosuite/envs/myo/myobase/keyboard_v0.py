"""=================================================
# Copyright (c) MyoSuite Authors
Authors  :: MyoHand QWERTY typing environment
================================================="""

import collections

import mujoco
import numpy as np

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils import gym


class KeyboardEnvV0(BaseV0):
    """MyoHand typing on a QWERTY keyboard.

    A right MyoHand (39 muscles) is mounted on a 3-DoF prismatic floating base
    (base_x, base_y, base_z, driven by 3 position servos appended after the
    muscles). The scene contains a QWERTY keyboard whose keys are spring-loaded
    sliders with `touch` sensors.

    Task (single-key press): a target key is selected each episode. The agent
    must bring a fingertip onto the key and depress it past `press_th` of its
    travel. Reward = fingertip->key reach + key depression + a press bonus,
    minus a muscle-effort penalty.
    """

    DEFAULT_OBS_KEYS = [
        "qpos", "qvel", "act",
        "fingertip_pos", "reach_err", "target_onehot", "target_travel",
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "press": 4.0,
        "bonus": 2.0,
        "act_reg": 1.0,
        "penalty": 10.0,
    }

    TIP_SITES = ["THtip", "IFtip", "MFtip", "RFtip", "LFtip"]

    # Keys comfortably reachable by a single RIGHT MyoHand (right/center region
    # of the board). Used as the default targetable set for the registered env;
    # pass target_keys=None to allow the full keyboard (e.g. once a left hand is
    # added). Names are the sanitized key names from make_keyboard.py.
    RIGHT_HAND_KEYS = [
        "6", "7", "8", "9", "0",
        "y", "u", "i", "o", "p",
        "h", "j", "k", "l", "semicolon",
        "n", "m", "comma", "period",
        "space",
    ]

    def __init__(
        self, model_path, obsd_model_path=None, seed=None, edit_fn=None, **kwargs
    ):
        # Two-step construction for pickling; see reach_v0.py for the rationale.
        gym.utils.EzPickle.__init__(
            self, model_path, obsd_model_path, seed, edit_fn=edit_fn, **kwargs
        )
        super().__init__(
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            edit_fn=edit_fn,
            env_credits=self.MYO_CREDIT,
        )
        self._setup(**kwargs)

    def _setup(
        self,
        target_keys: list = None,     # subset of key labels that can be targeted
        press_th: float = 0.6,        # fraction of key travel counted as "pressed"
        reach_th: float = 0.015,      # fingertip-key distance counted as "on key" (m)
        far_th: float = 0.30,         # episode-fail distance (m)
        base_init_noise: float = 0.0,  # randomize hand start over [-1,1]*noise*range
        base_init_raise: float = 0.0,  # bias the base_z start upward by this fraction
        pronate: float = 1.4,          # forearm pronation (palm down); 0 disables
        knuckle_arch: float = 0.15,    # MCP flexion so fingers arch downward
        hover_z: float = 0.03,         # base_z init so fingertips hover above keys
        wrist_stiffness: float = 8.0,  # passive stiffness holding the wrist pose
        fingertip_only_collision: bool = True,  # only fingertips collide with keys
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        frame_skip: int = 10,
        **kwargs,
    ):
        self.press_th = press_th
        self.reach_th = reach_th
        self.far_th = far_th
        self.base_init_noise = base_init_noise
        self.base_init_raise = base_init_raise
        self.pronate = pronate
        self.knuckle_arch = knuckle_arch
        self.hover_z = hover_z
        self.wrist_stiffness = wrist_stiffness
        self.fingertip_only_collision = fingertip_only_collision

        # Discover keyboard keys and fingertips from the model.
        self._discover_keys(target_keys)
        self.ftip_sids = np.array(
            [self.mj_model.site(s).id for s in self.TIP_SITES]
        )
        # visual marker is a mocap body; reward uses the real key site (below).
        self.target_mocap_id = int(self.mj_model.body("key_target").mocapid[0])

        # Base (non-muscle) actuator indices, and their ctrl ranges, for the
        # step() reprojection (BaseV0 only reprojects muscle actuators).
        self.muscle_act_ids = np.where(
            self.mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        )[0]
        self.base_act_ids = np.where(
            self.mj_model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE
        )[0]
        self.base_ctrlrange = self.mj_model.actuator_ctrlrange[self.base_act_ids].copy()

        # Base (slide) joint qpos indices + ranges, for start-state randomization.
        base_qpos, base_rng, base_is_z = [], [], []
        for jid in range(self.mj_model.njnt):
            jname = self.mj_model.joint(jid).name
            if jname.startswith("base_"):
                base_qpos.append(int(self.mj_model.joint(jid).qposadr[0]))
                base_rng.append(self.mj_model.jnt_range[jid].copy())
                base_is_z.append("_z" in jname or jname.endswith("z"))
        self.base_qpos_ids = np.array(base_qpos)
        self.base_qpos_range = np.array(base_rng) if base_rng else np.zeros((0, 2))
        self.base_is_z = np.array(base_is_z, dtype=bool)

        self.target_key = None  # set on reset

        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            sites=None,
            frame_skip=frame_skip,
            **kwargs,
        )

        # Physically-correct typing posture: pronate the forearm (palm down) and
        # arch the knuckles so the fingers press DOWN onto keys by flexion while
        # the wrist/palm hover ABOVE the keyboard. Without this the flat hand can
        # only "press" by sinking the whole hand through the keyboard. Done after
        # super()._setup so env_base's range-mean base init is already in place.
        if self.pronate:
            self._apply_typing_posture()
        if self.fingertip_only_collision:
            self._restrict_key_collisions()

    def _discover_keys(self, target_keys):
        """Introspect the model for key bodies produced by make_keyboard.py."""
        self.key_names = []       # sanitized names, e.g. "a", "space", "backspace"
        self.key_bids = []
        self.key_jnt_qposids = []
        self.key_travel = []
        self.key_site_ids = []
        for bid in range(self.mj_model.nbody):
            name = self.mj_model.body(bid).name
            if not name.startswith("key_") or name == "key_target":
                continue
            key = name[len("key_"):]
            jnt = self.mj_model.joint("key_%s_slide" % key)
            self.key_names.append(key)
            self.key_bids.append(bid)
            self.key_jnt_qposids.append(int(jnt.qposadr[0]))
            self.key_travel.append(float(jnt.range[1]))  # slide range [0, travel]
            self.key_site_ids.append(self.mj_model.site("key_%s_site" % key).id)
        self.key_bids = np.array(self.key_bids)
        self.key_jnt_qposids = np.array(self.key_jnt_qposids)
        self.key_travel = np.array(self.key_travel)
        self.key_site_ids = np.array(self.key_site_ids)

        # Targetable subset (indices into key_names).
        if target_keys is None:
            self.target_ids = np.arange(len(self.key_names))
        else:
            self.target_ids = np.array(
                [self.key_names.index(k) for k in target_keys]
            )

        # Hand DoF qpos indices = all joints that are neither base nor keys
        # (base joints are named base_*; works for uni- and bi-manual scenes).
        hand_qpos = []
        for jid in range(self.mj_model.njnt):
            jname = self.mj_model.joint(jid).name
            if jname.startswith("base_") or jname.startswith("key_"):
                continue
            hand_qpos.append(int(self.mj_model.joint(jid).qposadr[0]))
        self.hand_qpos_ids = np.array(hand_qpos)

    def _set_joint(self, name, val):
        try:
            self.init_qpos[self.mj_model.joint(name).qposadr[0]] = val
            return True
        except KeyError:
            return False

    def _apply_typing_posture(self):
        """Pronate the forearm(s) (palm down) + arch the knuckles so fingers
        press DOWN by flexion, and raise the base so fingertips hover above the
        keys. Works for uni- (pro_sup) and bi-manual (pro_sup + pro_sup_L, the
        left mirrored -> opposite sign)."""
        self._set_joint("pro_sup", self.pronate)
        self._set_joint("pro_sup_L", -self.pronate)   # left is mirrored
        for jid in range(self.mj_model.njnt):
            nm = self.mj_model.joint(jid).name
            if "mcp" in nm and "flex" in nm:
                self.init_qpos[self.mj_model.joint(jid).qposadr[0]] = self.knuckle_arch
        # start the base raised so fingertips hover just above the keys
        if len(self.base_qpos_ids):
            zids = self.base_qpos_ids[self.base_is_z]
            self.init_qpos[zids] = np.clip(
                self.hover_z,
                self.base_qpos_range[self.base_is_z][:, 0],
                self.base_qpos_range[self.base_is_z][:, 1],
            )
        # Passive wrist stiffness (like ligaments): spring the wrist joints toward
        # the pronated pose so the extrinsic finger flexors can press keys without
        # collapsing the wrist. Still fully muscle-drivable.
        if self.wrist_stiffness > 0:
            for name in ("pro_sup", "pro_sup_L", "flexion", "flexion_L",
                         "deviation", "deviation_L"):
                try:
                    jid = self.mj_model.joint(name).id
                except KeyError:
                    continue
                qadr = int(self.mj_model.jnt_qposadr[jid])
                for model in (self.mj_model, self.obsd_mj_model):
                    model.jnt_stiffness[jid] = self.wrist_stiffness
                    model.qpos_spring[qadr] = self.init_qpos[qadr]

    def _restrict_key_collisions(self):
        """Make keys collide ONLY with the fingertip geoms, so no other part of
        the hand can register a keypress. Uses contype/conaffinity bit 2: keys
        listen only on that bit; fingertip (distal-phalanx) geoms broadcast on
        it. Everything else keeps its default groups untouched."""
        # Collision bit scheme:
        #   fingertip geoms : contype=conaffinity=BIT  (interact ONLY on BIT)
        #   key geoms       : contype=0, conaffinity=BIT   (collide with fingertips)
        #   hand barrier    : contype=0, conaffinity=1 (XML) -> collides with every
        #                     other hand geom (default bit 1) but NOT fingertips,
        #                     stopping the palm/wrist at the key-top plane while
        #                     fingertips pass through to press keys.
        BIT = 4
        n_tip = 0
        for model in (self.mj_model, self.obsd_mj_model):
            tip_bodies = set(int(model.site_bodyid[model.site(s).id]) for s in self.TIP_SITES)
            for gid in range(model.ngeom):
                if int(model.geom_bodyid[gid]) in tip_bodies:
                    model.geom_contype[gid] = BIT
                    model.geom_conaffinity[gid] = BIT
                    n_tip += 1
                nm = model.geom(gid).name or ""
                if nm.startswith("key_") and nm.endswith("_geom"):
                    model.geom_contype[gid] = 0
                    model.geom_conaffinity[gid] = BIT
        self._n_fingertip_collision_geoms = n_tip // 2

    # ------------------------------------------------------------------ obs
    def _obs_common(self, mj_data):
        d = {}
        d["time"] = np.array([mj_data.time])
        d["qpos"] = mj_data.qpos[:].copy()
        d["qvel"] = mj_data.qvel[:].copy() * self.dt
        if self.mj_model.na > 0:
            d["act"] = mj_data.act[:].copy()

        tips = mj_data.site_xpos[self.ftip_sids].copy()          # (5,3)
        d["fingertip_pos"] = tips.ravel()
        # target position = the actual target key's site (on a moving slider
        # body, so its world xpos always tracks correctly).
        if self.target_key is not None:
            target_pos = mj_data.site_xpos[self.key_site_ids[self.target_key]].copy()
        else:
            target_pos = tips[0]
        # error from the *nearest* fingertip to the target key
        deltas = target_pos[None, :] - tips                     # (5,3)
        dists = np.linalg.norm(deltas, axis=-1)
        nearest = int(np.argmin(dists))
        d["reach_err"] = deltas[nearest]
        d["nearest_dist"] = np.array([dists[nearest]])

        # target one-hot (over targetable keys) so the policy knows what to press
        onehot = np.zeros(len(self.target_ids))
        if self.target_key is not None:
            pos = np.where(self.target_ids == self.target_key)[0]
            if len(pos):
                onehot[pos[0]] = 1.0
        d["target_onehot"] = onehot

        # fraction the target key is depressed
        if self.target_key is not None:
            qadr = self.key_jnt_qposids[self.target_key]
            frac = mj_data.qpos[qadr] / self.key_travel[self.target_key]
        else:
            frac = 0.0
        d["target_travel"] = np.array([np.clip(frac, 0.0, 1.5)])
        return d

    def get_obs_dict(self, mj_model, mj_data):
        return self._obs_common(mj_data)

    def get_obs_vec(self):
        self.obs_dict = self._obs_common(self.mj_data)
        _, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    # --------------------------------------------------------------- reward
    def get_reward_dict(self, obs_dict):
        # NOTE: env_base calls expand_dims() on obs_dict before this, so every
        # entry carries leading batch dims. Keep all reward terms at the same
        # (batched) shape -- do NOT squeeze -- exactly like reach_v0.
        reach_dist = np.linalg.norm(obs_dict["reach_err"], axis=-1)     # (...,)
        travel = obs_dict["target_travel"][..., 0]                      # (...,)
        act_mag = (
            np.linalg.norm(obs_dict["act"], axis=-1) / self.mj_model.na
            if self.mj_model.na != 0 else 0.0 * reach_dist
        )
        # disable the far-fail during the very first steps (settling)
        far_th = self.far_th if np.squeeze(obs_dict["time"]) > 2 * self.dt else np.inf
        on_key = reach_dist < 2 * self.reach_th
        pressed = travel > self.press_th

        rwd_dict = collections.OrderedDict((
            ("reach", -1.0 * reach_dist),
            ("press", np.clip(travel, 0.0, 1.0) * on_key),
            ("bonus", 1.0 * pressed),
            ("act_reg", -1.0 * act_mag),
            ("penalty", -1.0 * (reach_dist > far_th)),
            # must-have keys
            ("sparse", 1.0 * pressed),
            ("solved", pressed),
            ("done", reach_dist > far_th),
        ))
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    # ------------------------------------------------------------ dynamics
    def step(self, a, **kwargs):
        """Reproject base (non-muscle) actions from [-1,1] to their ctrlrange
        before delegating to BaseV0.step (which sigmoid-maps only muscles)."""
        a = np.asarray(a).copy()
        if self.normalize_act and len(self.base_act_ids):
            lo = self.base_ctrlrange[:, 0]
            hi = self.base_ctrlrange[:, 1]
            a[self.base_act_ids] = lo + (np.clip(a[self.base_act_ids], -1, 1) + 1) * 0.5 * (hi - lo)
        return super().step(a, **kwargs)

    # --------------------------------------------------------------- reset
    def _move_marker(self):
        """Move the visual mocap marker onto the current target key (both sims)."""
        key_pos = self.mj_data.site_xpos[self.key_site_ids[self.target_key]].copy()
        self.mj_data.mocap_pos[self.target_mocap_id] = key_pos
        self.obsd_mj_data.mocap_pos[self.target_mocap_id] = key_pos
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.obsd_mj_model, self.obsd_mj_data)

    def _randomized_init_qpos(self):
        """init_qpos with the hand's floating base start randomized (aiming task)."""
        qpos = self.init_qpos.copy()
        if self.base_init_noise <= 0 and self.base_init_raise <= 0:
            return qpos
        lo = self.base_qpos_range[:, 0]
        hi = self.base_qpos_range[:, 1]
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        # uniform noise around the range midpoint
        vals = mid + self.np_random.uniform(-1, 1, size=len(mid)) * self.base_init_noise * half
        # bias the vertical (base_z) start upward so the policy must descend
        if self.base_init_raise > 0 and np.any(self.base_is_z):
            vals[self.base_is_z] = hi[self.base_is_z] - (1 - self.base_init_raise) * (
                hi[self.base_is_z] - mid[self.base_is_z]
            )
        qpos[self.base_qpos_ids] = np.clip(vals, lo, hi)
        return qpos

    def reset(self, **kwargs):
        # Pick the target first so obs computed during super().reset() use it.
        self.target_key = int(self.np_random.choice(self.target_ids))
        if (self.base_init_noise > 0 or self.base_init_raise > 0) \
                and "reset_qpos" not in kwargs:
            kwargs["reset_qpos"] = self._randomized_init_qpos()
        obs = super().reset(**kwargs)
        # Base reset restored init pose; now position the visual marker.
        self._move_marker()
        return obs


class BimanualKeyboardEnvV0(KeyboardEnvV0):
    """Two MyoHands (right + mirrored left, 78 muscles) typing on a QWERTY
    keyboard. Identical task/reward to KeyboardEnvV0 but with ten fingertips
    (five per hand); reach uses the nearest of all ten. With both hands the
    whole keyboard is reachable, so the default targetable set is every key.

    Scene: myohand_keyboard_bimanual.xml. Action space nu = 84
    (39 right muscles + 39 left muscles + 6 base position servos).
    """

    TIP_SITES = [
        "THtip", "IFtip", "MFtip", "RFtip", "LFtip",
        "THtip_L", "IFtip_L", "MFtip_L", "RFtip_L", "LFtip_L",
    ]

    def _setup(self, target_keys=None, **kwargs):
        # Default: whole keyboard (both hands cover it).
        super()._setup(target_keys=target_keys, **kwargs)
