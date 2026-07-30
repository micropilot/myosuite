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
        "fingertip_pos", "finger_disp", "reach_err",
        "target_onehot", "finger_onehot", "target_travel",
        "released_seen",
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.5,       # ASSIGNED finger -> target key
        "press": 5.0,       # key depressed by the assigned finger
        "bonus": 3.0,
        "home": 1.0,        # keep the OTHER fingers near home (minimal/even motion)
        "act_reg": 1.0,
        "penalty": 10.0,
    }

    TIP_SITES = ["THtip", "IFtip", "MFtip", "RFtip", "LFtip"]

    # Touch-typing finger assignment: key -> (hand side, finger role).
    # Roles: 0 thumb, 1 index, 2 middle, 3 ring, 4 pinky. Each key is pressed by
    # its designated finger, forcing human-like multi-finger typing (rather than
    # sliding one finger to every key). Fingertip index = role for the right hand,
    # 5+role for the (mirrored) left hand in the bimanual scene.
    FINGER_ROLE = {
        # right hand
        "6": ("R", 1), "7": ("R", 1), "y": ("R", 1), "u": ("R", 1),
        "h": ("R", 1), "j": ("R", 1), "n": ("R", 1), "m": ("R", 1),
        "8": ("R", 2), "i": ("R", 2), "k": ("R", 2), "comma": ("R", 2),
        "9": ("R", 3), "o": ("R", 3), "l": ("R", 3), "period": ("R", 3),
        "0": ("R", 4), "p": ("R", 4), "semicolon": ("R", 4), "minus": ("R", 4),
        "equal": ("R", 4), "lbracket": ("R", 4), "rbracket": ("R", 4),
        "backslash": ("R", 4), "quote": ("R", 4), "slash": ("R", 4),
        "backspace": ("R", 4), "enter": ("R", 4), "shift_r": ("R", 4),
        # left hand
        "4": ("L", 1), "5": ("L", 1), "r": ("L", 1), "f": ("L", 1),
        "v": ("L", 1), "t": ("L", 1), "g": ("L", 1), "b": ("L", 1),
        "3": ("L", 2), "e": ("L", 2), "d": ("L", 2), "c": ("L", 2),
        "2": ("L", 3), "w": ("L", 3), "s": ("L", 3), "x": ("L", 3),
        "1": ("L", 4), "q": ("L", 4), "a": ("L", 4), "z": ("L", 4),
        "backtick": ("L", 4), "tab": ("L", 4), "caps_lock": ("L", 4),
        "shift": ("L", 4),
        # thumbs
        "space": ("R", 0),
    }

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
        release_th: float = 0.2,      # travel below which a key counts as released
        reach_th: float = 0.015,      # fingertip-key distance counted as "on key" (m)
        far_th: float = 0.30,         # episode-fail distance (m)
        sequence_length: int = 1,     # >1: type a sequence of keys in order (words)
        base_init_noise: float = 0.0,  # randomize hand start over [-1,1]*noise*range
        base_init_raise: float = 0.0,  # bias the base_z start upward by this fraction
        pronate: float = 1.4,          # forearm pronation (palm down); 0 disables
        knuckle_arch: float = 0.15,    # MCP flexion so fingers arch downward
        finger_spread: float = 0.3,    # MCP abduction to fan fingers across columns
        hover_z: float = 0.03,         # base_z init so fingertips hover above keys
        home_offset=(0.04, 0.04),      # (dx,dy) base shift so fingers rest on keys
        wrist_stiffness: float = 8.0,  # passive stiffness holding the wrist pose
        fingertip_only_collision: bool = True,  # only fingertips collide with keys
        reference_path=None,           # imitation: .npz (or list) of How-We-Type refs
        imitation_weight: float = 2.0,  # weight of the fingertip-tracking reward
        force_reference_path=None,     # TENDON-FORCE imitation: pose2tendon .npz (or list)
        force_imitation_weight: float = 2.0,  # weight of the muscle-force matching reward
        track_weight: float = 3.0,     # weight of the joint-angle (kinematic) tracking reward
        muscle_template_path=None,     # per-key HUMAN muscle template npz (build_muscle_template)
        muscle_match_weight: float = 0.0,  # weight pulling press activation -> human template
        template_min_count: int = 50,  # min human presses for a key's template to be used
        imitation_window: int = 6,     # keystrokes per imitation episode
        imitation_lead: float = 0.4,   # seconds of reference lead before 1st press
        press_grace: float = 0.15,     # s after a scheduled press before advancing
        playback_speed: float = 1.0,   # reference playback rate
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        frame_skip: int = 10,
        **kwargs,
    ):
        self.press_th = press_th
        self.release_th = release_th
        self.reach_th = reach_th
        self.far_th = far_th
        self.sequence_length = int(sequence_length)
        self.sequence = []
        self.seq_pos = 0
        self.released_seen = True
        self.base_init_noise = base_init_noise
        self.base_init_raise = base_init_raise
        self.pronate = pronate
        self.knuckle_arch = knuckle_arch
        self.finger_spread = finger_spread
        self.hover_z = hover_z
        self.home_offset = home_offset
        self.wrist_stiffness = wrist_stiffness
        self.fingertip_only_collision = fingertip_only_collision
        # per-key human muscle template (a light regularizer that makes a WORKING
        # typist's tendon forces imitate the human's per-key muscle coordination,
        # with no trajectory/phase alignment needed).
        self.muscle_template_path = muscle_template_path
        self.muscle_match_weight = muscle_match_weight
        self.template_min_count = int(template_min_count)
        # imitation setup. Two flavours share the keystroke-schedule machinery:
        #   pos_imitation   -> track human fingertip POSITIONS (anatomy-capped ~6cm)
        #   force_imitation -> match human muscle FORCES (pose2tendon; in the action
        #                      space, so reproducible exactly -- "proper" imitation)
        self.pos_imitation = reference_path is not None
        self.force_imitation = force_reference_path is not None
        self.imitation = self.pos_imitation or self.force_imitation
        self.imitation_weight = imitation_weight
        self.force_imitation_weight = force_imitation_weight
        self.track_weight = track_weight
        self.imitation_window = int(imitation_window)
        self.imitation_lead = imitation_lead
        self.press_grace = press_grace
        self.playback_speed = playback_speed
        self.motion_start_time = 0.0
        self.seq_finger = []
        self.cur_ref = None
        self.cur_force = None
        self.force_references = []
        self.track_qpos_ids = np.array([], dtype=int)
        self.track_slots = np.array([], dtype=int)
        if self.imitation:
            weighted_reward_keys = dict(weighted_reward_keys)
            # The key-press task is a secondary shaping signal; imitation dominates
            # (otherwise the policy trades imitation for pressing and the muscle
            # activations stop reflecting human coordination).
            weighted_reward_keys["home"] = 0.0   # reference already places all fingers
            weighted_reward_keys["reach"] = 0.5
            weighted_reward_keys["press"] = 1.0
            weighted_reward_keys["bonus"] = 0.5
            if self.pos_imitation:
                obs_keys = list(obs_keys) + ["ref_err"]
                weighted_reward_keys["imitate"] = imitation_weight
            if self.force_imitation:
                obs_keys = list(obs_keys) + ["ref_act"]
                weighted_reward_keys["imitate_force"] = force_imitation_weight
                weighted_reward_keys["act_reg"] = 0.0  # want human activation, not minimal
                if track_weight > 0:
                    # DeepMimic kinematic term: track the CLEAN human joint FLEXIONS
                    # (the reference joint angles, not pose2tendon's over-flexed
                    # output). On the pronated hand, finger flexion presses keys, so
                    # this is what actually makes the policy type.
                    weighted_reward_keys["track"] = track_weight
            self.sequence_length = max(self.sequence_length, self.imitation_window)

        if self.muscle_match_weight > 0:
            weighted_reward_keys = dict(weighted_reward_keys)
            weighted_reward_keys["muscle_match"] = muscle_match_weight

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

        # Touch-typing finger assignment (needs key_names + n_tips, both ready).
        # finger_home is a placeholder here so the obs probe inside super()._setup
        # works; the real home is captured after the typing posture is applied.
        n_tips = len(self.TIP_SITES)
        self.key_finger = np.array([
            self._finger_index(k, n_tips) for k in self.key_names
        ])
        self.finger_home = self.mj_data.site_xpos[self.ftip_sids].copy()

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

        # Capture each finger's home position at the neutral (post-posture) pose,
        # then assign each key to the geometrically nearest finger (a spatial
        # touch-typing map matched to THIS hand's actual rest layout). Space -> a
        # thumb; every other key -> nearest of the four long fingers on its side.
        self.finger_home = self._capture_finger_home()
        self.key_finger = self._assign_fingers_geometric()

        if self.pos_imitation:
            self._load_references(reference_path)
        if self.force_imitation:
            self._load_force_references(force_reference_path)
            self._setup_track_ids()
        if self.muscle_match_weight > 0 and self.muscle_template_path:
            self._load_muscle_template(self.muscle_template_path)

    def _load_muscle_template(self, path):
        """Per-key human muscle template -> (n_keys, n_muscle) aligned to key_names,
        with a mask for keys backed by >= template_min_count human presses."""
        nm = len(self.muscle_act_ids)
        z = np.load(path, allow_pickle=True)
        keys = [str(k) for k in z["keys"]]
        act = z["act"].astype(np.float32)          # (K, n_muscle)
        cnt = z["count"].astype(int)
        assert act.shape[1] == nm, f"template has {act.shape[1]} muscles, env has {nm}"
        lut = {k: (a, c) for k, a, c in zip(keys, act, cnt)}
        self.template_act = np.zeros((len(self.key_names), nm), np.float32)
        self.template_mask = np.zeros(len(self.key_names), dtype=bool)
        for kid, kname in enumerate(self.key_names):
            if kname in lut and lut[kname][1] >= self.template_min_count:
                self.template_act[kid] = lut[kname][0]
                self.template_mask[kid] = True
        print(f"[muscle-template] {int(self.template_mask.sum())}/{len(self.key_names)} "
              f"keys with >= {self.template_min_count} presses")

    def _load_force_references(self, force_reference_path):
        """Load pose2tendon FORCE references (build_force_reference.py npz). Each
        holds per-frame muscle controls act(N, n_muscle) on a uniform dt time base
        + a keystroke schedule. The muscle order must match this env's muscle
        actuators (unimanual: the 39 MyoHand muscles in model order)."""
        paths = force_reference_path if isinstance(force_reference_path, (list, tuple)) \
            else [force_reference_path]
        nm = len(self.muscle_act_ids)
        self.force_references = []
        for p in paths:
            z = np.load(p, allow_pickle=True)
            act = z["act"].astype(np.float32)              # (N, n_muscle)
            ja = z["joint_angles"].astype(np.float32)      # (N, 20) emg2pose order
            time = z["time"].astype(float)
            assert act.shape[1] == nm, \
                f"force ref has {act.shape[1]} muscles, env has {nm}"
            kp_key = [str(k) for k in z["kp_key"]]
            kp_time = z["kp_time"].astype(float)
            kp_finger = z["kp_finger"].astype(int)
            sched = [(t, self.key_names.index(k), f)
                     for t, k, f in zip(kp_time, kp_key, kp_finger)
                     if k in self.key_names]
            ref_dt = float(np.median(np.diff(time))) if len(time) > 1 else self.dt
            if len(sched) > self.imitation_window:
                self.force_references.append(
                    {"act": act, "ja": ja, "time": time, "dt": ref_dt, "sched": sched})
        assert self.force_references, "no usable force reference trials loaded"

    # emg2pose slot -> MyoHand FLEXION joint name (the kinematically meaningful
    # DoF; abduction slots are 0/unreliable in the IK, so excluded from tracking).
    _TRACK_FLEX = [
        (0, "cmc_flexion"), (2, "mp_flexion"), (3, "ip_flexion"),
        (5, "mcp2_flexion"), (6, "pm2_flexion"), (7, "md2_flexion"),
        (9, "mcp3_flexion"), (10, "pm3_flexion"), (11, "md3_flexion"),
        (13, "mcp4_flexion"), (14, "pm4_flexion"), (15, "md4_flexion"),
        (17, "mcp5_flexion"), (18, "pm5_flexion"), (19, "md5_flexion"),
    ]

    def _setup_track_ids(self):
        """qpos indices + reference slots for the flexion joints present in this
        model (right hand for unimanual; +_L variants would extend it)."""
        slots, qids = [], []
        for slot, name in self._TRACK_FLEX:
            try:
                qids.append(int(self.mj_model.joint(name).qposadr[0]))
                slots.append(slot)
            except KeyError:
                pass
        self.track_slots = np.array(slots, dtype=int)
        self.track_qpos_ids = np.array(qids, dtype=int)

    def _load_references(self, reference_path):
        """Load one or more How-We-Type reference trials (retargeted npz). Each
        holds fingertip trajectories (ReferenceMotion) + a keystroke schedule."""
        from myosuite.logger.reference_motion import ReferenceMotion
        paths = reference_path if isinstance(reference_path, (list, tuple)) else [reference_path]
        self.references = []
        for p in paths:
            z = np.load(p, allow_pickle=True)
            ref = ReferenceMotion(p, random_generator=getattr(self, "np_random", None))
            # map keystroke key names -> our key indices (drop keys not on this board)
            kp_key = [str(k) for k in z["kp_key"]]
            kp_time = z["kp_time"].astype(float)
            kp_finger = z["kp_finger"].astype(int)
            sched = [(t, self.key_names.index(k), f)
                     for t, k, f in zip(kp_time, kp_key, kp_finger)
                     if k in self.key_names]
            if len(sched) > self.imitation_window:
                self.references.append((ref, sched))
        assert self.references, "no usable reference trials loaded"

    def _finger_index(self, key, n_tips):
        side, role = self.FINGER_ROLE.get(key, ("R", 1))
        if n_tips > 5 and side == "L":
            return 5 + role
        return role

    def _capture_finger_home(self):
        """Fingertip world positions at the neutral (home) pose = init_qpos."""
        import mujoco as _mj
        d = _mj.MjData(self.mj_model)
        d.qpos[:] = self.init_qpos
        _mj.mj_forward(self.mj_model, d)
        return d.site_xpos[self.ftip_sids].copy()

    def _assign_fingers_geometric(self):
        """Assign each key to the nearest LONG finger (index/middle/ring/pinky)
        by home xy; space goes to the nearest thumb. Matches the hand's real rest
        layout so every key is a short reach for its finger (multi-finger typing,
        minimal even movement). Naturally keeps left keys on the left hand."""
        import mujoco as _mj
        d = _mj.MjData(self.mj_model)
        d.qpos[:] = self.init_qpos
        _mj.mj_forward(self.mj_model, d)
        home_xy = self.finger_home[:, :2]
        n_tips = len(self.ftip_sids)
        thumb_ids = [0] + ([5] if n_tips > 5 else [])
        long_ids = [i for i in range(n_tips) if i not in thumb_ids]
        assign = []
        for kid in range(len(self.key_names)):
            kxy = d.site_xpos[self.key_site_ids[kid]][:2]
            pool = thumb_ids if self.key_names[kid] == "space" else long_ids
            dists = [np.linalg.norm(home_xy[i] - kxy) for i in pool]
            assign.append(pool[int(np.argmin(dists))])
        return np.array(assign)

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
        # Fan the fingers out across columns (abduction) so each covers a distinct
        # region of the board -> balanced, minimal-movement multi-finger typing.
        spread_signs = {"mcp2_abduction": 1.0, "mcp3_abduction": 0.33,
                        "mcp4_abduction": -0.33, "mcp5_abduction": -1.0}
        for jid in range(self.mj_model.njnt):
            nm = self.mj_model.joint(jid).name
            base = nm[:-2] if nm.endswith("_L") else nm
            if base in spread_signs:
                sign = spread_signs[base]
                if nm.endswith("_L"):
                    sign = -sign   # mirrored hand fans the other way
                lo, hi = self.mj_model.jnt_range[jid]
                val = float(np.clip(sign * self.finger_spread, lo, hi))
                self.init_qpos[self.mj_model.joint(jid).qposadr[0]] = val
        # start the base raised so fingertips hover just above the keys
        if len(self.base_qpos_ids):
            zids = self.base_qpos_ids[self.base_is_z]
            self.init_qpos[zids] = np.clip(
                self.hover_z,
                self.base_qpos_range[self.base_is_z][:, 0],
                self.base_qpos_range[self.base_is_z][:, 1],
            )
        # Shift each hand so its fingers rest over the keys (right hand +dx, the
        # mirrored left hand -dx; both shifted +dy back onto the home row).
        dx, dy = self.home_offset
        for jn, off in (("base_x", dx), ("base_x_R", dx), ("base_x_L", -dx),
                        ("base_y", dy), ("base_y_R", dy), ("base_y_L", dy)):
            try:
                jid = self.mj_model.joint(jn).id
            except KeyError:
                continue
            qadr = int(self.mj_model.jnt_qposadr[jid])
            lo, hi = self.mj_model.jnt_range[jid]
            self.init_qpos[qadr] = float(np.clip(self.init_qpos[qadr] + off, lo, hi))
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
    def _assigned_finger(self):
        """Fingertip index that should press the current key: from the mocap
        keystroke schedule under imitation, else the geometric touch-typing map."""
        if self.imitation and self.seq_finger and self.seq_pos < len(self.seq_finger):
            f = int(self.seq_finger[self.seq_pos])
            if 0 <= f < len(self.ftip_sids):
                return f
        return int(self.key_finger[self.target_key]) if self.target_key is not None else 0

    def _ref_fingertips(self):
        """Retargeted human reference fingertip positions at the current phase.
        The reference is sampled at exactly the sim dt, so we index the frame
        directly (MyoSuite's get_reference between-frame blend is numerically
        unstable for this data)."""
        phase = self.motion_start_time + self.mj_data.time * self.playback_speed
        robot = self.cur_ref.reference["robot"]
        i = int(round(phase / self.dt))
        i = max(0, min(i, len(robot) - 1))
        pts = np.asarray(robot[i]).reshape(-1, 3)     # (10,3) world
        return pts[:len(self.ftip_sids)]

    def _ref_force(self):
        """pose2tendon reference muscle control at the current phase (n_muscle,).
        Indexed on the reference's own uniform dt (the pose2tendon resample dt),
        not the env step dt."""
        phase = self.motion_start_time + self.mj_data.time * self.playback_speed
        act = self.cur_force["act"]
        i = int(round(phase / self.cur_force["dt"]))
        i = max(0, min(i, len(act) - 1))
        return act[i]

    def _ref_ja(self):
        """Reference human joint angles (20 emg2pose slots) at the current phase."""
        phase = self.motion_start_time + self.mj_data.time * self.playback_speed
        ja = self.cur_force["ja"]
        i = int(round(phase / self.cur_force["dt"]))
        i = max(0, min(i, len(ja) - 1))
        return ja[i]

    def _obs_common(self, mj_data):
        d = {}
        d["time"] = np.array([mj_data.time])
        d["qpos"] = mj_data.qpos[:].copy()
        d["qvel"] = mj_data.qvel[:].copy() * self.dt
        if self.mj_model.na > 0:
            d["act"] = mj_data.act[:].copy()

        n_tips = len(self.ftip_sids)
        tips = mj_data.site_xpos[self.ftip_sids].copy()          # (n_tips,3)
        d["fingertip_pos"] = tips.ravel()
        # displacement of each fingertip from its home position (for the movement
        # penalty / so the policy can bring fingers back home).
        d["finger_disp"] = (tips - self.finger_home).ravel()

        # target position = the actual target key's site (on a moving slider
        # body, so its world xpos always tracks correctly). Assigned finger comes
        # from the mocap schedule under imitation, else the geometric map.
        if self.target_key is not None:
            target_pos = mj_data.site_xpos[self.key_site_ids[self.target_key]].copy()
            fa = self._assigned_finger()
        else:
            target_pos = tips[0]
            fa = 0
        # error is measured for the ASSIGNED finger (touch typing), not nearest.
        d["reach_err"] = target_pos - tips[fa]
        d["assigned_dist"] = np.array([np.linalg.norm(target_pos - tips[fa])])

        # position imitation: per-finger error to the human reference fingertips
        if self.pos_imitation:
            if self.cur_ref is not None:
                ref = self._ref_fingertips()               # (n_tips,3)
                err = ref - tips
            else:                                          # obs probe before a ref is active
                err = np.zeros_like(tips)
            d["ref_err"] = err.ravel()
            d["ref_dist"] = np.array([float(np.mean(np.linalg.norm(err, axis=-1)))])
        # force imitation: reference muscle control (obs target) + realized-vs-ref
        # mean-squared control error (reward driver).
        if self.force_imitation:
            ref_act = self._ref_force() if self.cur_force is not None \
                else np.zeros(len(self.muscle_act_ids), np.float32)
            # MyoSuite drives muscles via the activation STATE (mj_data.act), which
            # is also what pose2tendon's _act.bin records and what EMG reflects.
            realized = mj_data.act[self.muscle_act_ids].copy()
            d["ref_act"] = ref_act
            d["force_err"] = np.array([float(np.mean((realized - ref_act) ** 2))])
            # kinematic tracking: realized vs reference human joint flexions.
            # Always present (0 before track ids are set / no active ref) so the
            # 'track' reward key never goes missing.
            terr = 0.0
            if len(self.track_qpos_ids) and self.cur_force is not None:
                ref_flex = self._ref_ja()[self.track_slots]
                cur_flex = mj_data.qpos[self.track_qpos_ids]
                terr = float(np.mean((cur_flex - ref_flex) ** 2))
            d["track_err"] = np.array([terr])

        # target one-hot (over targetable keys) so the policy knows what to press
        onehot = np.zeros(len(self.target_ids))
        if self.target_key is not None:
            pos = np.where(self.target_ids == self.target_key)[0]
            if len(pos):
                onehot[pos[0]] = 1.0
        d["target_onehot"] = onehot
        # which finger should press it
        fonehot = np.zeros(n_tips)
        fonehot[fa] = 1.0
        d["finger_onehot"] = fonehot
        # mean displacement of the NON-assigned fingers from home (movement pen.)
        mask = np.ones(n_tips, dtype=bool); mask[fa] = False
        other_disp = np.linalg.norm((tips - self.finger_home)[mask], axis=-1)
        d["other_finger_disp"] = np.array([float(np.mean(other_disp))])

        # fraction the target key is depressed
        if self.target_key is not None:
            qadr = self.key_jnt_qposids[self.target_key]
            frac = mj_data.qpos[qadr] / self.key_travel[self.target_key]
        else:
            frac = 0.0
        d["target_travel"] = np.array([np.clip(frac, 0.0, 1.5)])

        # Release-gate latch (the one bit of hidden sequence state): whether the
        # CURRENT key has been seen released (travel < release_th) since it became
        # active. This is an unbounded-horizon OR-accumulator over travel history,
        # so it is NOT recoverable from the current travel nor a short frame stack
        # -- expose it directly so a distinct repeated keystroke is fully
        # observable. Always True (1.0) outside sequence mode, where it is inert.
        d["released_seen"] = np.array([1.0 if self.released_seen else 0.0])
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
        reach_dist = np.linalg.norm(obs_dict["reach_err"], axis=-1)     # assigned finger
        travel = obs_dict["target_travel"][..., 0]                      # (...,)
        other_disp = obs_dict["other_finger_disp"][..., 0]             # non-assigned fingers
        act_mag = (
            np.linalg.norm(obs_dict["act"], axis=-1) / self.mj_model.na
            if self.mj_model.na != 0 else 0.0 * reach_dist
        )
        # disable the far-fail during the very first steps (settling)
        far_th = self.far_th if np.squeeze(obs_dict["time"]) > 2 * self.dt else np.inf
        on_key = reach_dist < 2 * self.reach_th
        pressed = (travel > self.press_th) & on_key   # pressed BY the assigned finger

        rwd_dict = collections.OrderedDict((
            ("reach", -1.0 * reach_dist),
            ("press", np.clip(travel, 0.0, 1.0) * on_key),
            ("bonus", 1.0 * pressed),
            ("home", -1.0 * other_disp),   # keep other fingers home -> minimal, even motion
            ("act_reg", -1.0 * act_mag),
            ("penalty", -1.0 * (reach_dist > far_th)),
            # must-have keys
            ("sparse", 1.0 * pressed),
            ("solved", pressed),
            ("done", reach_dist > far_th),
        ))
        if self.pos_imitation:
            # DeepMimic-style tracking: exp(-k * mean fingertip error to the human
            # reference). Bounded [0,1], strong gradient near the reference.
            rwd_dict["imitate"] = np.exp(-30.0 * obs_dict["ref_dist"][..., 0])
        if self.force_imitation:
            # DeepMimic in ACTION space: exp(-k * mean-squared muscle-control error
            # to the pose2tendon reference. Bounded [0,1]; the optimal action is
            # literally the reference force, so this is directly reproducible.
            rwd_dict["imitate_force"] = np.exp(-20.0 * obs_dict["force_err"][..., 0])
            if "track_err" in obs_dict:
                rwd_dict["track"] = np.exp(-10.0 * obs_dict["track_err"][..., 0])
        if self.muscle_match_weight > 0:
            # pull the current key's press activation toward the human template
            tm = getattr(self, "template_mask", None)
            if tm is not None and self.target_key is not None and tm[self.target_key] \
                    and "act" in obs_dict:
                mse = np.mean((obs_dict["act"] - self.template_act[self.target_key]) ** 2,
                              axis=-1)
                rwd_dict["muscle_match"] = np.exp(-20.0 * mse)
            else:
                rwd_dict["muscle_match"] = 0.0 * reach_dist
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
        obs, rew, term, trunc, info = super().step(a, **kwargs)
        if self.imitation and self.sequence:
            rew, term, info = self._advance_imitation(rew, term, info)
        elif self.sequence_length > 1 and self.sequence:
            rew, term, info = self._advance_sequence(rew, term, info)
        if self.imitation:
            info = dict(info)
            if self.pos_imitation:
                info["imitate"] = float(np.squeeze(self.rwd_dict.get("imitate", 0.0)))
                info["ref_dist"] = float(np.squeeze(self.obs_dict.get("ref_dist", 0.0)))
            if self.force_imitation:
                info["imitate_force"] = float(np.squeeze(self.rwd_dict.get("imitate_force", 0.0)))
                info["force_err"] = float(np.squeeze(self.obs_dict.get("force_err", 0.0)))
                info["track"] = float(np.squeeze(self.rwd_dict.get("track", 0.0)))
        if self.muscle_match_weight > 0:
            info = dict(info)
            info["muscle_match"] = float(np.squeeze(self.rwd_dict.get("muscle_match", 0.0)))
        return obs, rew, term, trunc, info

    def _key_travel(self, kid):
        return float(self.mj_data.qpos[self.key_jnt_qposids[kid]] / self.key_travel[kid])

    def _advance_sequence(self, rew, term, info):
        """When the current key is pressed by its assigned finger (after having
        been released -> a distinct keystroke), advance to the next key. Reward a
        completion bonus; finish the episode when the whole word is typed."""
        tk = self.target_key
        travel = self._key_travel(tk)
        tip = self.mj_data.site_xpos[self.ftip_sids[self._assigned_finger()]]
        kpos = self.mj_data.site_xpos[self.key_site_ids[tk]]
        on_key = np.linalg.norm(kpos - tip) < 2 * self.reach_th
        if travel < self.release_th:
            self.released_seen = True
        if self.released_seen and travel > self.press_th and on_key:
            self.seq_pos += 1
            rew = float(rew) + 5.0  # keystroke completed
            if self.seq_pos >= len(self.sequence):
                term = True
            else:
                self.target_key = self.sequence[self.seq_pos]
                self._move_marker()
                self.released_seen = self._key_travel(self.target_key) < self.release_th
        info = dict(info)
        info["seq_pos"] = self.seq_pos
        info["seq_len"] = len(self.sequence)
        # 'solved' now means the WHOLE word was typed (override the per-key value)
        info["solved"] = self.seq_pos >= len(self.sequence)
        return rew, term, info

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
        # Pick the target(s) first so obs computed during super().reset() use them.
        if self.imitation:
            self._reset_imitation_window()
        elif self.sequence_length > 1:
            self.sequence = [int(self.np_random.choice(self.target_ids))
                             for _ in range(self.sequence_length)]
            self.seq_pos = 0
            self.target_key = self.sequence[0]
        else:
            self.target_key = int(self.np_random.choice(self.target_ids))
        if (self.base_init_noise > 0 or self.base_init_raise > 0) \
                and "reset_qpos" not in kwargs:
            kwargs["reset_qpos"] = self._randomized_init_qpos()
        obs = super().reset(**kwargs)
        # Base reset restored init pose; now position the visual marker.
        self._move_marker()
        self.released_seen = True  # hand at home, all keys released
        return obs

    def _reset_imitation_window(self):
        """Pick a reference trial and a random window of `imitation_window`
        keystrokes; the reference plays from `imitation_lead` s before the first
        keystroke (so the hands can approach), phase-locked to sim time. The
        target key is driven by the reference PHASE (advances when the reference
        passes each keystroke's scheduled time), keeping imitation + task aligned."""
        if self.force_imitation:
            idx = int(self.np_random.integers(len(self.force_references)))
            self.cur_force = self.force_references[idx]
            self.cur_ref, sched = None, self.cur_force["sched"]
        else:
            idx = int(self.np_random.integers(len(self.references)))
            self.cur_ref, sched = self.references[idx]
            self.cur_force = None
        k0 = int(self.np_random.integers(0, len(sched) - self.imitation_window))
        window = sched[k0:k0 + self.imitation_window]
        self.sequence = [k for (_, k, _) in window]
        self.seq_finger = [f for (_, _, f) in window]
        self.window_kp_times = np.array([tm for (tm, _, _) in window])
        self.seq_pos = 0
        self.n_hit = 0
        self.cur_target_pressed = False
        self.target_key = self.sequence[0]
        self.motion_start_time = window[0][0] - self.imitation_lead

    def _advance_imitation(self, rew, term, info):
        """Phase-driven keystroke schedule: the target key follows the reference
        (advances when the reference passes each keystroke's time). Rewards the
        policy for pressing the current key while it is active; counts a hit if
        it does so before the reference moves on."""
        phase = self.motion_start_time + self.mj_data.time * self.playback_speed
        tk = self.target_key
        if self._key_travel(tk) > self.press_th and not self.cur_target_pressed:
            self.cur_target_pressed = True
            rew = float(rew) + 1.0
        # commit keystrokes whose scheduled time (+grace) has passed
        while (self.seq_pos < len(self.sequence)
               and phase >= self.window_kp_times[self.seq_pos] + self.press_grace):
            if self.cur_target_pressed:
                self.n_hit += 1
            self.seq_pos += 1
            self.cur_target_pressed = False
            if self.seq_pos < len(self.sequence):
                self.target_key = self.sequence[self.seq_pos]
                self._move_marker()
        if self.seq_pos >= len(self.sequence):
            term = True
        info = dict(info)
        info["seq_pos"] = self.n_hit          # keystrokes hit on schedule
        info["seq_len"] = len(self.sequence)
        info["solved"] = self.n_hit == len(self.sequence)
        return rew, term, info


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
