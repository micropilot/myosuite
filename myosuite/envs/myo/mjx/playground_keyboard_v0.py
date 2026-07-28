"""GPU (MJX / MuJoCo-Warp) single-key keyboard-press task for the MyoHand.

Milestone-1 GPU port of `myosuite.envs.myo.myobase.keyboard_v0.KeyboardEnvV0`
(SINGLE-KEY mode, sequence_length=1). One right MyoHand (39 muscles) on a 3-DoF
prismatic base hovers over a small QWERTY keyboard; each episode a target key is
sampled. Reward = bring the ASSIGNED fingertip onto the target key (reach) +
depress it past `press_th` of its travel (press) + a press bonus, minus muscle
effort and an other-finger-movement penalty. Episode success = key pressed by
the assigned finger.

Everything the CPU env did with mutable Python state at build time is baked into
the compiled MuJoCo model / constant arrays here so the per-step logic is a pure
JAX function that vmaps across thousands of parallel envs:
  * fingertip-only contact filtering (bit-4 scheme) + wrap-geom contact disable,
    baked into `geom_contype/conaffinity` before `mjx.put_model` (so `put_model`
    succeeds and only fingertip-mesh x key-box contacts remain);
  * typing posture (pronation, knuckle arch, finger spread, base hover/offset)
    and passive wrist stiffness baked into the compiled model + init qpos;
  * geometric finger->key assignment precomputed into a constant array.
The per-episode target key lives in the per-env `info` dict and is resampled
functionally on reset (mirrors the shipped `MjxReachEnvV0` reset pattern).
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import State
from mujoco_playground._src import mjx_env

from myosuite.envs.myo.mjx.mjx_base_env import MjxMyoBase, make_data


class MjxKeyboardEnvV0(MjxMyoBase):
    # Candidate fingertip sites; the left (`_L`) set is present only in the
    # bimanual scene. The env auto-detects which exist -> works for uni & bi.
    _RIGHT_TIPS = ["THtip", "IFtip", "MFtip", "RFtip", "LFtip"]
    _LEFT_TIPS = ["THtip_L", "IFtip_L", "MFtip_L", "RFtip_L", "LFtip_L"]

    # fingertip-only collision bit (keys listen only on this bit; fingertips
    # broadcast on it). Matches KeyboardEnvV0._restrict_key_collisions.
    _COLLISION_BIT = 4

    def __init__(
        self,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        # Replicate MjxMyoBase.__init__ but insert model edits (contact filtering,
        # typing posture, wrist stiffness) BEFORE mjx.put_model so put_model
        # succeeds on the contact-rich keyboard and the sim runs the right pose.
        mjx_env.MjxEnv.__init__(self, config, config_overrides)

        spec = mujoco.MjSpec.from_file(config.model_path.as_posix())
        self.impl = self._config.impl
        spec = self.preprocess_spec(spec)
        self._mj_spec = spec
        self._mj_model = spec.compile()

        self._discover_keyboard(self._mj_model)
        self._apply_contact_filtering(self._mj_model)
        self._bake_typing_posture(self._mj_model)  # sets self._init_qpos, stiffness
        self._assign_fingers_geometric(self._mj_model)  # sets self._key_finger, home

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = config.model_path.as_posix()
        self._n_substeps = int(config.ctrl_dt / config.sim_dt)

        # muscle vs base-servo actuator split (base position servos need a linear
        # [-1,1]->ctrlrange map, NOT the muscle sigmoid).
        is_muscle = (
            self._mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        ).astype(np.float32)
        self._muscle_mask = jp.asarray(is_muscle)
        self._base_mask = jp.asarray(1.0 - is_muscle)
        ctrl = self._mj_model.actuator_ctrlrange
        self._base_lo = jp.asarray(ctrl[:, 0])
        self._base_hi = jp.asarray(ctrl[:, 1])

        # base (slide) joint qpos indices + ranges for start-state randomization.
        base_qpos, base_rng, base_is_z = [], [], []
        for jid in range(self._mj_model.njnt):
            jname = self._mj_model.joint(jid).name
            if jname.startswith("base_"):
                base_qpos.append(int(self._mj_model.joint(jid).qposadr[0]))
                base_rng.append(self._mj_model.jnt_range[jid].copy())
                base_is_z.append("_z" in jname or jname.endswith("z"))
        self._base_qpos_ids = jp.asarray(np.array(base_qpos), dtype=jp.int32)
        rng_arr = np.array(base_rng) if base_rng else np.zeros((0, 2))
        self._base_qpos_lo = jp.asarray(rng_arr[:, 0])
        self._base_qpos_hi = jp.asarray(rng_arr[:, 1])
        self._base_is_z = jp.asarray(np.array(base_is_z, dtype=bool))

        self._na = int(self._mj_model.na)
        # muscle-template imitation reward (milestone 2c): pull the current
        # target key's muscle activation toward the per-key HUMAN template.
        self._load_muscle_template()
        # human muscle-synergy action space (Task 1): if enabled, the policy acts
        # in a low-D human synergy latent (per hand) decoded to muscle activations.
        self._setup_synergy()
        self._far_th = float(self._config.far_th)
        self._press_th = float(self._config.press_th)
        self._reach_th = float(self._config.reach_th)
        self._release_th = float(self._config.get("release_th", 0.2))
        self._n_targets = int(self._target_ids.shape[0])
        # sequence (word-typing) mode: type `sequence_length` keys in order, each
        # a debounced keystroke (press-after-release). 1 -> single-key hold task.
        self._seq_len = int(self._config.get("sequence_length", 1))
        self._is_seq = self._seq_len > 1

    # ------------------------------------------------------------- build-time
    def _discover_keyboard(self, model):
        """Introspect key bodies produced by make_keyboard.py -> constant arrays."""
        key_names, key_site_ids, key_qposadr, key_travel = [], [], [], []
        for bid in range(model.nbody):
            name = model.body(bid).name
            if not name.startswith("key_") or name == "key_target":
                continue
            key = name[len("key_"):]
            jnt = model.joint("key_%s_slide" % key)
            key_names.append(key)
            key_qposadr.append(int(jnt.qposadr[0]))
            key_travel.append(float(jnt.range[1]))
            key_site_ids.append(model.site("key_%s_site" % key).id)
        self.key_names = key_names
        self._key_site_ids = jp.asarray(np.array(key_site_ids), dtype=jp.int32)
        self._key_qposadr = jp.asarray(np.array(key_qposadr), dtype=jp.int32)
        self._key_travel = jp.asarray(np.array(key_travel))
        self._key_site_ids_np = np.array(key_site_ids)
        self._key_qposadr_np = np.array(key_qposadr)

        # auto-detect fingertip sites present in the model (5 uni / 10 bimanual).
        def _has_site(s):
            try:
                model.site(s); return True
            except KeyError:
                return False
        self._tip_site_names = [s for s in (self._RIGHT_TIPS + self._LEFT_TIPS)
                                if _has_site(s)]
        self._n_tips = len(self._tip_site_names)
        self._tip_sids_np = np.array([model.site(s).id for s in self._tip_site_names])
        self._tip_sids = jp.asarray(self._tip_sids_np, dtype=jp.int32)

        # targetable subset (indices into key_names)
        target_keys = self._config.get("target_keys", None)
        if target_keys is None or len(target_keys) == 0:
            target_ids = np.arange(len(key_names))
        else:
            target_ids = np.array([key_names.index(k) for k in target_keys])
        self._target_ids = jp.asarray(target_ids, dtype=jp.int32)

    def _apply_contact_filtering(self, model):
        """Fingertip-only key collisions: fingertip geoms broadcast on bit-4, key
        geoms listen on bit-4, and muscle-wrap ellipsoid/cylinder geoms have all
        contacts disabled (they would otherwise form unsupported ellipsoid/cyl x
        box pairs that MJX rejects). Mirrors KeyboardEnvV0._restrict_key_collisions
        plus MjxMyoBase's wrap-geom disabling."""
        BIT = self._COLLISION_BIT
        tip_bodies = set(
            int(model.site_bodyid[model.site(s).id]) for s in self._tip_site_names
        )
        n_tip = 0
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) in tip_bodies:
                model.geom_contype[gid] = BIT
                model.geom_conaffinity[gid] = BIT
                n_tip += 1
            nm = model.geom(gid).name or ""
            if nm.startswith("key_") and nm.endswith("_geom"):
                model.geom_contype[gid] = 0
                model.geom_conaffinity[gid] = BIT
            gtype = int(model.geom_type[gid])
            if gtype in (mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_CYLINDER):
                model.geom_contype[gid] = 0
                model.geom_conaffinity[gid] = 0
        self._n_fingertip_collision_geoms = n_tip

    def _set_joint(self, model, init_qpos, name, val):
        try:
            init_qpos[int(model.joint(name).qposadr[0])] = val
            return True
        except KeyError:
            return False

    def _bake_typing_posture(self, model):
        """Pronate the forearm (palm down) + arch knuckles + fan fingers + raise
        the base so fingertips hover over the keys, and set passive wrist
        stiffness. Baked into a constant init qpos + the compiled model. Mirrors
        KeyboardEnvV0._apply_typing_posture."""
        c = self._config
        init_qpos = np.array(model.qpos0, dtype=np.float64).copy()

        # pronation: right forearm palm-down; the mirrored left forearm gets the
        # opposite sign (bimanual scene has pro_sup_L).
        self._set_joint(model, init_qpos, "pro_sup", c.pronate)
        self._set_joint(model, init_qpos, "pro_sup_L", -c.pronate)
        for jid in range(model.njnt):
            nm = model.joint(jid).name
            if "mcp" in nm and "flex" in nm:
                init_qpos[int(model.joint(jid).qposadr[0])] = c.knuckle_arch
        # fan fingers across columns; the mirrored left hand fans the other way.
        spread_signs = {"mcp2_abduction": 1.0, "mcp3_abduction": 0.33,
                        "mcp4_abduction": -0.33, "mcp5_abduction": -1.0}
        for jid in range(model.njnt):
            nm = model.joint(jid).name
            base = nm[:-2] if nm.endswith("_L") else nm
            if base in spread_signs:
                sign = spread_signs[base]
                if nm.endswith("_L"):
                    sign = -sign
                lo, hi = model.jnt_range[jid]
                val = float(np.clip(sign * c.finger_spread, lo, hi))
                init_qpos[int(model.joint(jid).qposadr[0])] = val

        # base start: raise base_z to hover_z; shift base_x/base_y by home_offset
        # (right hand +dx, the mirrored left hand -dx; both +dy onto the home row).
        dx, dy = c.home_offset
        for jid in range(model.njnt):
            jname = model.joint(jid).name
            if not jname.startswith("base_"):
                continue
            qadr = int(model.joint(jid).qposadr[0])
            lo, hi = model.jnt_range[jid]
            if "_z" in jname or jname.endswith("z"):
                init_qpos[qadr] = float(np.clip(c.hover_z, lo, hi))
            elif "_x" in jname or jname.endswith("x"):
                off = -dx if jname.endswith("_L") else dx
                init_qpos[qadr] = float(np.clip(init_qpos[qadr] + off, lo, hi))
            elif "_y" in jname or jname.endswith("y"):
                init_qpos[qadr] = float(np.clip(init_qpos[qadr] + dy, lo, hi))

        # passive wrist stiffness (springs the wrist toward the pronated pose so
        # extrinsic finger flexors can press keys without collapsing the wrist).
        if c.wrist_stiffness > 0:
            for name in ("pro_sup", "pro_sup_L", "flexion", "flexion_L",
                         "deviation", "deviation_L"):
                try:
                    jid = model.joint(name).id
                except KeyError:
                    continue
                qadr = int(model.jnt_qposadr[jid])
                model.jnt_stiffness[jid] = c.wrist_stiffness
                model.qpos_spring[qadr] = init_qpos[qadr]

        self._init_qpos = jp.asarray(init_qpos)
        self._init_qpos_np = init_qpos

    def _assign_fingers_geometric(self, model):
        """Assign each key to the nearest LONG finger (index/middle/ring/pinky) by
        home xy; space -> nearest thumb. Computed once at the posture pose.
        Mirrors KeyboardEnvV0._assign_fingers_geometric."""
        d = mujoco.MjData(model)
        d.qpos[:] = self._init_qpos_np
        mujoco.mj_forward(model, d)
        finger_home = d.site_xpos[self._tip_sids_np].copy()
        home_xy = finger_home[:, :2]
        n_tips = len(self._tip_sids_np)
        # thumbs press space; the four long fingers press everything else. Bimanual
        # adds a second thumb at index 5. Long fingers naturally split L/R by xy.
        thumb_ids = [0] + ([5] if n_tips > 5 else [])
        long_ids = [i for i in range(n_tips) if i not in thumb_ids]
        assign = []
        for kid, kname in enumerate(self.key_names):
            kxy = d.site_xpos[self._key_site_ids_np[kid]][:2]
            pool = thumb_ids if kname == "space" else long_ids
            dists = [np.linalg.norm(home_xy[i] - kxy) for i in pool]
            assign.append(pool[int(np.argmin(dists))])
        self._key_finger = jp.asarray(np.array(assign), dtype=jp.int32)
        self._key_finger_np = np.array(assign)
        self._finger_home = jp.asarray(finger_home)

    def _load_muscle_template(self):
        """Per-key HUMAN muscle template -> static (n_keys, na) target-activation
        array (indexed by MODEL key id) + a per-muscle validity mask (n_keys, na)
        + a per-key gate (n_keys,) for keys backed by >= template_min_count human
        presses. Mirrors KeyboardEnvV0._load_muscle_template. Inert (weight 0)
        unless configured.

        The 39-muscle order of the How-We-Type template matches this env's muscle
        actuators. For the BIMANUAL scene (na=78) the RIGHT-hand template fills
        actuators 0..38 and the LEFT-hand template (`muscle_template_path_left`)
        fills actuators 39..77 (identical mirrored muscle order). Each board key is
        matched only on the 39 muscles of the hand that presses it (from the
        geometric finger assignment: tips 0..4 = right, 5..9 = left) -> the idle
        hand's muscles are excluded from `muscle_match`. Single-hand scenes match
        all 39 right muscles (unchanged from milestone 2)."""
        rc = self._config.reward_config
        self._muscle_match_weight = float(rc.get("muscle_match_weight", 0.0))
        n_keys = len(self.key_names)
        self._template_act = jp.zeros((n_keys, self._na))
        self._template_mvalid = jp.zeros((n_keys, self._na))
        self._template_mask = jp.zeros((n_keys,))
        self._template_nm = 0
        min_count = int(self._config.get("template_min_count", 50))

        # (npz path, muscle-block offset, hand tag). Right block -> actuators
        # 0..38; left block (bimanual only) -> actuators 39..77.
        sources = []
        rpath = self._config.get("muscle_template_path", "")
        if rpath:
            sources.append((rpath, 0, "R"))
        lpath = self._config.get("muscle_template_path_left", "")
        if lpath and self._na >= 78:
            sources.append((lpath, 39, "L"))
        if not sources:
            return

        # per-key hand: single-hand scene -> all right; bimanual -> tip>=5 is left.
        is_bimanual = self._na >= 78
        key_hand = np.where(self._key_finger_np >= 5, "L", "R") if is_bimanual \
            else np.array(["R"] * n_keys)

        tmpl = np.zeros((n_keys, self._na), np.float32)
        mvalid = np.zeros((n_keys, self._na), np.float32)
        mask = np.zeros(n_keys, np.float32)
        n_used, matched_nm = 0, 0
        for path, off, hand in sources:
            z = np.load(path, allow_pickle=True)
            keys = [str(k) for k in z["keys"]]
            act = z["act"].astype(np.float32)      # (K, 39)
            cnt = z["count"].astype(int)
            nm = act.shape[1]
            matched_nm = nm
            assert off + nm <= self._na, f"template block {off}+{nm} > na={self._na}"
            lut = {k: (a, c) for k, a, c in zip(keys, act, cnt)}
            for kid, kname in enumerate(self.key_names):
                if key_hand[kid] != hand:
                    continue
                if kname in lut and lut[kname][1] >= min_count:
                    tmpl[kid, off:off + nm] = lut[kname][0]
                    mvalid[kid, off:off + nm] = 1.0
                    mask[kid] = 1.0
                    n_used += 1
        self._template_act = jp.asarray(tmpl)
        self._template_mvalid = jp.asarray(mvalid)
        self._template_mask = jp.asarray(mask)
        self._template_nm = matched_nm
        print(f"[mjx muscle-template] {n_used}/{n_keys} keys with >= {min_count} "
              f"human presses ({'bimanual R+L' if len(sources) > 1 else 'right'}); "
              f"per-key matching {matched_nm} muscles of the pressing hand; "
              f"weight={self._muscle_match_weight}")

    def _setup_synergy(self):
        """Fixed linear HUMAN muscle-synergy action space (Task 1). When enabled,
        the brax policy outputs [z (K per hand), base] instead of raw muscles.
        Each hand's K-latent is decoded to 39 human muscle activations by

            act = clip(mean + components[:K].T @ (z * scale), eps, 1-eps)

        (mean/components from PCA on How-We-Type right-hand `data.act`; `scale` =
        span * between-key std of the human per-key template in synergy space, so
        z in [-1,1] spans the human key-conditional range). The activation is then
        inverse-sigmoid'd to the env's native muscle action so the env's own
        `norm_actions` sigmoid recovers exactly `act`. This is a FIXED matmul on
        the policy output -- identical to the CPU `synergy_action.build_decoder`.
        For the bimanual scene (78 muscles) the same 39-muscle basis is applied
        block-wise to each hand (z has 2*K latents)."""
        self._syn_on = bool(self._config.get("synergy", False))
        if not self._syn_on:
            return
        # GENERATIVE decoder branch: a nonlinear MLP g(z)->39 activations instead of
        # the linear basis. Applied block-wise per hand for the bimanual scene.
        dec_path = self._config.get("synergy_decoder_path", "")
        self._syn_generative = bool(dec_path)
        if self._syn_generative:
            Wd = np.load(dec_path)
            nL = int(Wd["n_layers"])
            self._dec_layers = [(jp.asarray(Wd["W%d" % i], jp.float32),
                                 jp.asarray(Wd["b%d" % i], jp.float32)) for i in range(nL)]
            self._dec_zscale = jp.asarray(Wd["z_scale"], jp.float32)
            self._syn_k = int(Wd["z_scale"].shape[0])
            n_musc = int(np.sum(np.asarray(self._muscle_mask) > 0.5))
            self._syn_nblocks = n_musc // 39
            self._muscle_idx = jp.arange(n_musc, dtype=jp.int32)
            self._base_idx = jp.arange(n_musc, self.mjx_model.nu, dtype=jp.int32)
            print(f"[mjx synergy-generative] decoder={dec_path}, K={self._syn_k}/hand, "
                  f"blocks={self._syn_nblocks}, layers={nL}, "
                  f"action_size {self.mjx_model.nu} -> {self.action_size}")
            return
        K = int(self._config.get("synergy_k", 12))
        span = float(self._config.get("synergy_span", 4.0))
        basis_path = self._config.get("synergy_basis_path", "")
        participant = str(self._config.get("synergy_participant", ""))
        S = np.load(basis_path, allow_pickle=True)

        # SELECTABLE basis (AAAI experiment matrix). Three npz layouts are
        # supported so a run can pick human global-28, JoSE dynamics-derived, or a
        # per-participant basis without touching the env:
        #   * global/JoSE:   `components` (n_syn, 39) [+ optional `mean` (39,)]
        #   * per-participant: one (12, 39) array per participant id (npz key) ->
        #                      choose which with `synergy_participant`.
        if participant and participant in S.files:
            C = S[participant].astype(np.float64)       # (n_syn, 39)
            basis_tag = f"per-participant[{participant}]"
        elif "components" in S.files:
            C = S["components"].astype(np.float64)      # (n_syn, 39)
            basis_tag = "components"
        else:
            raise KeyError(
                f"synergy basis {basis_path} has no 'components' and "
                f"participant='{participant}' not among {list(S.files)[:5]}...")
        assert K <= C.shape[0], f"K={K} > available synergies {C.shape[0]}"
        Ck = C[:K]                                      # (K, 39)

        # scale (and, if the basis has no baked mean, the decode offset) come from
        # the human right-hand per-key template spread (matches CPU). Human-derived
        # bases (global-28) ship a `mean`; dynamics-derived (JoSE) and per-
        # participant bases do not -> use the human template's valid-key mean as a
        # sensible muscle operating point so z=0 sits at a human posture.
        tpath = self._config.get("synergy_scale_template", "") \
            or self._config.get("muscle_template_path", "")
        T = np.load(tpath, allow_pickle=True)
        valid = T["count"].astype(int) >= int(self._config.get("template_min_count", 50))
        if "mean" in S.files:
            mean = S["mean"].astype(np.float64)         # (39,)
            mean_tag = "basis"
        else:
            mean = T["act"].astype(np.float64)[valid].mean(axis=0)   # (39,)
            mean_tag = "template"
        scores = (T["act"].astype(np.float64)[valid] - mean) @ Ck.T   # (n_valid, K)
        scale = span * scores.std(axis=0)               # (K,)

        n_musc = int(np.sum(np.asarray(self._muscle_mask) > 0.5))
        assert n_musc % 39 == 0, f"muscle count {n_musc} not a multiple of 39"
        self._syn_k = K
        self._syn_nblocks = n_musc // 39                # 1 (uni) or 2 (bimanual)
        self._syn_mean = jp.asarray(mean, dtype=jp.float32)
        self._syn_Ck = jp.asarray(Ck, dtype=jp.float32)
        self._syn_scale = jp.asarray(scale, dtype=jp.float32)
        self._muscle_idx = jp.arange(n_musc, dtype=jp.int32)
        self._base_idx = jp.arange(n_musc, self.mjx_model.nu, dtype=jp.int32)
        print(f"[mjx synergy] basis={basis_path} ({basis_tag}, mean<-{mean_tag}); "
              f"K={K}/hand, blocks={self._syn_nblocks}, span={span}, "
              f"action_size {self.mjx_model.nu} -> {self.action_size}; "
              f"scale={np.round(scale, 3)}")

    def _decode_synergy(self, a):
        """[z (K per hand), base] -> native (nu,) action in [-1,1]."""
        nb, K = self._syn_nblocks, self._syn_k
        z = a[:nb * K].reshape(nb, K)                   # (nb, K)
        base = a[nb * K:]
        if getattr(self, "_syn_generative", False):
            x = self._dec_zscale * (3.0 * jp.tanh(z / 3.0))   # (nb, K)
            nL = len(self._dec_layers)
            for li, (W, b) in enumerate(self._dec_layers):
                x = x @ W + b
                x = 1.0 / (1.0 + jp.exp(-x)) if li == nL - 1 else jp.maximum(x, 0.0)
            act = x                                     # (nb, 39) in [0,1]
        else:
            latent = z * self._syn_scale                # (nb, K)
            act = latent @ self._syn_Ck + self._syn_mean  # (nb, 39) = mean + Ck.T@latent
        act = jp.clip(act, 1e-3, 1.0 - 1e-3)
        a_musc = jp.clip(0.5 + jp.log(act / (1.0 - act)) / 5.0, -1.0, 1.0).reshape(-1)
        full = jp.zeros((self.mjx_model.nu,))
        full = full.at[self._muscle_idx].set(a_musc)
        full = full.at[self._base_idx].set(base)
        return full

    @property
    def action_size(self) -> int:
        if getattr(self, "_syn_on", False):
            n_base = int(self.mjx_model.nu) - int(np.sum(np.asarray(self._muscle_mask) > 0.5))
            return self._syn_k * self._syn_nblocks + n_base
        return int(self.mjx_model.nu)

    def preprocess_spec(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        # base handles cylinder/margin (jax) + option (iters, timestep). Contact
        # filtering + posture are applied post-compile on the model arrays.
        spec = super().preprocess_spec(spec)
        # Drop touch sensors: MJX/Warp reject `touch` sensors, and the task reads
        # key depression from qpos (not the sensors). Lets the full & bimanual
        # scenes (which include the sensored board) load unmodified.
        for sensor in list(spec.sensors):
            spec.delete(sensor)
        if self.impl == "jax":
            # The MJX-JAX collider rejects any non-zero margin/gap on several geom
            # pairs (e.g. capsule x mesh among the finger segments). Zero them on
            # every geom so put_model succeeds. (Warp's GJK/EPA needs none of this.)
            for geom in spec.geoms:
                geom.margin = 0
                geom.gap = 0
        return spec

    def _get_data(self, qpos, qvel):
        naconmax = int(self._config.naconmax_per_env) * self._config.num_envs
        return make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros((self.mjx_model.nu,)),
            impl=self._config.impl,
            naconmax=naconmax,
            njmax=self._mj_model.njmax if self._mj_model.njmax != -1 else 1_000,
            naccdmax=naconmax,
        )

    # ---------------------------------------------------------------- helpers
    def _current_pos(self, info):
        """Position (into target_ids) of the currently-active key in the sequence.
        In single-key mode seq_pos is always 0, so this is just the one target."""
        return info["sequence"][info["seq_pos"]]

    def _target_key(self, info):
        """Model key index of the current per-env target."""
        return self._target_ids[self._current_pos(info)]

    def _key_travel_frac(self, data, key):
        return jp.clip(data.qpos[self._key_qposadr[key]] / self._key_travel[key], 0.0, 1.5)

    def _key_geometry(self, data, info):
        """(reach_err, reach_dist, travel_frac, assigned_finger) for the target."""
        key = self._target_key(info)
        fa = self._key_finger[key]
        target_pos = data.site_xpos[self._key_site_ids[key]]
        tip_pos = data.site_xpos[self._tip_sids[fa]]
        reach_err = target_pos - tip_pos
        reach_dist = jp.linalg.norm(reach_err)
        return reach_err, reach_dist, self._key_travel_frac(data, key), fa

    def _seq_advance(self, data, info):
        """Debounced keystroke detection for the CURRENT sequence key: it counts
        as a completed keystroke when depressed past `press_th` by its assigned
        finger AFTER having been seen released (travel < release_th). Mirrors
        KeyboardEnvV0._advance_sequence. Returns (advance_bool, released_seen_now)."""
        _, reach_dist, travel, _ = self._key_geometry(data, info)
        on_key = reach_dist < 2.0 * self._reach_th
        released_now = jp.logical_or(info["released_seen"] > 0.5, travel < self._release_th)
        advance = jp.logical_and(
            jp.logical_and(released_now, travel > self._press_th), on_key
        )
        return advance, released_now

    # ------------------------------------------------------------------ reset
    def _sample_sequence(self, rng):
        """A sequence of `seq_len` target positions (indices into target_ids)."""
        return jax.random.randint(
            rng, (self._seq_len,), 0, self._n_targets, dtype=jp.int32
        )

    def _randomized_init_qpos(self, rng):
        qpos = self._init_qpos
        noise = self._config.base_init_noise
        if noise <= 0 or self._base_qpos_ids.shape[0] == 0:
            return qpos
        half = 0.5 * (self._base_qpos_hi - self._base_qpos_lo)
        d = jax.random.uniform(
            rng, (self._base_qpos_ids.shape[0],), minval=-1.0, maxval=1.0
        ) * noise * half
        vals = jp.clip(qpos[self._base_qpos_ids] + d, self._base_qpos_lo, self._base_qpos_hi)
        return qpos.at[self._base_qpos_ids].set(vals)

    def reset(self, rng: jp.ndarray) -> State:
        rng, rng_t, rng_q = jax.random.split(rng, 3)
        sequence = self._sample_sequence(rng_t)
        qpos = self._randomized_init_qpos(rng_q)
        qvel = jp.zeros(self.mjx_model.nv)

        info = {"rng": rng, "sequence": sequence,
                "seq_pos": jp.array(0, dtype=jp.int32),
                "released_seen": jp.array(1.0),
                "step_count": jp.array(0, dtype=jp.int32)}
        data = self._get_data(qpos, qvel)
        obs = self._get_obs(data, info)
        zero = jp.zeros(())
        metrics = {"reach_reward": zero, "press_reward": zero, "bonus_reward": zero,
                   "penalty_reward": zero, "press_success": zero, "solved_frac": zero,
                   "keystrokes": zero, "words_completed": zero, "muscle_match": zero}
        return State(data, obs, zero, zero, metrics, info)

    # -------------------------------------------------------------------- obs
    def _get_obs(self, data: mjx.Data, info: Dict) -> Dict:
        reach_err, reach_dist, travel, fa = self._key_geometry(data, info)
        tips = data.site_xpos[self._tip_sids]                 # (n_tips, 3)
        finger_disp = (tips - self._finger_home).ravel()
        target_onehot = jax.nn.one_hot(self._current_pos(info), self._n_targets)
        finger_onehot = jax.nn.one_hot(fa, self._n_tips)
        parts = [
            data.qpos,
            data.qvel * self.mjx_model.opt.timestep,
            data.act,
            tips.ravel(),
            finger_disp,
            reach_err,
            target_onehot,
            finger_onehot,
            jp.array([travel]),
        ]
        # P5: optionally expose the release-gate latch (the one bit of hidden
        # sequence state) directly. Off by default so the GPU obs stays a pure
        # single-frame; on -> the debounced release latch is observable, the
        # exact-latch alternative to frame-stacking a history window.
        if self._config.get("obs_released_seen", False):
            parts.append(jp.array([info["released_seen"]]))
        obs = jp.concatenate(parts)
        return {"state": obs}

    # ----------------------------------------------------------------- reward
    def _other_finger_disp(self, data, fa):
        tips = data.site_xpos[self._tip_sids]
        disp = jp.linalg.norm(tips - self._finger_home, axis=-1)
        n = disp.shape[0]
        mask = 1.0 - jax.nn.one_hot(fa, n)
        return jp.sum(disp * mask) / (n - 1)

    def _get_rewards(self, data: mjx.Data, info: Dict) -> Dict:
        rc = self._config.reward_config
        reach_err, reach_dist, travel, fa = self._key_geometry(data, info)
        on_key = reach_dist < 2.0 * self._reach_th
        pressed = jp.logical_and(travel > self._press_th, on_key)
        far_th = jp.where(
            data.time > 2.0 * self.mjx_model.opt.timestep, self._far_th, jp.inf
        )
        act_mag = jp.linalg.norm(data.act) / self._na if self._na > 0 else 0.0
        other_disp = self._other_finger_disp(data, fa)

        rewards = {
            "reach": -reach_dist * rc.reach_weight,
            "press": jp.clip(travel, 0.0, 1.0) * on_key.astype(jp.float32) * rc.press_weight,
            "bonus": pressed.astype(jp.float32) * rc.bonus_weight,
            "home": -other_disp * rc.home_weight,
            "act_reg": -act_mag * rc.act_reg_weight,
            "penalty": -1.0 * (reach_dist > far_th).astype(jp.float32) * rc.penalty_weight,
        }
        if self._is_seq:
            # sparse bonus each time a keystroke in the word is completed.
            advance, _ = self._seq_advance(data, info)
            rewards["keystroke"] = advance.astype(jp.float32) * rc.keystroke_weight
        if self._muscle_match_weight > 0 and self._template_nm > 0:
            rewards["muscle_match"] = (
                self._muscle_match(data, info) * self._muscle_match_weight
            )
        return rewards

    def _muscle_match(self, data, info):
        """exp(-20 * mean((act - human_template[target_key])^2)) over the muscles
        of the hand that presses the target key (per-key validity mask), gated to
        keys backed by enough human presses (0 otherwise). Mirrors
        KeyboardEnvV0.get_reward_dict's `muscle_match`; for the bimanual scene the
        idle hand's muscles are excluded via `_template_mvalid`."""
        key = self._target_key(info)
        mv = self._template_mvalid[key]                     # (na,)
        diff = (data.act[:self._na] - self._template_act[key]) * mv
        mse = jp.sum(diff ** 2) / jp.maximum(jp.sum(mv), 1.0)
        return jp.exp(-20.0 * mse) * self._template_mask[key]

    def _get_done(self, state: State) -> float:
        _, reach_dist, _, _ = self._key_geometry(state.data, state.info)
        far_th = jp.where(
            state.data.time > 2.0 * self.mjx_model.opt.timestep, self._far_th, jp.inf
        )
        done = reach_dist > far_th
        if self._is_seq:
            advance, _ = self._seq_advance(state.data, state.info)
            word_done = (state.info["seq_pos"] + advance.astype(jp.int32)) >= self._seq_len
            done = jp.logical_or(done, word_done)
        return 1.0 * done

    def _get_metrics(self, state: State) -> Dict:
        rewards = self._get_rewards(state.data, state.info)
        _, reach_dist, travel, _ = self._key_geometry(state.data, state.info)
        on_key = reach_dist < 2.0 * self._reach_th
        pressed = 1.0 * jp.logical_and(travel > self._press_th, on_key)
        zero = jp.zeros(())
        keystrokes, words = zero, zero
        if self._is_seq:
            advance, _ = self._seq_advance(state.data, state.info)
            word_done = (state.info["seq_pos"] + advance.astype(jp.int32)) >= self._seq_len
            keystrokes = advance.astype(jp.float32)
            words = word_done.astype(jp.float32)
        muscle_match = zero
        if self._template_nm > 0:
            # raw (unweighted) human-similarity in [0,1] for the current target key
            # -- measured whenever a template is loaded, even at reward weight 0.
            muscle_match = self._muscle_match(state.data, state.info)
        return {
            "reach_reward": rewards["reach"],
            "press_reward": rewards["press"],
            "bonus_reward": rewards["bonus"],
            "penalty_reward": rewards["penalty"],
            "press_success": pressed,
            "solved_frac": pressed / self._config.max_episode_steps,
            "keystrokes": keystrokes,
            "words_completed": words,
            "muscle_match": muscle_match,
        }

    def _get_info(self, state: State) -> Dict:
        info = state.info
        done = state.done
        truncation = jp.where(
            info["step_count"] >= self._config.max_episode_steps,
            1.0 - done, jp.array(0.0),
        )
        rng, rng_t = jax.random.split(info["rng"])

        if self._is_seq:
            advance, released_now = self._seq_advance(state.data, info)
            seq_pos_after = info["seq_pos"] + advance.astype(jp.int32)
            word_done = seq_pos_after >= self._seq_len
            # released_seen for the next step: after advancing it is set from the
            # NEW key (fresh, un-pressed -> released), else the debounced value.
            new_pos = jp.minimum(seq_pos_after, self._seq_len - 1)
            new_key = self._target_ids[info["sequence"][new_pos]]
            new_travel = self._key_travel_frac(state.data, new_key)
            released_after = jp.where(
                advance, (new_travel < self._release_th).astype(jp.float32), released_now.astype(jp.float32)
            )
            reset_ep = jp.logical_or(jp.logical_or(done > 0.5, truncation > 0.5), word_done)
            sequence = jp.where(reset_ep, self._sample_sequence(rng_t), info["sequence"])
            seq_pos = jp.where(reset_ep, jp.array(0, dtype=jp.int32), seq_pos_after)
            released_seen = jp.where(reset_ep, jp.array(1.0), released_after)
        else:
            reset_ep = jp.logical_or(done > 0.5, truncation > 0.5)
            sequence = jp.where(reset_ep, self._sample_sequence(rng_t), info["sequence"])
            seq_pos = jp.array(0, dtype=jp.int32)
            released_seen = jp.array(1.0)

        step_count = jp.where(reset_ep, jp.array(0, dtype=jp.int32), info["step_count"])
        return {**info, "rng": rng, "step_count": step_count, "sequence": sequence,
                "seq_pos": seq_pos, "released_seen": released_seen}

    # --------------------------------------------------------------- dynamics
    def _step_simulation(self, state, action):
        """Muscle actuators get the MyoSuite sigmoid; base position servos get a
        linear [-1,1]->ctrlrange map (a sigmoid would pin them to their max)."""
        action = jp.clip(action, -1.0, 1.0)
        if getattr(self, "_syn_on", False):
            # decode the human-synergy latent -> native (nu,) action, then the
            # standard muscle-sigmoid / base-linear split recovers `act` exactly.
            action = self._decode_synergy(action)
        muscle_ctrl = self.__class__.norm_actions(action)
        base_ctrl = self._base_lo + (action + 1.0) * 0.5 * (self._base_hi - self._base_lo)
        ctrl = muscle_ctrl * self._muscle_mask + base_ctrl * self._base_mask
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self._n_substeps)
        return state.replace(
            data=data,
            info={**state.info, "step_count": state.info["step_count"] + 1},
        )
