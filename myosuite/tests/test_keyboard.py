"""=================================================
# Copyright (c) MyoSuite Authors
Tests for the MyoHand QWERTY keyboard typing env.
================================================="""

import unittest

import numpy as np

import myosuite  # noqa: F401  (registers envs)
from myosuite.tests.test_envs import TestEnvs
from myosuite.utils import gym

KEYPRESS_ENVS = [
    "myoHandKeyPress-v0",
    "myoSarcHandKeyPress-v0",
    "myoFatiHandKeyPress-v0",
    "myoReafHandKeyPress-v0",
]

BIMANUAL_ENVS = [
    "myoBimanualKeyPress-v0",
]


class TestKeyboard(TestEnvs):
    def test_keypress_envs(self):
        # Standard MyoSuite conformance: reset/step/pickle round-trip/seeding.
        self.check_envs("KeyPress", KEYPRESS_ENVS)

    def test_bimanual_envs(self):
        self.check_envs("BimanualKeyPress", BIMANUAL_ENVS)

    def test_bimanual_shapes_and_hands(self):
        env = gym.make("myoBimanualKeyPress-v0")
        u = env.unwrapped
        # 78 muscles (39 per hand) + 6 base servos.
        self.assertEqual(env.action_space.shape[0], 84)
        self.assertEqual(int(u.muscle_act_ids.size), 78)
        self.assertEqual(int(u.base_act_ids.size), 6)
        self.assertEqual(int(u.ftip_sids.size), 10)  # five fingertips per hand
        # both hands present in the muscle set
        names = [u.mj_model.actuator(i).name for i in u.muscle_act_ids]
        self.assertTrue(any(n.endswith("_L") for n in names))
        self.assertTrue(any(not n.endswith("_L") for n in names))
        # whole keyboard targetable with two hands
        self.assertEqual(len(u.target_ids), len(u.key_names))

    def test_left_hand_is_mirror_of_right(self):
        # Left fingertips should be the x-reflection of the right fingertips.
        # Use the raw (non-pronated) posture so this checks pure mirror geometry.
        env = gym.make("myoBimanualKeyPress-v0", pronate=0.0)
        u = env.unwrapped
        env.reset(seed=0)
        d = u.mj_data
        for r, l in [("IFtip", "IFtip_L"), ("THtip", "THtip_L")]:
            pr = d.site_xpos[u.mj_model.site(r).id]
            pl = d.site_xpos[u.mj_model.site(l).id]
            # y and z match; x is mirrored about the two mounts -> just check the
            # left tip sits on the opposite side of / left of its right partner.
            np.testing.assert_allclose(pr[1:], pl[1:], atol=0.02)
            self.assertLess(pl[0], pr[0])  # left hand is on the -x (left) side

    def test_action_and_obs_shapes(self):
        env = gym.make("myoHandKeyPress-v0")
        u = env.unwrapped
        # 39 muscles + 3 base position servos.
        self.assertEqual(env.action_space.shape[0], 42)
        self.assertEqual(int((u.muscle_act_ids.size)), 39)
        self.assertEqual(int((u.base_act_ids.size)), 3)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, env.observation_space.shape)

    def test_target_selection_is_targetable(self):
        env = gym.make("myoHandKeyPress-v0")
        u = env.unwrapped
        for seed in range(5):
            env.reset(seed=seed)
            self.assertIn(u.target_key, list(u.target_ids))
            # marker sits on the selected key
            marker = u.mj_data.mocap_pos[u.target_mocap_id]
            key_site = u.mj_data.site_xpos[u.key_site_ids[u.target_key]]
            np.testing.assert_allclose(marker, key_site, atol=1e-6)

    def test_no_instant_done_at_rest(self):
        # Zero action should not immediately fail the episode.
        env = gym.make("myoHandKeyPress-v0")
        env.reset(seed=0)
        dones = 0
        for _ in range(20):
            _, _, term, trunc, _ = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            dones += int(term or trunc)
        self.assertEqual(dones, 0)

    def test_keys_rest_undepressed(self):
        # gravcomp + spring should keep keys at ~0 travel at rest.
        env = gym.make("myoHandKeyPress-v0")
        u = env.unwrapped
        env.reset(seed=0)
        rest = (u.mj_data.qpos[u.key_jnt_qposids] / u.key_travel).max()
        self.assertLess(float(rest), 0.05)

    def test_full_keyboard_option(self):
        # target_keys=None exposes the whole board.
        env = gym.make("myoHandKeyPress-v0", target_keys=None)
        u = env.unwrapped
        self.assertEqual(len(u.target_ids), len(u.key_names))
        self.assertGreaterEqual(len(u.key_names), 50)


if __name__ == "__main__":
    unittest.main()
