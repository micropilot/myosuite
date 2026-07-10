# MyoHand QWERTY Keyboard Typing Env

MyoSuite environments in which muscle-driven MyoHand(s) type on a **QWERTY
keyboard** whose keys are spring-loaded sliders with touch sensors.

- **Unimanual** — one right MyoHand (39 muscles). Reaches the right/center of
  the board.
- **Bimanual** — right + mirrored **left** MyoHand (78 muscles). Reaches the
  whole board.

Registered ids (each gets sarcopenia/fatigue/reafferentation auto-variants via
`register_env_with_variants`):

| id | hands | muscles |
|---|---|---|
| `myoHandKeyPress-v0` | right | 39 |
| `myoBimanualKeyPress-v0` | right + left | 78 |

```python
import myosuite
from myosuite.utils import gym
env = gym.make("myoHandKeyPress-v0")               # right-hand key subset
env = gym.make("myoHandKeyPress-v0", target_keys=None)   # full keyboard
env = gym.make("myoBimanualKeyPress-v0")           # two hands, full keyboard
```

## Model

- **`myohand_keyboard.xml`** — unimanual scene: right MyoHand on a 3-DoF
  prismatic *floating base* (`base_x/y/z`) so it can travel over the board, plus
  the keyboard and a mocap target marker.
- **`myohand_keyboard_bimanual.xml`** — bimanual scene: right MyoHand +
  mirrored left MyoHand, each on its own floating base (`base_*_R`, `base_*_L`),
  positioned over the right/left halves of the keyboard (home-row split: right
  index rests near `J`, left index near `F`).
- **`qwerty_keyboard.xml`** — auto-generated keyboard fragment (54 keys). **Do
  not edit by hand**; regenerate with:
  ```bash
  python myosuite/envs/myo/assets/hand/make_keyboard.py
  ```
- **`make_keyboard.py`** — parametric keyboard generator. Each key is a `slide`
  joint (spring return, `gravcomp=1` so it rests undepressed) + box geom +
  `touch` sensor. Key label → body-name mapping is in a comment atop the output.
- **`mirror_hand.py`** — generates the LEFT MyoHand by reflecting the right
  MyoHand MJCF across the sagittal plane (MyoSuite ships only a right hand).
  Emits `myohand_body_left.xml`, `myohand_assets_left.xml` (mirrored, `_L`-suffixed
  names, negative-x mesh scale for chirality) and `myohand_body_right.xml`
  (torso-stripped right hand for the bimanual scene). Regenerate with:
  ```bash
  python myosuite/envs/myo/assets/hand/mirror_hand.py
  ```

## Action space

Unimanual `nu = 42`, bimanual `nu = 84`:

- muscle activations (`muscle_act_ids`; 39 uni / 78 bi), sigmoid-mapped
  `[-1,1]→[0,1]` by `BaseV0.step`.
- base position servos (`base_act_ids`; 3 uni / 6 bi), reprojected
  `[-1,1]→ctrlrange` by `KeyboardEnvV0.step`.

## Observations (default keys)

`qpos, qvel, act, fingertip_pos, reach_err, target_onehot, target_travel`
— `reach_err` is the vector from the **nearest fingertip** to the target key;
`target_onehot` encodes which key to press; `target_travel` is how far it is
currently depressed.

## Task & reward (single-key press)

Each episode a target key is selected (`target_keys` subset). Reward =
`reach` (−distance) + `press` (key depression while a fingertip is on it) +
`bonus` (key past `press_th`) − `act_reg` (muscle effort) − `penalty` (fingertip
strays past `far_th`). `solved` when the key is depressed past `press_th`.

Tunable via env kwargs: `target_keys`, `press_th` (0.6), `reach_th` (0.015 m),
`far_th` (0.30 m), `weighted_reward_keys`.

## Scope & limitations

- Unimanual: the **default `target_keys` covers the right/center region**
  (`y u i o p`, `h j k l ;`, `n m , .`, digits `6–0`, space) that a single right
  hand reaches comfortably. Pass `target_keys=None` for the full board.
- Bimanual: both hands cover the **whole** keyboard (default `target_keys=None`).
- A naive open-loop base-only controller (fingers unflexed) presses roughly half
  the reachable keys; fully depressing every key requires coordinated finger
  **muscle** activation — i.e. what a trained policy (PPO) learns.

This is the first step toward mapping the
[emg2qwerty](https://github.com/facebookresearch/emg2qwerty) dataset
(typed text ↔ EMG) onto muscle/tendon control via RL.
