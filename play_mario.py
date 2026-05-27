from pathlib import Path

import gym
import gym_super_mario_bros
import gym_super_mario_bros.actions
import numpy as np
import torch
import torch.nn as nn
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py._image_viewer import ImageViewer
from nes_py._rom import ROM
from nes_py.wrappers import JoypadSpace

import wrappers


# Compatibility patches for gym-super-mario-bros / nes-py with newer NumPy.
SuperMarioBrosEnv._left_x_position = property(
    lambda self: (int(self.ram[0x86]) - int(self.ram[0x071c])) % 256
)
SuperMarioBrosEnv._x_position = property(
    lambda self: int(self.ram[0x6d]) * 0x100 + int(self.ram[0x86])
)
ROM.prg_rom_size = property(lambda self: 16 * int(self.header[4]))
ROM.chr_rom_size = property(lambda self: 8 * int(self.header[5]))
ROM.prg_ram_size = property(lambda self: 8 * (int(self.header[8]) or 1))

_original_viewer_open = ImageViewer.open


def _open_viewer_with_close_flag(self):
    _original_viewer_open(self)
    self.closed_by_user = False

    @self._window.event
    def on_close():
        self.closed_by_user = True
        return True


ImageViewer.open = _open_viewer_with_close_flag


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "mario_dqn_latest.pt"
EPISODES = 5
MAX_STEPS = 5000
PLAY_EPSILON = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        output = self.conv(torch.zeros(1, *shape))
        return output.numel()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = x.float().to(device)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def create_env():
    env = gym_super_mario_bros.make(
        "SuperMarioBros-v3",
        apply_api_compatibility=True,
        disable_env_checker=True
    )

    env = JoypadSpace(
        env,
        gym_super_mario_bros.actions.RIGHT_ONLY
    )

    env = wrappers.MaxAndSkipEnv(env)
    env = wrappers.ProcessFrame84(env)
    env = wrappers.ImageToPyTorch(env)
    env = wrappers.BufferWrapper(env, 4)
    env = wrappers.ScaledFloatFrame(env)

    return env


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        return next_state, reward, terminated or truncated, info

    next_state, reward, done, info = result
    return next_state, reward, done, info


def iter_wrapped_envs(env):
    current = env
    seen = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "env", None)


def get_render_viewer(env):
    for wrapped_env in iter_wrapped_envs(env):
        viewer = getattr(wrapped_env, "viewer", None)
        if viewer is not None:
            return viewer

    return None


def render_env(env):
    viewer = get_render_viewer(env)
    if getattr(viewer, "closed_by_user", False):
        return False

    for wrapped_env in iter_wrapped_envs(env):
        if hasattr(wrapped_env, "render_mode"):
            try:
                wrapped_env.render_mode = "human"
            except AttributeError:
                pass

    env.render()
    viewer = get_render_viewer(env)
    return not getattr(viewer, "closed_by_user", False)


def load_policy(model):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_PATH}. Train first with mario_GPU.py."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["policy_net"])
    model.eval()

    print(
        f"Loaded {CHECKPOINT_PATH} "
        f"from episode {checkpoint.get('episode', 'unknown')}"
    )


def select_action(model, state, env):
    if np.random.random() < PLAY_EPSILON:
        return env.action_space.sample()

    with torch.no_grad():
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        q_values = model(state_tensor)
        return torch.argmax(q_values).item()


def main():
    if not CHECKPOINT_PATH.exists():
        print(f"No checkpoint found at {CHECKPOINT_PATH}. Train first with mario_GPU.py.")
        return

    env = create_env()
    model = DQN(env.observation_space.shape, env.action_space.n).to(device)
    load_policy(model)

    try:
        for episode in range(EPISODES):
            state = reset_env(env)
            episode_reward = 0

            for _ in range(MAX_STEPS):
                action = select_action(model, state, env)
                next_state, reward, done, info = step_env(env, action)

                if not render_env(env):
                    print("Mario window closed. Exiting playback.")
                    return

                episode_reward += reward
                state = next_state

                if done:
                    break

            print(f"Episode: {episode} | Reward: {episode_reward:.2f}")

    except KeyboardInterrupt:
        print("Playback interrupted. Closing Mario window.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
