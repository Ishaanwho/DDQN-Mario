import random
from collections import deque
from pathlib import Path

import gym
import gym_super_mario_bros
import gym_super_mario_bros.actions
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py._image_viewer import ImageViewer
from nes_py._rom import ROM
from nes_py.wrappers import JoypadSpace

import wrappers


# gym-super-mario-bros 7.4.0 casts negative x offsets with np.uint8(...).
# Recent NumPy versions raise OverflowError for that cast, so keep the wraparound
# behavior explicit in plain Python integers.
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


# =========================================================
# DEVICE CONFIG
# =========================================================

print("CUDA Available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# HYPERPARAMETERS
# =========================================================

EPISODES = 500
MAX_STEPS = 1000
RENDER_GAME = True
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "mario_dqn_latest.pt"
SAVE_EVERY_EPISODES = 1

BATCH_SIZE = 64
MEMORY_SIZE = 10000

GAMMA = 0.99
LEARNING_RATE = 1e-4

EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99995

TARGET_UPDATE_FREQ = 1000


# =========================================================
# DQN NETWORK
# =========================================================

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
        o = self.conv(torch.zeros(1, *shape))
        return o.numel()

    def forward(self, x):

        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = x.float().to(device)

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        return self.fc(x)


# =========================================================
# ENVIRONMENT
# =========================================================

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


env = create_env()


# =========================================================
# NETWORKS
# =========================================================

policy_net = DQN(
    env.observation_space.shape,
    env.action_space.n
).to(device)

target_net = DQN(
    env.observation_space.shape,
    env.action_space.n
).to(device)

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(
    policy_net.parameters(),
    lr=LEARNING_RATE
)

loss_fn = nn.MSELoss()


# =========================================================
# CHECKPOINTS
# =========================================================

def save_checkpoint(path, episode, epsilon, global_step):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "epsilon": epsilon,
            "global_step": global_step,
            "policy_net": policy_net.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path
    )
    print(f"Saved checkpoint: {path.resolve()}")


def load_checkpoint(path):
    if not path.exists():
        print(f"No checkpoint found. New checkpoints will save to: {path.resolve()}")
        return 0, EPSILON, 0

    checkpoint = torch.load(path, map_location=device)
    policy_net.load_state_dict(checkpoint["policy_net"])
    target_net.load_state_dict(checkpoint["target_net"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    next_episode = int(checkpoint.get("episode", -1)) + 1
    loaded_epsilon = float(checkpoint.get("epsilon", EPSILON))
    loaded_global_step = int(checkpoint.get("global_step", 0))

    print(
        f"Loaded checkpoint: {path.resolve()} | "
        f"resuming at episode {next_episode}"
    )

    return next_episode, loaded_epsilon, loaded_global_step


# =========================================================
# REPLAY MEMORY
# =========================================================

memory = deque(maxlen=MEMORY_SIZE)


# =========================================================
# ACTION SELECTION
# =========================================================

def select_action(state, epsilon):

    if random.random() < epsilon:
        return env.action_space.sample()

    with torch.no_grad():

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        q_values = policy_net(state_tensor)

        return torch.argmax(q_values).item()


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
    if not RENDER_GAME:
        return True

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


# =========================================================
# TRAINING STEP
# =========================================================

def train():

    if len(memory) < BATCH_SIZE:
        return None

    batch = random.sample(memory, BATCH_SIZE)

    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.from_numpy(
    np.stack(states)
    ).float().to(device)

    actions = torch.tensor(
        actions,
        dtype=torch.int64,
        device=device
    )

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=device
    )

    next_states = torch.from_numpy(
    np.stack(next_states)
    ).float().to(device)

    dones = torch.tensor(
        dones,
        dtype=torch.float32,
        device=device
    )

    # Current Q values
    current_q_values = policy_net(states)

    current_q = current_q_values.gather(
        1,
        actions.unsqueeze(1)
    ).squeeze(1)

    # Double DQN Target Calculation
    with torch.no_grad():

        next_actions = policy_net(next_states).argmax(1)

        next_q_values = target_net(next_states)

        next_q = next_q_values.gather(
            1,
            next_actions.unsqueeze(1)
        ).squeeze(1)

        target_q = rewards + GAMMA * next_q * (1 - dones)

    loss = loss_fn(current_q, target_q)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()


# =========================================================
# TRAINING LOOP
# =========================================================

start_episode, epsilon, global_step = load_checkpoint(CHECKPOINT_PATH)
stop_training = False
last_episode = start_episode - 1

try:
    for episode in range(start_episode, EPISODES):
        last_episode = episode
        if stop_training:
            break

        state = reset_env(env)

        episode_reward = 0

        for step in range(MAX_STEPS):

            global_step += 1

            action = select_action(state, epsilon)

            next_state, reward, done, info = step_env(env, action)

            if not render_env(env):
                print("Mario window closed. Stopping training.")
                stop_training = True
                break

            episode_reward += reward

            memory.append(
                (
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )
            )

            state = next_state

            loss = train()

            # Update target network
            if global_step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(
                    policy_net.state_dict()
                )

            # Epsilon decay
            epsilon = max(
                EPSILON_MIN,
                epsilon * EPSILON_DECAY
            )

            if done:
                break

        print(
            f"Episode: {episode} | "
            f"Reward: {episode_reward:.2f} | "
            f"Epsilon: {epsilon:.4f}"
        )

        if (episode + 1) % SAVE_EVERY_EPISODES == 0:
            save_checkpoint(CHECKPOINT_PATH, episode, epsilon, global_step)

except KeyboardInterrupt:
    print("Training interrupted. Closing Mario window.")

finally:
    if last_episode >= 0:
        save_checkpoint(CHECKPOINT_PATH, last_episode, epsilon, global_step)
    env.close()
