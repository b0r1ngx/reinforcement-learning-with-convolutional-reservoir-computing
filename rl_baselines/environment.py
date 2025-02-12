import multiprocessing as mp
import numpy as np
import cv2
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from copy import copy
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs
from stable_baselines3.common.vec_env.subproc_vec_env import _worker

cv2.ocl.setUseOpenCL(False)


def worker(
        remote: mp.connection.Connection,
        parent_remote: mp.connection.Connection,
        env_fn_wrapper: CloudpickleWrapper
):
    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                ob, reward, term, trunc, info = env.step(data)
                done = term or trunc
                if done:
                    ob, reset_info = env.reset()

                remote.send((ob, reward, done, info))
            elif cmd == "reset":
                observation, reset_info = env.reset(seed=data)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((
                    env.observation_space,
                    env.action_space,
                    env.spec
                ))
            else:
                raise NotImplementedError(f"{cmd} is not implemented in the worker")
        except EOFError:
            break


class SubprocVecEnv:
    def __init__(self, env_fns, start_method=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        # store info returned by the reset method
        self.reset_infos = [{} for _ in range(self.num_envs)]
        # seeds to be used in the next call to env.reset()
        self._seeds = [None for _ in range(self.num_envs)]

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            # todo: we can try to use _worker from subproc_vec_env.py
            # todo: check how it works?
            # pytype: disable=attribute-error
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            # pytype: enable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space, self.spec = self.remotes[0].recv()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", self._seeds[env_idx]))
        results = [remote.recv() for remote in self.remotes]
        obs, self.reset_infos = zip(*results)
        # Seeds are only used once
        self._reset_seeds()
        return _flatten_obs(obs, self.observation_space)

    def _reset_seeds(self) -> None:
        """Reset the seeds that are going to be used at the next reset."""
        self._seeds = [None for _ in range(self.num_envs)]

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))

        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="human"):
        for pipe in self.remotes:
            pipe.send(("render",))
        imgs = np.stack(
            [pipe.recv() for pipe in self.remotes]
        )
        bigimg = tile_images(imgs)
        if mode == "human":
            import cv2

            cv2.imshow("vecenv", bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return float(np.sign(reward))


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


def make_single_env(env_name, render_mode=None, **kwargs):
    env = gym.make(env_name, render_mode=render_mode)
    env = AddEpisodeStats(env)
    if "NoFrameskip" in env_name:
        env = wrap_deepmind(make_atari(env, env_name), **kwargs)
    return env


def make_atari(env, env_id, max_episode_steps=4500):
    env._max_episode_steps = max_episode_steps * 4
    assert "NoFrameskip" in env.spec.id
    env = StickyActionEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    if "Montezuma" in env_id or "Pitfall" in env_id:
        env = MontezumaInfoWrapper(env, room_address=3 if "Montezuma" in env_id else 1)
    else:
        env = DummyMontezumaInfoWrapper(env)
    env = AddRandomStateToInfo(env)
    return env


def wrap_deepmind(env, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    # env = NormalizeObservation(env)
    return env


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        rl_common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, term, trunc, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, term, trunc, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        done = term or trunc
        self.visited_rooms.add(self.get_current_room())
        if done:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"].update(
                visited_rooms=copy(self.visited_rooms)
            )
            self.visited_rooms.clear()
        return obs, rew, term, trunc, info

    def reset(self):
        return self.env.reset()


class DummyMontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(DummyMontezumaInfoWrapper, self).__init__(env)

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        done = term or trunc
        if done:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"].update(
                pos_count=0,
                visited_rooms=set([0])
            )
        return obs, rew, term, trunc, info

    def reset(self):
        return self.env.reset()


class AddEpisodeStats(gym.Wrapper):
    def __init__(self, env):
        """Adds the random state to the info field on the first step after reset
        """
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, r, term, trunc, info = self.env.step(action)
        d = term or trunc
        self.reward += r
        self.length += 1
        if d:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"]["reward"] = self.reward
            info["episode"]["length"] = self.length
        return ob, r, term, trunc, info

    def reset(self, **kwargs):
        self.reward = 0
        self.length = 0
        return self.env.reset(**kwargs)


class AddRandomStateToInfo(gym.Wrapper):
    def __init__(self, env):
        """Adds the random state to the info field on the first step after reset
        """
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, r, term, trunc, info = self.env.step(action)
        d = term or trunc
        if d:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"]["rng_at_episode_start"] = self.rng_at_episode_start
        return ob, r, term, trunc, info

    def reset(self, **kwargs):
        self.rng_at_episode_start = copy(self.unwrapped.np_random)
        return self.env.reset(**kwargs)


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def reset(self):
        self.last_action = 0
        return self.env.reset()

    def step(self, action):
        if self.unwrapped.np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, term, trunc, info = self.env.step(action)
        return obs, reward, term, trunc, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            done = term or trunc
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def tile_images(array, n_cols=None, max_images=None, div=1):
    if max_images is not None:
        array = array[:max_images]
    if len(array.shape) == 4 and array.shape[3] == 1:
        array = array[:, :, :, 0]
    assert len(array.shape) in [3, 4], "wrong number of dimensions - shape {}".format(
        array.shape
    )
    if len(array.shape) == 4:
        assert array.shape[3] == 3, "wrong number of channels- shape {}".format(
            array.shape
        )
    if n_cols is None:
        n_cols = max(int(np.sqrt(array.shape[0])) // div * div, div)
    n_rows = int(np.ceil(float(array.shape[0]) / n_cols))

    def cell(i, j):
        ind = i * n_cols + j
        return array[ind] if ind < array.shape[0] else np.zeros(array[0].shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)
