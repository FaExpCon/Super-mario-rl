"""마리오 환경 생성 + 전처리 래퍼 모음"""
import gym
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# nes-py 8.2.1의 JoypadSpace.reset이 seed 등 인자를 받지 못하는 버그 패치
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)


class SkipFrame(gym.Wrapper):
    """같은 행동을 skip 프레임 동안 반복하고 보상을 합산 → 학습 속도 대폭 향상"""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def make_mario_env(stage="SuperMarioBros-1-1-v0", render_mode="rgb_array"):
    env = gym_super_mario_bros.make(
        stage, apply_api_compatibility=True, render_mode=render_mode
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)          # 행동 7개로 축소
    env = SkipFrame(env, skip=4)                     # 프레임 스킵
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)  # 흑백
    env = gym.wrappers.ResizeObservation(env, (84, 84))          # 84x84
    return env
