"""3단계: 학습된 모델이 플레이하는 모습 보기"""
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from mario_env import make_mario_env

model_path = sys.argv[1] if len(sys.argv) > 1 else "mario_ppo_final"
env = DummyVecEnv([lambda: make_mario_env(render_mode="human")])
env = VecTransposeImage(VecFrameStack(env, n_stack=4, channels_order="last"))
model = PPO.load(model_path)

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    if done[0]:
        print(f"x_pos={info[0]['x_pos']}, 깃발 도달={info[0]['flag_get']}")
