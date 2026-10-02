"""1단계: 환경이 제대로 동작하는지 랜덤 행동으로 확인"""
from mario_env import make_mario_env

env = make_mario_env(render_mode="human")  # 게임 창이 뜸
obs, info = env.reset()
print("관측값 shape:", obs.shape)  # (84, 84, 1) 이 나와야 정상

for step in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        print(f"에피소드 종료 - 도달 거리 x_pos={info['x_pos']}")
        obs, info = env.reset()
env.close()
