"""2단계: PPO로 학습"""
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from mario_env import make_mario_env


def build_vec_env():
    env = DummyVecEnv([make_mario_env])
    env = VecMonitor(env)  # 에피소드 보상/길이 기록 → TensorBoard의 rollout/ep_rew_mean
    env = VecFrameStack(env, n_stack=4, channels_order="last")  # 최근 4프레임 겹치기
    return VecTransposeImage(env)  # (H,W,C) → (C,H,W), CNN 입력 형태


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--resume", type=str, default=None, help="이어서 학습할 모델 경로")
    args = parser.parse_args()

    env = build_vec_env()
    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log="./logs/")
    else:
        model = PPO(
            "CnnPolicy", env,
            learning_rate=1e-4, n_steps=512, batch_size=64,
            gamma=0.9, ent_coef=0.01,
            verbose=1, tensorboard_log="./logs/",
        )

    print("학습 장치:", model.device)  # cuda 가 나오면 GPU 사용 중
    checkpoint = CheckpointCallback(save_freq=50_000, save_path="./checkpoints/",
                                    name_prefix="mario")
    model.learn(total_timesteps=args.steps, callback=checkpoint,
                reset_num_timesteps=args.resume is None)
    model.save("mario_ppo_final")
