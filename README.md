# Super Mario Bros 강화학습 AI

PPO 알고리즘으로 슈퍼 마리오 브라더스 1-1 스테이지를 플레이하는 강화학습 에이전트입니다.

## 기술 스택

- Python 3.10
- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) / nes-py — NES 에뮬레이터 기반 게임 환경
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (PPO) + PyTorch

## 구조

| 파일 | 설명 |
|---|---|
| `mario_env.py` | 환경 생성 및 전처리 (행동 축소, 프레임 스킵, 흑백, 84x84 리사이즈) |
| `check_env.py` | 랜덤 행동으로 환경 동작 확인 |
| `train.py` | PPO 학습 (체크포인트 자동 저장, 이어서 학습 지원) |
| `play.py` | 학습된 모델 플레이 시연 |

## 실행 방법

```bash
conda create -n mario python=3.10 -y
conda activate mario
# GPU 사용 시 pytorch.org에서 CUDA 버전에 맞는 torch를 먼저 설치
pip install -r requirements.txt

python check_env.py            # 환경 확인
python train.py                # 학습 (기본 100만 스텝)
python train.py --resume checkpoints/mario_500000_steps.zip   # 이어서 학습
python play.py checkpoints/mario_500000_steps.zip             # 플레이 보기
tensorboard --logdir logs      # 학습 그래프 확인
```

## 전처리

- **행동 공간 축소**: 256가지 버튼 조합 → `SIMPLE_MOVEMENT` 7가지
- **프레임 스킵**: 같은 행동을 4프레임 반복하여 학습 속도 향상
- **흑백 + 84x84 리사이즈**: 입력 크기 축소
- **프레임 스택**: 최근 4프레임을 겹쳐 이동 방향·속도 정보 제공

## 학습 결과

> 학습 진행 후 업데이트 예정 (보상 그래프, 플레이 GIF)
