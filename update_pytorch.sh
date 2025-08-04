#!/bin/bash

echo "🚀 PyTorch 업데이트 시작"

# 현재 버전 확인
echo "현재 PyTorch 버전:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 기존 PyTorch 제거
echo "기존 PyTorch 제거 중..."
pip uninstall torch torchvision torchaudio -y

# 최신 PyTorch 설치 (CUDA 12.4)
echo "최신 PyTorch 설치 중 (CUDA 12.4)..."
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# 설치 확인
echo "설치 확인:"
python -c "
import torch
print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'Compute Capability: {props.major}.{props.minor}')
    print(f'GPU 메모리: {props.total_memory / 1e9:.1f}GB')
else:
    print('CUDA를 사용할 수 없습니다.')
"

echo "✅ PyTorch 업데이트 완료!" 