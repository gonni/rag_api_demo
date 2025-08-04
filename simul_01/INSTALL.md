# simul_01 프로젝트 설치 가이드

## 📋 개요

이 가이드는 simul_01 RAG 실험 프로젝트의 환경 설정을 설명합니다.

## 🚀 빠른 설치

### 1. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### 2. 패키지 설치

#### CPU 전용 환경
```bash
pip install -r requirements_simul.txt
```

#### GPU 환경 (권장)
```bash
pip install -r requirements_gpu.txt
```

## 🔧 상세 설치 가이드

### 필수 패키지 확인

```bash
# Python 버전 확인 (3.8 이상 권장)
python --version

# pip 업그레이드
pip install --upgrade pip
```

### GPU 환경 설정 (선택사항)

#### CUDA 확인
```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch GPU 지원 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### GPU 패키지 설치
```bash
# CUDA 12.1용 PyTorch 설치
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

### Ollama 설정

#### Ollama 서버 설치
```bash
# Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# https://ollama.ai/download 에서 다운로드
```

#### 모델 다운로드
```bash
# exaone3.5:latest 모델 다운로드
ollama pull exaone3.5:latest
```

## 🧪 설치 확인

### 기본 테스트
```bash
# Python 패키지 import 테스트
python -c "
import langchain
import langchain_ollama
import faiss
import torch
import pandas
import matplotlib
print('✅ 모든 패키지가 정상적으로 설치되었습니다!')
"
```

### GPU 테스트
```bash
# GPU 사용 가능 여부 확인
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('CPU 모드로 실행됩니다.')
"
```

### Ollama 연결 테스트
```bash
# Ollama 서버 연결 확인
python -c "
from langchain_ollama import OllamaEmbeddings
try:
    embeddings = OllamaEmbeddings(model='exaone3.5:latest')
    print('✅ Ollama 서버에 정상적으로 연결되었습니다!')
except Exception as e:
    print(f'❌ Ollama 연결 실패: {e}')
"
```

## 📁 프로젝트 구조 확인

```bash
# 필수 파일들 확인
ls -la simul_01/
# 예상 파일들:
# - rag_experiment.py
# - gpu_optimized_rag.py
# - document_analyzer.py
# - result_analyzer.py
# - simple_test.py
# - check_tables.py
# - README.md
```

## 🚀 첫 실행

### 1. 기본 테스트
```bash
cd simul_01
python simple_test.py
```

### 2. GPU 최적화 실험
```bash
python gpu_optimized_rag.py
```

### 3. 전체 실험
```bash
python rag_experiment.py
```

## ⚠️ 문제 해결

### 일반적인 문제들

#### 1. CUDA 오류
```bash
# CUDA 버전 확인
nvcc --version

# PyTorch 재설치
pip uninstall torch torchvision
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

#### 2. Ollama 연결 오류
```bash
# Ollama 서버 상태 확인
ollama list

# 서버 재시작
sudo systemctl restart ollama
# 또는
ollama serve
```

#### 3. 메모리 부족
```bash
# GPU 메모리 확인
nvidia-smi

# 배치 크기 조정 (코드에서 수정)
# batch_size = 16  # 더 작은 값으로 조정
```

#### 4. 패키지 충돌
```bash
# 가상환경 재생성
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements_gpu.txt
```

## 📊 성능 최적화

### GPU 메모리 최적화
```bash
# 실시간 GPU 모니터링
watch -n 1 nvidia-smi

# 메모리 정리
python -c "import torch; torch.cuda.empty_cache()"
```

### 배치 크기 조정
- GPU 메모리 8GB 이하: `batch_size = 16`
- GPU 메모리 8GB 이상: `batch_size = 32`
- GPU 메모리 16GB 이상: `batch_size = 64`

## 📞 지원

문제가 발생하면 다음을 확인하세요:

1. **로그 확인**: 각 스크립트의 상세한 로그 출력
2. **환경 확인**: Python 버전, CUDA 버전, 패키지 버전
3. **메모리 확인**: GPU/CPU 메모리 사용량
4. **네트워크 확인**: Ollama 서버 연결 상태

## 🎯 성공 지표

설치가 성공적으로 완료되면 다음이 가능합니다:

- ✅ `python simple_test.py` 실행 성공
- ✅ purchaseState 관련 정보 31개 발견
- ✅ GPU 사용량 모니터링 가능
- ✅ Ollama 모델 `exaone3.5:latest` 사용 가능
- ✅ 실험 결과 JSON 파일 생성
- ✅ 시각화 차트 생성 