# 🚀 VS Code Remote Development - 빠른 시작 가이드

이 가이드는 VS Code Remote Development를 사용하여 로컬에서 개발하고 GPU 서버에서 실행하는 환경을 빠르게 설정하는 방법을 설명합니다.

## ⚡ 1분 만에 시작하기

### 1단계: 자동 설정 실행
```bash
python scripts/setup_vscode_remote.py
```

### 2단계: SSH 설정 (한 번만)
`~/.ssh/config` 파일에 GPU 서버 정보 추가:
```bash
Host gpu-server
    HostName your-gpu-server.com
    User your_username
    IdentityFile ~/.ssh/id_rsa
    ForwardAgent yes
```

### 3단계: 연결 및 실행
```bash
# VS Code로 원격 연결
code --remote ssh-remote+gpu-server ~/rag_api_demo

# 또는 스크립트 사용
./scripts/connect_remote.sh gpu-server ~/rag_api_demo
```

## 📖 상세 설정 가이드

### 방법 A: Remote-SSH (추천 🌟)

#### 장점
- 설정이 간단
- 실시간 디버깅 가능  
- 모든 VS Code 기능 사용 가능

#### 설정 순서
1. **VS Code 확장 설치**
   - Remote - SSH
   - Python
   - Jupyter

2. **SSH 키 설정** (비밀번호 없이 연결)
   ```bash
   # SSH 키 생성
   ssh-keygen -t rsa -b 4096 -C "your.email@example.com"
   
   # 공개 키를 서버에 복사
   ssh-copy-id your_username@gpu-server.com
   ```

3. **SSH Config 설정** (`~/.ssh/config`)
   ```bash
   Host gpu-server
       HostName gpu-server.example.com
       User your_username
       Port 22
       IdentityFile ~/.ssh/id_rsa
       ForwardAgent yes
       ServerAliveInterval 60
   ```

4. **VS Code에서 연결**
   - `Ctrl+Shift+P` → "Remote-SSH: Connect to Host"
   - `gpu-server` 선택
   - 원격 서버에서 프로젝트 폴더 열기

#### 사용법
```bash
# 명령어로 직접 연결
code --remote ssh-remote+gpu-server ~/rag_api_demo

# Python 스크립트 실행 (원격 서버에서)
python simul_02/rag_chain_test.py

# 디버깅: F5 키 또는 Debug 메뉴 사용
```

### 방법 B: Remote-Containers (Docker 기반)

#### 장점
- 환경 일관성 보장
- GPU 드라이버 호환성 문제 해결
- 팀 협업에 이상적

#### 설정 순서
1. **Docker 설치 확인**
   ```bash
   docker --version
   nvidia-docker --version  # GPU 서버에서
   ```

2. **VS Code 확장 설치**
   - Remote - Containers
   - Docker

3. **컨테이너에서 개발**
   - `Ctrl+Shift+P` → "Remote-Containers: Reopen in Container"
   - 자동으로 GPU 지원 컨테이너 빌드 및 연결

#### 사용법
```bash
# 로컬에서 컨테이너 개발 (GPU 없이)
code .
# Ctrl+Shift+P → "Remote-Containers: Reopen in Container"

# 원격 서버에서 컨테이너 개발 (GPU 포함)
code --remote ssh-remote+gpu-server ~/rag_api_demo
# Ctrl+Shift+P → "Remote-Containers: Reopen in Container"
```

## 🔧 개발 워크플로우

### 일반적인 작업 순서
1. **VS Code로 원격 연결**
2. **Python 인터프리터 선택**: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. **터미널에서 의존성 설치**: `pip install -r requirements_gpu.txt`
4. **코드 편집 및 실행**
5. **디버깅**: F5 또는 breakpoint 설정

### 빠른 실행 버튼 (Tasks)
- `Ctrl+Shift+P` → "Tasks: Run Task"
  - "Run RAG Chain Test"
  - "Run Complete Test" 
  - "Check GPU Status"
  - "Start Jupyter Lab"

### 디버깅 설정
프로젝트에 이미 설정된 디버그 구성:
- Debug RAG Chain Test
- Debug Complete Test
- Debug GPU Optimized RAG
- Debug Korean Optimized RAG

## 🐛 트러블슈팅

### SSH 연결 문제
```bash
# 연결 테스트
ssh -v gpu-server

# SSH 에이전트 시작
ssh-add ~/.ssh/id_rsa

# VS Code 로그 확인
# Ctrl+Shift+P → "Remote-SSH: Show Log"
```

### GPU 인식 문제
```bash
# 원격 서버에서 GPU 확인
nvidia-smi

# Docker GPU 테스트
docker run --rm --gpus all nvidia/cuda:12.4-base nvidia-smi
```

### Python 패키지 문제
```bash
# 가상환경 재생성
python -m venv venv
source venv/bin/activate
pip install -r requirements_gpu.txt
```

### 포트 충돌
```bash
# 사용 중인 포트 확인
netstat -tulpn | grep :8000

# 다른 포트 사용
python -m http.server 8001
```

## 🎯 성능 최적화 팁

### 1. SSH 연결 최적화
```bash
# ~/.ssh/config에 추가
Host gpu-server
    ControlMaster auto
    ControlPath ~/.ssh/control-%r@%h:%p
    ControlPersist 10m
```

### 2. VS Code 설정 최적화
```json
{
    "remote.SSH.connectTimeout": 30,
    "remote.SSH.enableDynamicForwarding": false,
    "files.watcherExclude": {
        "**/venv/**": true,
        "**/__pycache__/**": true
    }
}
```

### 3. GPU 메모리 관리
```python
# 코드에서 GPU 메모리 정리
import torch
torch.cuda.empty_cache()
```

## 📝 실제 사용 예시

### 시나리오 1: RAG 모델 실험
```bash
# 1. VS Code로 원격 연결
code --remote ssh-remote+gpu-server ~/rag_api_demo

# 2. 실험 스크립트 실행 (원격 서버에서)
python simul_02/rag_chain_test.py

# 3. 결과 확인
cat result/experiment_results.json
```

### 시나리오 2: Jupyter 노트북 개발
```bash
# 1. 원격 서버에서 Jupyter Lab 시작
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# 2. 로컬에서 접근 (포트 포워딩 자동)
# http://localhost:8888
```

### 시나리오 3: 실시간 디버깅
1. VS Code에서 breakpoint 설정
2. F5로 디버그 모드 실행
3. 원격 서버에서 실행되는 코드를 로컬에서 디버깅

## 🤝 팀 협업

### Git 워크플로우
```bash
# 로컬에서 개발
git checkout -b feature/new-experiment

# 원격 서버에서 테스트
git push origin feature/new-experiment

# 결과 공유
git add result/
git commit -m "Add experiment results"
git push
```

### 환경 공유
- `.devcontainer/devcontainer.json`: 개발 환경 설정
- `requirements_gpu.txt`: GPU 의존성
- `.vscode/`: VS Code 설정 공유

이제 VS Code Remote Development를 사용하여 로컬에서 편리하게 개발하고 GPU 서버에서 실행할 수 있습니다! 🎉