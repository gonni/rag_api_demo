# VS Code Remote-Containers 설정 가이드

## 1. 필요한 소프트웨어 설치

### 로컬 머신
- **Docker Desktop** (Windows/Mac) 또는 **Docker Engine** (Linux)
- **VS Code**
- **Remote - Containers** 확장 (ms-vscode-remote.remote-containers)

### GPU 서버 (원격 실행 시)
- **Docker**
- **nvidia-docker2** (GPU 지원용)

```bash
# Ubuntu에서 nvidia-docker2 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 2. Dev Container 설정 파일 생성

프로젝트 루트에 `.devcontainer` 폴더를 생성하고 설정 파일들을 추가합니다.

### 2.1 devcontainer.json
```json
{
    "name": "RAG API Demo - GPU",
    "dockerFile": "../Dockerfile.gpu",
    "context": "..",
    
    // GPU 지원 설정
    "runArgs": [
        "--gpus", "all",
        "--shm-size=2g"
    ],
    
    // 포트 포워딩
    "forwardPorts": [8000, 11434, 8888],
    "portsAttributes": {
        "8000": {
            "label": "FastAPI Server",
            "onAutoForward": "notify"
        },
        "11434": {
            "label": "Ollama Server",
            "onAutoForward": "silent"
        },
        "8888": {
            "label": "Jupyter Lab",
            "onAutoForward": "openBrowser"
        }
    },
    
    // 볼륨 마운트
    "mounts": [
        "source=${localWorkspaceFolder}/data,target=/app/data,type=bind",
        "source=${localWorkspaceFolder}/result,target=/app/result,type=bind"
    ],
    
    // 환경 변수
    "containerEnv": {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": "/app"
    },
    
    // VS Code 설정
    "settings": {
        "python.defaultInterpreterPath": "/usr/bin/python",
        "python.linting.enabled": true,
        "python.linting.pylintEnabled": true,
        "python.formatting.provider": "black",
        "files.watcherExclude": {
            "**/venv/**": true
        }
    },
    
    // 확장 프로그램
    "extensions": [
        "ms-python.python",
        "ms-toolsai.jupyter",
        "ms-python.black-formatter",
        "ms-python.pylint",
        "ms-vscode.vscode-json"
    ],
    
    // 컨테이너 생성 후 실행할 명령
    "postCreateCommand": "pip install -r requirements_gpu.txt",
    
    // 사용자 설정
    "remoteUser": "root",
    "workspaceFolder": "/app"
}
```

### 2.2 docker-compose.yml (대안)
Docker Compose를 사용하는 경우:
```json
{
    "name": "RAG API Demo - Compose",
    "dockerComposeFile": "../docker-compose.gpu.yml",
    "service": "rag-gpu",
    "workspaceFolder": "/app",
    
    "settings": {
        "python.defaultInterpreterPath": "/usr/bin/python"
    },
    
    "extensions": [
        "ms-python.python",
        "ms-toolsai.jupyter"
    ],
    
    "forwardPorts": [8000, 11434],
    "postCreateCommand": "pip install -r requirements_gpu.txt"
}
```

## 3. 개발용 Dockerfile 수정

기존 `Dockerfile.gpu`를 개발 환경에 맞게 개선:

```dockerfile
# Dockerfile.dev
FROM nvidia/cuda:12.4-devel-ubuntu22.04

# 개발 도구 설치
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    git \
    curl \
    wget \
    vim \
    htop \
    nvidia-smi \
    && rm -rf /var/lib/apt/lists/*

# Python 심볼릭 링크
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 설치
COPY requirements_gpu.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements_gpu.txt

# Jupyter Lab 설치 (선택사항)
RUN pip install jupyterlab

# 개발용 추가 패키지
RUN pip install black pylint ipykernel

# 포트 노출
EXPOSE 8000 11434 8888

# 개발 모드로 실행
CMD ["tail", "-f", "/dev/null"]
```

## 4. 사용 방법

### 4.1 로컬에서 개발
1. VS Code에서 프로젝트 폴더 열기
2. `Ctrl+Shift+P` → "Remote-Containers: Reopen in Container"
3. 컨테이너 빌드 및 연결 대기
4. 개발 시작!

### 4.2 원격 서버에서 개발
```bash
# 1. 로컬에서 원격 서버에 연결
code --remote ssh-remote+gpu-server /home/user/rag_api_demo

# 2. 원격 서버에서 컨테이너 실행
# VS Code가 원격 서버에서 컨테이너를 실행
```

### 4.3 명령줄에서 직접 실행
```bash
# 컨테이너 빌드
docker build -f Dockerfile.gpu -t rag-gpu-dev .

# 컨테이너 실행
docker run -it --gpus all \
  -v $(pwd):/app \
  -p 8000:8000 -p 11434:11434 -p 8888:8888 \
  rag-gpu-dev bash
```

## 5. 개발 워크플로우

### 5.1 코드 편집 및 실행
```bash
# 컨테이너 내에서
cd /app

# RAG 테스트 실행
python simul_02/rag_chain_test.py

# Jupyter Lab 시작 (선택사항)
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### 5.2 디버깅 설정
`.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug RAG Chain",
            "type": "python",
            "request": "launch",
            "program": "/app/simul_02/rag_chain_test.py",
            "console": "integratedTerminal",
            "cwd": "/app",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            }
        }
    ]
}
```

## 6. GPU 리소스 관리

### 6.1 GPU 사용량 모니터링
```bash
# 컨테이너 내에서
nvidia-smi

# 또는 실시간 모니터링
watch -n 1 nvidia-smi
```

### 6.2 멀티 GPU 환경
```json
{
    "runArgs": [
        "--gpus", "device=0,1",  // GPU 0, 1만 사용
        "--shm-size=4g"
    ],
    "containerEnv": {
        "CUDA_VISIBLE_DEVICES": "0,1"
    }
}
```

## 7. 성능 최적화

### 7.1 Docker 볼륨 사용
```json
{
    "mounts": [
        "source=rag-data,target=/app/data,type=volume",
        "source=rag-models,target=/app/models,type=volume"
    ]
}
```

### 7.2 이미지 캐싱
```bash
# 베이스 이미지 미리 빌드
docker build --target base -t rag-base .
```

## 트러블슈팅

### GPU 인식 안됨
```bash
# Docker에서 GPU 테스트
docker run --rm --gpus all nvidia/cuda:12.4-base nvidia-smi
```

### 권한 문제
```json
{
    "remoteUser": "vscode",
    "containerUser": "vscode"
}
```

### 포트 충돌
```bash
# 사용 중인 포트 확인
netstat -tulpn | grep :8000
```