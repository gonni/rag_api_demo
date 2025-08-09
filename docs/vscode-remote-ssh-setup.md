# VS Code Remote-SSH 설정 가이드

## 1. 필요한 확장 설치

VS Code에서 다음 확장들을 설치합니다:

### 필수 확장
- **Remote - SSH** (ms-vscode-remote.remote-ssh)
- **Remote - SSH: Editing Configuration Files** (ms-vscode-remote.remote-ssh-edit)

### 권장 추가 확장
- **Python** (ms-python.python)
- **Jupyter** (ms-toolsai.jupyter)
- **Docker** (ms-azuretools.vscode-docker)

## 2. SSH 키 설정 (권장)

패스워드 없이 연결하기 위해 SSH 키를 설정합니다:

```bash
# 로컬에서 SSH 키 생성 (이미 있다면 생략)
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 공개 키를 원격 서버에 복사
ssh-copy-id your_username@gpu-server.com

# 또는 수동으로 복사
cat ~/.ssh/id_rsa.pub | ssh your_username@gpu-server.com "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

## 3. SSH Config 파일 설정

`~/.ssh/config` 파일을 편집하여 연결 설정을 저장합니다:

```bash
# ~/.ssh/config
Host gpu-server
    HostName gpu-server.example.com
    User your_username
    Port 22
    IdentityFile ~/.ssh/id_rsa
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3

Host gpu-server-local
    HostName 192.168.1.100
    User your_username
    Port 22
    IdentityFile ~/.ssh/id_rsa
    ForwardAgent yes

# AWS/GCP 인스턴스 예시
Host aws-gpu
    HostName ec2-xxx-xxx-xxx-xxx.compute-1.amazonaws.com
    User ubuntu
    IdentityFile ~/.ssh/aws-key.pem
    ForwardAgent yes
```

## 4. VS Code에서 원격 서버 연결

### 4.1 명령 팔레트를 통한 연결
1. `Ctrl+Shift+P` (또는 `Cmd+Shift+P`)
2. "Remote-SSH: Connect to Host..." 선택
3. 설정한 호스트 이름 선택 (예: `gpu-server`)

### 4.2 사이드바를 통한 연결
1. 왼쪽 사이드바에서 Remote Explorer 아이콘 클릭
2. SSH Targets에서 호스트 선택
3. 폴더 아이콘 클릭하여 연결

### 4.3 명령줄을 통한 연결
```bash
code --remote ssh-remote+gpu-server /home/your_username/rag_api_demo
```

## 5. 원격 환경 설정

### 5.1 Python 인터프리터 설정
1. 원격 서버에 연결 후 `Ctrl+Shift+P`
2. "Python: Select Interpreter" 선택
3. 가상환경 또는 시스템 Python 선택

### 5.2 터미널 환경 설정
```bash
# 원격 서버에서 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements_gpu.txt
```

### 5.3 환경 변수 설정
`.vscode/settings.json` 파일 생성:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "terminal.integrated.env.linux": {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": "${workspaceFolder}"
    }
}
```

## 6. 프로젝트별 설정

### 6.1 런치 설정 (.vscode/launch.json)
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "RAG Chain Test",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/simul_02/rag_chain_test.py",
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            },
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Complete Test",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/simul_02/run_complete_test.py",
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            }
        }
    ]
}
```

### 6.2 작업 설정 (.vscode/tasks.json)
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run RAG Test",
            "type": "shell",
            "command": "python",
            "args": ["simul_02/rag_chain_test.py"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "options": {
                "env": {
                    "CUDA_VISIBLE_DEVICES": "0"
                }
            }
        },
        {
            "label": "Install Dependencies",
            "type": "shell",
            "command": "pip",
            "args": ["install", "-r", "requirements_gpu.txt"],
            "group": "build"
        }
    ]
}
```

## 7. 포트 포워딩 설정

웹 서버나 Jupyter 서버를 사용하는 경우:

1. `Ctrl+Shift+P` → "Remote-SSH: Port Forward"
2. 포트 번호 입력 (예: 8000, 8888)
3. 로컬에서 `localhost:8000`으로 접근 가능

## 8. 파일 동기화 및 백업

### 8.1 자동 저장 설정
```json
{
    "files.autoSave": "afterDelay",
    "files.autoSaveDelay": 1000
}
```

### 8.2 Git 설정
```bash
# 원격 서버에서 Git 설정
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# SSH 키를 GitHub에 등록하여 push 가능하게 설정
```

## 트러블슈팅

### 연결 문제
```bash
# SSH 연결 테스트
ssh -v gpu-server

# VS Code 로그 확인
# Command Palette → "Remote-SSH: Show Log"
```

### 성능 최적화
```json
{
    "remote.SSH.connectTimeout": 30,
    "remote.SSH.enableDynamicForwarding": false,
    "remote.SSH.maxReconnectionAttempts": 3
}
```