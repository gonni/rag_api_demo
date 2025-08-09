#!/usr/bin/env python3
"""
VS Code Remote Development 환경 설정 스크립트
"""

import os
import subprocess
import json
import sys
from pathlib import Path

def check_requirements():
    """필요한 소프트웨어가 설치되었는지 확인"""
    requirements = {
        'code': 'VS Code가 설치되어 있지 않습니다.',
        'docker': 'Docker가 설치되어 있지 않습니다.',
        'ssh': 'SSH 클라이언트가 설치되어 있지 않습니다.'
    }
    
    missing = []
    for cmd, msg in requirements.items():
        try:
            subprocess.run([cmd, '--version'], capture_output=True, check=True)
            print(f"✅ {cmd} 설치됨")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {msg}")
            missing.append(cmd)
    
    return len(missing) == 0

def install_vscode_extensions():
    """필요한 VS Code 확장 설치"""
    extensions = [
        'ms-vscode-remote.remote-ssh',
        'ms-vscode-remote.remote-ssh-edit',
        'ms-vscode-remote.remote-containers',
        'ms-python.python',
        'ms-toolsai.jupyter',
        'ms-azuretools.vscode-docker'
    ]
    
    print("VS Code 확장 설치 중...")
    for ext in extensions:
        try:
            result = subprocess.run(['code', '--install-extension', ext], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {ext} 설치 완료")
            else:
                print(f"⚠️  {ext} 설치 실패: {result.stderr.strip()}")
        except Exception as e:
            print(f"❌ {ext} 설치 중 오류: {e}")

def setup_ssh_config():
    """SSH 설정 도움말 출력"""
    ssh_config_example = """
# ~/.ssh/config 파일에 추가할 설정 예시:

Host gpu-server
    HostName your-gpu-server.com
    User your_username
    Port 22
    IdentityFile ~/.ssh/id_rsa
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3

# SSH 키 생성 (필요한 경우):
# ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 공개 키를 서버에 복사:
# ssh-copy-id your_username@your-gpu-server.com
"""
    
    print("SSH 설정 가이드:")
    print(ssh_config_example)
    
    ssh_config_path = Path.home() / '.ssh' / 'config'
    if ssh_config_path.exists():
        print(f"기존 SSH 설정 파일: {ssh_config_path}")
    else:
        print("~/.ssh/config 파일을 생성하여 위 설정을 추가하세요.")

def test_docker_gpu():
    """Docker GPU 지원 테스트"""
    print("Docker GPU 지원 테스트 중...")
    try:
        result = subprocess.run([
            'docker', 'run', '--rm', '--gpus', 'all', 
            'nvidia/cuda:12.4-base', 'nvidia-smi'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Docker GPU 지원 정상 작동")
            return True
        else:
            print(f"❌ Docker GPU 테스트 실패: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Docker GPU 테스트 시간 초과")
        return False
    except Exception as e:
        print(f"❌ Docker GPU 테스트 중 오류: {e}")
        return False

def create_example_ssh_script():
    """SSH 연결 예시 스크립트 생성"""
    script_content = """#!/bin/bash
# VS Code Remote-SSH 연결 스크립트

# 사용법: ./connect_remote.sh [host_name] [project_path]

HOST=${1:-gpu-server}
PROJECT_PATH=${2:-~/rag_api_demo}

echo "Connecting to $HOST..."
echo "Project path: $PROJECT_PATH"

# VS Code로 원격 연결
code --remote ssh-remote+$HOST $PROJECT_PATH

echo "VS Code Remote 연결이 시작되었습니다."
echo "새 VS Code 창에서 원격 서버에 연결됩니다."
"""
    
    script_path = Path('scripts/connect_remote.sh')
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    print(f"✅ SSH 연결 스크립트 생성: {script_path}")

def main():
    print("🚀 VS Code Remote Development 환경 설정")
    print("=" * 50)
    
    # 1. 필요 소프트웨어 확인
    print("\n1. 필요 소프트웨어 확인...")
    if not check_requirements():
        print("\n필요한 소프트웨어를 먼저 설치해주세요.")
        sys.exit(1)
    
    # 2. VS Code 확장 설치
    print("\n2. VS Code 확장 설치...")
    install_vscode_extensions()
    
    # 3. Docker GPU 테스트 (선택사항)
    print("\n3. Docker GPU 지원 테스트...")
    if input("Docker GPU 테스트를 실행하시겠습니까? (y/N): ").lower() == 'y':
        test_docker_gpu()
    
    # 4. SSH 설정 가이드
    print("\n4. SSH 설정 가이드...")
    setup_ssh_config()
    
    # 5. 헬퍼 스크립트 생성
    print("\n5. 헬퍼 스크립트 생성...")
    create_example_ssh_script()
    
    print("\n🎉 설정 완료!")
    print("\n다음 단계:")
    print("1. ~/.ssh/config 파일을 설정하세요")
    print("2. VS Code에서 Ctrl+Shift+P → 'Remote-SSH: Connect to Host'")
    print("3. 또는 Dev Container 사용: Ctrl+Shift+P → 'Remote-Containers: Reopen in Container'")
    print("4. 또는 스크립트 사용: ./scripts/connect_remote.sh gpu-server ~/rag_api_demo")

if __name__ == "__main__":
    main()