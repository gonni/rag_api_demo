#!/usr/bin/env python3
"""
원격 서버에서 Python 스크립트를 실행하는 헬퍼 스크립트
"""

import subprocess
import argparse
import os
from pathlib import Path

def sync_and_run(host: str, user: str, remote_path: str, script_path: str, gpu_id: int = 0):
    """로컬 코드를 원격 서버에 동기화하고 실행"""
    
    # 1. rsync로 코드 동기화
    local_project = Path(__file__).parent.parent
    sync_cmd = [
        "rsync", "-avz", "--exclude=venv", "--exclude=__pycache__", 
        "--exclude=.git", "--exclude=result/*", 
        f"{local_project}/", f"{user}@{host}:{remote_path}/"
    ]
    
    print(f"Syncing code to {host}...")
    result = subprocess.run(sync_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Sync failed: {result.stderr}")
        return False
    
    # 2. 원격에서 스크립트 실행
    run_cmd = [
        "ssh", f"{user}@{host}",
        f"cd {remote_path} && CUDA_VISIBLE_DEVICES={gpu_id} python {script_path}"
    ]
    
    print(f"Running {script_path} on {host} (GPU {gpu_id})...")
    result = subprocess.run(run_cmd)
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Run Python script on remote GPU server")
    parser.add_argument("--host", required=True, help="Remote server hostname/IP")
    parser.add_argument("--user", required=True, help="Remote server username")
    parser.add_argument("--remote-path", required=True, help="Remote project path")
    parser.add_argument("--script", required=True, help="Python script to run")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    
    args = parser.parse_args()
    
    success = sync_and_run(args.host, args.user, args.remote_path, args.script, args.gpu)
    if not success:
        exit(1)

if __name__ == "__main__":
    main()