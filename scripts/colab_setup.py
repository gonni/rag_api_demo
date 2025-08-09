#!/usr/bin/env python3
"""
Google Colab에서 실행하기 위한 설정 스크립트
"""

def setup_colab_environment():
    """Colab 환경에서 필요한 패키지 설치 및 설정"""
    
    # Google Drive 마운트 (선택사항)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully")
    except ImportError:
        print("Not running in Colab environment")
    
    # GPU 확인
    import torch
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("GPU not available")
    
    # 프로젝트 파일 다운로드 (GitHub에서)
    import subprocess
    try:
        subprocess.run(["git", "clone", "https://github.com/your-repo/rag_api_demo.git"], 
                      check=True, cwd="/content")
        print("Repository cloned successfully")
    except subprocess.CalledProcessError:
        print("Failed to clone repository")
    
    # 의존성 설치
    subprocess.run(["pip", "install", "-r", "/content/rag_api_demo/requirements_gpu.txt"], 
                  check=True)
    
    return "/content/rag_api_demo"

if __name__ == "__main__":
    project_path = setup_colab_environment()
    print(f"Project ready at: {project_path}")