# Python 스크립트 원격 실행 가이드

이 가이드는 개발환경과 GPU 실행환경을 분리하여 Python 스크립트를 실행하는 방법들을 설명합니다.

## 방법 1: VS Code Remote Development
### 장점
- Jupyter notebook과 가장 유사한 경험
- 실시간 디버깅 가능
- 파일 탐색기, 터미널 등 모든 기능 사용 가능

### 사용법
```bash
# Remote-SSH 확장 설치 후
code --remote ssh-remote+user@gpu-server /path/to/rag_api_demo

# 또는 Docker 컨테이너 사용
code --remote containers+gpu-container /workspace
```

### 단점
- 네트워크 지연에 민감
- VS Code 전용

## 방법 2: Docker 기반 분리
### 장점
- 환경 일관성 보장
- 스케일링 용이
- 로컬/원격 환경 동일

### 사용법
```bash
# GPU 서버에서 실행
docker-compose -f docker-compose.gpu.yml up rag-gpu

# 로컬 개발
docker-compose -f docker-compose.gpu.yml --profile dev up rag-dev
```

### 단점
- Docker 학습 곡선
- 초기 설정 복잡

## 방법 3: SSH 원격 실행
### 장점
- 가장 직관적
- 설정이 간단
- 모든 서버에서 사용 가능

### 사용법
```bash
# 원격 실행 스크립트 사용
python scripts/remote_run.py \
  --host gpu-server.com \
  --user your_username \
  --remote-path /home/user/rag_api_demo \
  --script simul_02/rag_chain_test.py \
  --gpu 0

# 또는 직접 SSH
rsync -avz --exclude=venv . user@gpu-server:/path/to/project/
ssh user@gpu-server "cd /path/to/project && python simul_02/rag_chain_test.py"
```

### 단점
- 매번 동기화 필요
- 실시간 디버깅 어려움

## 방법 4: 클라우드 플랫폼
### 장점
- GPU 자원 즉시 사용 가능
- 인프라 관리 불필요
- 비용 효율적 (사용한 만큼만 지불)

### 사용법
```python
# Google Colab에서
!git clone https://github.com/your-repo/rag_api_demo.git
!cd rag_api_demo && python scripts/colab_setup.py
!cd rag_api_demo && python simul_02/rag_chain_test.py
```

### 단점
- 데이터 업로드/다운로드 필요
- 세션 시간 제한
- 플랫폼 종속성

## 방법 5: IDE 원격 인터프리터
### 장점
- IDE 기능 모두 사용 가능
- 디버깅 편리
- 파일 동기화 자동

### 사용법 (PyCharm Professional)
1. Settings → Python Interpreter → Add
2. SSH Interpreter 선택
3. 원격 서버 정보 입력
4. 평소처럼 개발 및 실행

### 단점
- 유료 IDE 필요 (PyCharm Professional)
- 설정 복잡

## 추천 방법

### 개발 초기 (실험 단계)
- **Google Colab** 또는 **Kaggle**: 빠른 프로토타이핑

### 지속적 개발
- **VS Code Remote-SSH**: 가장 편리한 개발 경험
- **Docker**: 배포 및 협업 고려시

### 프로덕션 배포
- **Docker + Kubernetes**: 스케일링 필요시
- **클라우드 서비스**: 인프라 관리 최소화

## 현재 프로젝트 적용 예시

```bash
# 1. 원격 서버 설정
ssh gpu-server "mkdir -p ~/rag_api_demo"

# 2. 코드 동기화 및 실행
python scripts/remote_run.py \
  --host gpu-server \
  --user your_username \
  --remote-path ~/rag_api_demo \
  --script simul_02/rag_chain_test.py

# 3. 결과 다운로드
rsync -avz gpu-server:~/rag_api_demo/result/ ./result/
```