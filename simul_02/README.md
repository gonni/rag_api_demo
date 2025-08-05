# RAG 시스템 임베딩 모델 비교 시뮬레이션

이 프로젝트는 RAG(Retrieval-Augmented Generation) 시스템에서 다양한 임베딩 모델과 LLM 모델의 성능을 비교하여 최적의 조합을 찾는 시뮬레이션입니다.

## 목표

- **임베딩 모델 비교**: 다양한 Ollama 기반 임베딩 모델의 검색 성능을 비교
- **LLM 체인 테스트**: Ollama에서 실행 가능한 다양한 LLM 모델의 답변 생성 성능 테스트
- **최적 조합 찾기**: 검색 성능과 답변 품질이 가장 좋은 임베딩 모델 + LLM 모델 조합 발견

## 테스트 대상 문서

- **문서**: `../data/dev_center_guide_allmd_touched.md`
- **테스트 쿼리**: "PNS의 purchaseState의 값은 무엇이 있나요"
- **목표**: purchaseState 관련 정보가 포함된 문서 청크를 정확히 검색

## 파일 구조

```
simul_02/
├── embedding_model_comparison.py  # 임베딩 모델 비교 테스트
├── rag_chain_test.py             # RAG 체인 테스트
├── run_complete_test.py          # 통합 테스트 실행
├── test_simple_search.py         # 간단한 시스템 테스트
├── requirements.txt              # 필요한 패키지 목록
├── README.md                    # 이 파일
└── results/                     # 테스트 결과 파일들
    ├── embedding_comparison_results.json
    └── rag_chain_results.json
```

## 테스트할 모델들

### 임베딩 모델 (Ollama 기반)
1. **nomic-embed-text**: Nomic의 범용 임베딩 모델
2. **llama2**: Llama 2 기반 임베딩
3. **llama2:13b**: Llama 2 13B 기반 임베딩
4. **codellama**: Code Llama 기반 임베딩
5. **mistral**: Mistral 기반 임베딩
6. **qwen2**: Qwen2 기반 임베딩

### LLM 모델 (Ollama)
1. **llama2**: Meta의 Llama 2 모델
2. **llama2:13b**: Llama 2 13B 파라미터 모델
3. **codellama**: 코드 생성에 특화된 Llama 모델
4. **mistral**: Mistral AI의 7B 모델
5. **qwen2**: Alibaba의 Qwen2 모델

## 설치 및 실행

### 1. 환경 설정

```bash
# 가상환경 활성화
source ../venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. Ollama 설치 및 모델 다운로드

```bash
# Ollama 설치 (https://ollama.ai/)
# macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# 임베딩 모델 다운로드
ollama pull nomic-embed-text
ollama pull llama2
ollama pull llama2:13b
ollama pull codellama
ollama pull mistral
ollama pull qwen2
```

### 3. 테스트 실행

#### 시스템 테스트 (먼저 실행 권장)
```bash
python test_simple_search.py
```

#### 전체 테스트 실행 (권장)
```bash
python run_complete_test.py
```

#### 개별 테스트 실행

임베딩 모델 비교만 실행:
```bash
python embedding_model_comparison.py
```

RAG 체인 테스트만 실행:
```bash
python rag_chain_test.py
```

## 평가 지표

### 임베딩 모델 평가
- **관련성 점수**: 타겟 키워드("purchaseState", "COMPLETED", "CANCELED" 등)가 검색 결과에 포함된 비율
- **검색 시간**: 쿼리당 평균 검색 시간
- **키워드 매칭**: 목표 키워드들이 검색된 문서에 포함된 정도

### LLM 모델 평가
- **답변 품질**: 관련 키워드가 답변에 포함된 정도
- **응답 시간**: 질문당 평균 응답 생성 시간
- **답변 길이**: 적절한 길이의 답변 생성 여부

## 결과 해석

### 임베딩 모델 결과
- `embedding_comparison_results.json` 파일에서 각 모델의 성능 비교 확인
- 관련성 점수가 높고 검색 시간이 짧은 모델이 우수

### RAG 체인 결과
- `rag_chain_results.json` 파일에서 LLM 모델별 성능 확인
- 관련성 점수가 높고 응답 시간이 적절한 모델이 우수

## 최적화 팁

1. **문서 분할**: 청크 크기와 오버랩을 조정하여 검색 정확도 향상
2. **프롬프트 엔지니어링**: RAG 프롬프트를 조정하여 답변 품질 개선
3. **모델 선택**: 한국어 기술 문서에 특화된 모델 선택 고려
4. **하드웨어**: GPU 가속을 활용하여 처리 속도 향상

## 문제 해결

### 일반적인 문제들

1. **메모리 부족**: 더 작은 임베딩 모델 사용
2. **느린 처리 속도**: GPU 사용 또는 더 빠른 모델 선택
3. **Ollama 연결 오류**: Ollama 서비스가 실행 중인지 확인
4. **모델 다운로드 실패**: 인터넷 연결 확인 및 재시도

### 로그 확인
- 각 스크립트 실행 시 상세한 로그 출력
- 실패한 모델에 대한 오류 메시지 확인
- 결과 파일에서 성공/실패 모델 구분

## Ollama 기반 시스템의 장점

1. **일관된 환경**: 임베딩과 LLM 모두 Ollama에서 실행
2. **로컬 실행**: 인터넷 연결 없이도 실행 가능
3. **커스터마이징**: 다양한 모델을 쉽게 전환 가능
4. **성능 최적화**: 로컬 하드웨어에 최적화된 실행

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다. 