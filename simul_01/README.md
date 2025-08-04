# RAG 문서 분할 전략 실험 도구

이 디렉토리는 다양한 문서 분할 전략을 테스트하고 성능을 비교하는 실험 도구들을 포함합니다.

## 📋 개요

원스토어 인앱 결제 가이드 문서(`dev_center_guide_allmd_touched.md`)를 대상으로, 다양한 문서 분할 전략을 실험하여 RAG 검색 성능을 향상시키는 방법을 연구합니다.

특히 "PNS의 purchaseState에는 어떤 값들이 있나요?"와 같은 쿼리에 대해 관련 테이블 정보를 정확히 검색할 수 있도록 하는 것이 목표입니다.

## 🎯 실험 전략

### 1. 계층적 분할 + 전체 맥락 포함 (strategy_1)
- 대제목(##) 기준으로 메인 섹션 생성
- 각 메인 섹션의 전체 내용을 하나의 문서로 생성
- 소제목(###, ####) 기준으로 세부 분할 추가
- 중복 내용 허용으로 전체 맥락 보존

### 2. 향상된 계층적 분할 (strategy_2)
- 모든 헤더 레벨에 대해 개별 문서 생성
- 제목 계층 구조를 명확히 포함
- 헤더 레벨 정보를 메타데이터에 추가

### 3. 테이블 인식 분할 (strategy_3)
- 테이블이 포함된 섹션을 특별 처리
- 테이블을 하나의 단위로 유지
- 테이블 정보를 메타데이터에 기록

### 4. 키워드 강화 분할 (strategy_4)
- 중요 키워드(PNS, purchaseState, COMPLETED, CANCELED 등) 감지
- 발견된 키워드를 문서 내용에 명시적으로 포함
- 키워드 정보를 메타데이터에 추가

## 📁 파일 구조

```
simul_01/
├── rag_experiment.py          # 메인 실험 코드
├── document_analyzer.py       # 문서 분석 도구
├── result_analyzer.py         # 결과 분석 도구
├── run_experiments.py         # 전체 실험 실행 스크립트
├── README.md                  # 이 파일
└── experiment_results_*.json  # 실험 결과 (자동 생성)
```

## 🚀 사용법

### 1. 전체 실험 실행 (권장)

```bash
python simul_01/run_experiments.py
```

이 스크립트는 다음을 순차적으로 실행합니다:
1. 문서 구조 분석
2. RAG 실험 실행
3. 결과 분석 및 시각화

### 2. 개별 도구 실행

#### 문서 분석
```bash
cd simul_01
python document_analyzer.py
```

#### RAG 실험
```bash
cd simul_01
python rag_experiment.py
```

#### 결과 분석
```bash
cd simul_01
python result_analyzer.py
```

## 📊 실험 결과

실험 후 다음 파일들이 생성됩니다:

### 데이터 파일
- `experiment_results_YYYYMMDD_HHMMSS.json`: 실험 결과 데이터
- `document_analysis_report.json`: 문서 구조 분석 결과

### 시각화 파일
- `header_distribution.png`: 헤더 레벨 분포
- `keyword_frequency.png`: 중요 키워드 빈도
- `document_structure_summary.png`: 문서 구조 요약
- `strategy_comparison.png`: 전략별 성능 비교
- `query_performance_heatmap.png`: 쿼리별 성능 히트맵
- `best_strategy_distribution.png`: 최고 성능 전략 분포

### 보고서
- `detailed_analysis_report.md`: 상세 분석 보고서

## 🔍 테스트 쿼리

실험에서 사용되는 테스트 쿼리들:

1. "PNS의 purchaseState에는 어떤 값들이 있나요?"
2. "purchaseState COMPLETED CANCELED 값"
3. "원스토어 결제 상태 값들"
4. "PNS payment notification service purchaseState"
5. "COMPLETED CANCELED 결제 상태"

## 📈 성능 평가

각 전략의 성능은 다음 기준으로 평가됩니다:

- **관련성 점수**: purchaseState, COMPLETED, CANCELED 키워드 포함 여부
- **문서 수**: 생성된 문서의 개수
- **검색 정확도**: 원하는 테이블 정보가 검색되는지 여부

## 🛠️ 의존성

필요한 Python 패키지들:

```bash
pip install langchain langchain-community langchain-ollama langchain-text-splitters faiss-cpu matplotlib seaborn pandas
```

## 📝 주의사항

1. **Ollama 모델**: `nomic-embed-text` 모델이 필요합니다.
2. **메모리**: 대용량 문서 처리 시 충분한 메모리가 필요합니다.
3. **실행 시간**: 전체 실험은 10-30분 정도 소요될 수 있습니다.

## 🔧 커스터마이징

### 새로운 전략 추가

`rag_experiment.py`의 `DocumentSplitterExperiment` 클래스에 새로운 전략 메서드를 추가하고, `run_experiment` 메서드의 `strategies` 딕셔너리에 등록하면 됩니다.

### 새로운 쿼리 추가

`rag_experiment.py`의 `main` 함수에서 `test_queries` 리스트를 수정하면 됩니다.

### 평가 기준 수정

`test_query` 메서드의 `relevance_score` 계산 로직을 수정하여 다른 평가 기준을 적용할 수 있습니다.

## 📞 문제 해결

### 일반적인 문제들

1. **Ollama 연결 오류**: Ollama 서비스가 실행 중인지 확인
2. **메모리 부족**: 문서 크기를 줄이거나 청크 크기를 조정
3. **의존성 오류**: 필요한 패키지들이 모두 설치되었는지 확인

### 로그 확인

각 스크립트는 상세한 로그를 출력하므로, 오류 발생 시 로그를 확인하여 문제를 파악할 수 있습니다.

## 🎯 기대 효과

이 실험을 통해 다음과 같은 효과를 기대할 수 있습니다:

1. **검색 정확도 향상**: purchaseState 관련 테이블 정보의 정확한 검색
2. **문맥 보존**: 전체 맥락을 유지하면서 세부 정보 검색 가능
3. **전략 비교**: 다양한 분할 전략의 성능 비교 및 최적 전략 선택
4. **시각화**: 실험 결과의 직관적인 이해

## 📚 참고 자료

- [LangChain 문서 분할 가이드](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [FAISS 벡터 검색](https://faiss.ai/)
- [원스토어 인앱 결제 가이드](data/dev_center_guide_allmd_touched.md) 