# 계층적 맥락 보존 RAG 시스템

이 프로젝트는 문서의 계층적 구조를 보존하여 검색 시 상위 맥락 정보를 포함한 정확한 검색 결과를 제공하는 시스템입니다.

## 🎯 해결하는 문제

### 기존 RAG 시스템의 한계
- **단순 제목만으로는 맥락 파악이 어려운 문제**
  - 예: "개요" 제목만으로는 어떤 내용의 개요인지 알 수 없음
- **상위 섹션 정보가 손실되어 잘못된 정보 제공 위험**
  - 예: purchaseState가 SDK인지 Server API인지 구분 불가
- **범용 용어의 모듈별 차이점 구분 불가**
  - 예: SDK의 purchaseState (SUCCEED, FAILED) vs Server API의 purchaseState (COMPLETED, FAILED, PROCESSING)

### 근본적인 문제 시나리오
```
사용자 질문: "Server to Server API에서 purchaseState는 어떤 값들이 있나요?"

기존 방식 문제:
- "개요" 섹션에서 purchaseState 검색 → SDK의 SUCCEED, FAILED 정보 제공 (잘못된 정보)
- 상위 맥락 정보 없음 → 어떤 모듈의 정보인지 구분 불가

개선된 방식:
- "Server to Server API > API Specification > purchaseState" 전체 경로 제공
- 정확한 COMPLETED, FAILED, PROCESSING 정보 제공
```

## 🚀 주요 기능

### 1. 계층적 문서 파싱
- 마크다운 헤더 구조를 기반으로 계층적 섹션 생성
- 부모-자식 관계 추적 및 전체 경로 구축

### 2. 맥락 정보 보존
- 각 섹션의 상위 맥락 정보 자동 생성
- 검색 시 전체 경로 정보 포함

### 3. 모듈별 검색
- 특정 모듈 내에서만 검색 가능
- 모듈 간 비교 검색 지원

### 4. 앙상블 검색
- BM25 + 벡터 검색 조합
- 맥락 정보에 더 높은 가중치 부여

## 📁 파일 구조

```
code_dev_agent/15/
├── hierarchical_context_rag.py          # 메인 구현 파일
├── hierarchical_context_rag_demo.ipynb  # Jupyter Notebook 데모
└── README.md                           # 이 파일
```

## 🛠️ 사용 방법

### 1. 기본 사용법

```python
from hierarchical_context_rag import HierarchicalContextRAG

# RAG 시스템 초기화
rag = HierarchicalContextRAG()

# 문서 파싱 및 검색기 구축
sections = rag.parse_markdown_hierarchy(markdown_text, "doc_id")
contextual_docs = rag.create_contextual_documents(sections)
search_docs = rag.build_search_documents(contextual_docs)
rag.build_retrievers(search_docs)

# 검색 실행
results = rag.search_with_context("purchaseState는 어떤 값들이 있나요?", k=5)
```

### 2. 모듈별 검색

```python
# SDK 모듈에서만 검색
sdk_results = rag.search_by_module("purchaseState 값", "SDK", k=3)

# Server to Server API 모듈에서만 검색
server_results = rag.search_by_module("purchaseState 값", "Server to Server API", k=3)
```

### 3. 모듈 간 비교

```python
# 여러 모듈에서 동일한 쿼리로 검색하여 비교
comparison = rag.compare_across_modules("purchaseState", ["SDK", "Server to Server API"])
```

## 📊 성능 비교

| 구분 | 기존 방식 | 계층적 맥락 방식 |
|------|-----------|------------------|
| 제목 정보 | 단순 제목만 | 전체 경로 제공 |
| 맥락 파악 | 어려움 | 명확함 |
| 모듈 구분 | 불가능 | 가능 |
| 검색 정확도 | 낮음 | 높음 |
| 환각 위험 | 높음 | 낮음 |

## 🔧 기술적 특징

### 데이터 구조
- **HierarchicalSection**: 계층적 섹션 정보
- **ContextualDocument**: 맥락 정보가 포함된 문서
- **검색 문서**: 제목 + 상위 맥락 + 내용 구조

### 검색 알고리즘
- **BM25**: 정확한 키워드 매칭 (가중치 0.6)
- **벡터 검색**: 의미적 유사성 (가중치 0.4)
- **앙상블**: 두 방식의 장점 결합

### 맥락 정보 활용
- 검색 텍스트에 제목과 상위 맥락 포함
- 메타데이터에 계층 구조 정보 저장
- 모듈별 필터링 지원

## 🧪 테스트 시나리오

### 샘플 문서 구조
```
# SDK
## 개요
## API Specification
### purchaseState (SUCCEED, FAILED)

# Server to Server API
## 개요
## API Specification
### purchaseState (COMPLETED, FAILED, PROCESSING)
```

### 테스트 케이스
1. **일반 검색**: "purchaseState는 어떤 값들이 있나요?"
2. **모듈별 검색**: SDK vs Server to Server API
3. **모듈 간 비교**: 동일한 용어의 차이점 확인

## 🎯 기대 효과

### 1. 검색 정확도 향상
- 맥락 정보로 인한 정확한 섹션 매칭
- 잘못된 정보 제공 위험 감소

### 2. 사용자 경험 개선
- 명확한 섹션 경로 제공
- 모듈별 차이점 명확히 구분

### 3. 환각(Hallucination) 방지
- 상위 맥락 정보로 정확성 검증
- 모듈별 필터링으로 혼동 방지

## 🔄 향후 개선 방향

1. **동적 가중치 조정**: 쿼리 유형에 따른 검색 가중치 최적화
2. **멀티모달 지원**: 이미지, 코드 블록 등 다양한 콘텐츠 타입 지원
3. **실시간 업데이트**: 문서 변경 시 검색기 자동 업데이트
4. **성능 최적화**: 대용량 문서 처리 성능 향상

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!
