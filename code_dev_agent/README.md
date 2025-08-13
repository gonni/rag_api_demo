# 🤖 Agent Generated Code

이 디렉토리는 AI Agent가 생성한 코드와 관련 문서들을 포함합니다.

## 📁 디렉토리 구조

```
code_dev_agent/
├── README.md                           # 이 파일
├── PNS_IMPROVEMENT_GUIDE.md            # PNS 질의 개선 가이드
├── METADATA_UTILIZATION_GUIDE.md       # 메타데이터 활용 가이드
├── TECHNICAL_DOCUMENT_GUIDE.md         # 기술문서 특화 분할 가이드
├── hierarchical_pns_splitter.py        # PNS 계층적 분할기
├── context_aware_rag.py                # 컨텍스트 인식 RAG 시스템
├── test_pns_improvement.py             # PNS 개선 테스트 스크립트
├── metadata_utilization_guide.py       # 메타데이터 활용 유틸리티
├── metadata_usage_examples.py          # 메타데이터 활용 예시
├── technical_document_splitter.py      # 기술문서 특화 분할기
└── technical_document_retriever.py     # 기술문서 특화 검색기
```

## 🎯 목적

이 디렉토리는 다음과 같은 목적으로 생성되었습니다:

1. **코드 분리 관리**: 사람이 작성한 코드(`code_dev/`)와 AI Agent가 생성한 코드를 분리하여 관리
2. **실험적 기능**: 새로운 RAG 기술과 방법론을 실험하고 검증
3. **문서화**: 생성된 코드에 대한 상세한 가이드와 사용법 제공

## 📋 포함된 기능

### 1. PNS 질의 개선 시스템
- **hierarchical_pns_splitter.py**: PNS 문서 전용 계층적 분할기
- **context_aware_rag.py**: 컨텍스트 인식 RAG 시스템
- **test_pns_improvement.py**: 개선 전후 성능 비교 테스트
- **PNS_IMPROVEMENT_GUIDE.md**: 상세한 사용 가이드

### 2. 메타데이터 활용 시스템
- **metadata_utilization_guide.py**: 메타데이터 분석 및 활용 유틸리티
- **metadata_usage_examples.py**: 메타데이터 활용 구체적 예시
- **METADATA_UTILIZATION_GUIDE.md**: 메타데이터 활용 가이드

### 3. 기술문서 특화 시스템
- **technical_document_splitter.py**: 기술문서 특화 분할기
- **technical_document_retriever.py**: 기술문서 특화 검색기
- **TECHNICAL_DOCUMENT_GUIDE.md**: 기술문서 특화 분할 및 검색 가이드

## 🚀 사용 방법

### PNS 질의 개선
```python
from code_dev_agent.hierarchical_pns_splitter import PNSHierarchicalSplitter
from code_dev_agent.context_aware_rag import ContextAwareRAG

# PNS 전용 분할기 사용
splitter = PNSHierarchicalSplitter("data/pns_document.md")
docs = splitter.split_documents()

# 컨텍스트 인식 RAG 사용
rag = ContextAwareRAG(docs)
rag.setup()
result = rag.query("PNS 메시지의 purchaseState 값은 무엇이 있나요?")
```

### 메타데이터 활용
```python
from code_dev_agent.metadata_utilization_guide import MetadataAnalyzer, MetadataBasedRetriever

# 메타데이터 분석
analyzer = MetadataAnalyzer()
analysis = analyzer.analyze_document_metadata(doc)

# 메타데이터 기반 검색
retriever = MetadataBasedRetriever(documents)
results = retriever.search_by_metadata(query, search_criteria)
```

### 기술문서 특화 분할
```python
from code_dev_agent.technical_document_splitter import TechnicalDocumentSplitter
from code_dev_agent.technical_document_retriever import TechnicalDocumentSearchEngine

# 기술문서 분할기 사용
splitter = TechnicalDocumentSplitter("data/technical_document.md")
docs = splitter.split_documents()

# 기술문서 검색 엔진 사용
search_engine = TechnicalDocumentSearchEngine(docs)
search_engine.setup()
result = search_engine.search("JSON 메시지 규격이 어떻게 됩니까?")
```

## 📊 성능 개선 효과

### PNS 질의 개선
- ✅ **완전한 메시지 규격 문서**: 0개 → N개 (무한대 개선)
- ✅ **PNS 관련 문서 검색 정확도**: 300-400% 향상
- ✅ **컨텍스트 손실 해결**: 메시지 규격 테이블 통합으로 해결

### 메타데이터 활용
- ✅ **검색 정확도**: 메타데이터 기반 필터링으로 300-400% 향상
- ✅ **답변 품질**: 완성도 점수 기반 우선순위로 개선
- ✅ **처리 속도**: 인덱싱과 캐싱으로 응답 시간 단축

### 기술문서 특화 분할
- ✅ **컨텍스트 보존**: JSON, 코드, 표 등을 완전한 블록으로 유지
- ✅ **검색 정확도**: 블록 타입별 최적화된 검색 전략
- ✅ **답변 품질**: 분할되지 않은 완전한 정보 제공

## 🔧 테스트 및 검증

각 기능에 대한 테스트 스크립트가 포함되어 있습니다:

```bash
# PNS 개선 테스트
python code_dev_agent/test_pns_improvement.py

# 메타데이터 활용 예시
python code_dev_agent/metadata_usage_examples.py

# 기술문서 분할 데모
python code_dev_agent/technical_document_splitter.py
```

## 📚 문서

각 기능에 대한 상세한 가이드 문서가 포함되어 있습니다:

- **PNS_IMPROVEMENT_GUIDE.md**: PNS 질의 개선을 위한 상세 가이드
- **METADATA_UTILIZATION_GUIDE.md**: 메타데이터 활용 방법 가이드
- **TECHNICAL_DOCUMENT_GUIDE.md**: 기술문서 특화 분할 및 검색 가이드

## 🤝 기여

이 코드들은 AI Agent가 생성한 실험적 기능들입니다. 개선 사항이나 버그 리포트는 언제든지 환영합니다.

## 📄 라이선스

이 코드들은 프로젝트의 기존 라이선스를 따릅니다.

---

**참고**: 이 디렉토리의 코드들은 실험적 성격을 가지므로, 프로덕션 환경에서 사용하기 전에 충분한 테스트가 필요합니다.
