# 계층별 헤더 기반 문서 분할기

RAG(Retrieval-Augmented Generation) 개발을 위한 마크다운 문서 분할 전략 구현

## 개요

이 프로젝트는 기술 문서의 마크다운 헤더 계층구조를 활용하여 RAG에 최적화된 문서 청크를 생성하는 시스템입니다. 특히 원스토어 개발자센터 가이드와 같은 기술 문서에서 코드값, 메시지, 함수 규격 등의 정보를 효과적으로 검색할 수 있도록 설계되었습니다.

## 주요 기능

### 📊 계층별 문서 분할
- `#` (H1) ~ `####` (H4) 헤더 레벨별 문서 분할
- 각 계층에 대해 개별 청크 생성
- 상위 컨텍스트 보존 옵션

### 🔍 스마트 청크 생성
- 최대 청크 크기 제한으로 적절한 크기 유지
- 긴 섹션 자동 분할
- 계층적 메타데이터 포함

### 💡 최적화된 검색
- 키워드 기반 관련 청크 검색
- 제목 및 내용 모두에서 검색 지원
- 청크별 상세 메타데이터 제공

## 파일 구조

```
code_dev_agent/16/
├── hierarchical_document_splitter.py    # 핵심 분할 로직
├── hierarchical_document_split_test.ipynb    # 테스트 노트북
├── requirements.txt                     # 의존성 패키지
└── README.md                           # 이 파일
```

## 사용법

### 1. 기본 사용

```python
from hierarchical_document_splitter import HierarchicalDocumentSplitter

# 분할기 초기화
splitter = HierarchicalDocumentSplitter(
    include_parent_context=True,  # 상위 컨텍스트 포함
    max_chunk_size=2000          # 최대 청크 크기
)

# 문서 읽기
with open('your_document.md', 'r', encoding='utf-8') as f:
    document_text = f.read()

# 문서 분할
chunks = splitter.split_document(document_text)

# 결과 확인
print(f"총 {len(chunks)}개 청크 생성")
splitter.print_chunk_summary(chunks)
```

### 2. 테스트 실행

Jupyter 노트북을 사용한 전체 테스트:

```bash
# Jupyter 설치 (필요시)
pip install jupyter notebook

# 노트북 실행
jupyter notebook hierarchical_document_split_test.ipynb
```

### 3. 특정 질의 테스트

```python
# PNS purchaseState 질의 예제
from hierarchical_document_splitter import demo_pns_query_test

demo_pns_query_test(chunks)
```

## 예제: PNS purchaseState 질의

### 질의
"PNS의 purchaseState는 어떤 값이 있나요?"

### 예상 답변
"COMPLETED : 결제완료 / CANCELED : 취소"

### 시스템 동작
1. **청크 검색**: "purchaseState" 키워드로 관련 청크 식별
2. **컨텍스트 추출**: PNS 관련 섹션에서 purchaseState 정보 추출
3. **답변 생성**: COMPLETED와 CANCELED 값을 정확히 반환

## 클래스 및 메서드

### HierarchicalDocumentSplitter

주요 메서드:
- `split_document(text)`: 문서를 계층별로 분할
- `get_chunks_by_level(chunks, level)`: 특정 레벨 청크 반환
- `find_relevant_chunks(chunks, query)`: 쿼리 관련 청크 검색
- `export_chunks_to_json(chunks, filename)`: JSON 형태로 내보내기
- `print_chunk_summary(chunks)`: 청크 요약 정보 출력

### DocumentChunk

청크 데이터 구조:
- `content`: 청크 내용
- `level`: 헤더 레벨 (1-4)
- `title`: 섹션 제목
- `full_path`: 계층적 경로
- `metadata`: 추가 메타데이터
- `start_line`, `end_line`: 원본 문서 라인 범위

## 특징

### ✅ 장점
- **계층적 구조 보존**: 문서의 논리적 구조 유지
- **상위 컨텍스트 포함**: 검색 정확도 향상
- **유연한 청크 크기**: RAG 모델에 최적화된 크기 조절
- **풍부한 메타데이터**: 검색 및 필터링 지원

### 🎯 적용 분야
- 기술 문서 (API 가이드, 개발자 문서)
- 제품 매뉴얼
- 정책/규정 문서
- FAQ 및 가이드

## 성능 최적화

### 청크 크기 조정
```python
# 작은 청크 (빠른 검색)
splitter = HierarchicalDocumentSplitter(max_chunk_size=1000)

# 큰 청크 (풍부한 컨텍스트)
splitter = HierarchicalDocumentSplitter(max_chunk_size=3000)
```

### 컨텍스트 설정
```python
# 상위 컨텍스트 제외 (간결한 청크)
splitter = HierarchicalDocumentSplitter(include_parent_context=False)

# 상위 컨텍스트 포함 (풍부한 정보)
splitter = HierarchicalDocumentSplitter(include_parent_context=True)
```

## 향후 개선 사항

1. **테이블 처리**: 마크다운 테이블 특별 처리
2. **코드 블록 보존**: 코드 블록 내용 완전 보존
3. **이미지 메타데이터**: 이미지 링크 및 설명 처리
4. **임베딩 통합**: 벡터 임베딩 생성 및 유사도 검색
5. **성능 최적화**: 대용량 문서 처리 최적화

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다.
