# 🎯 메타데이터 활용 가이드

## 📋 개요

메타데이터는 RAG 시스템에서 검색 정확도와 답변 품질을 크게 향상시킬 수 있는 핵심 요소입니다. 이 가이드는 메타데이터를 효과적으로 활용하는 방법을 구체적으로 설명합니다.

## 🔍 메타데이터의 핵심 역할

### 1. 검색 정확도 향상
- **컨텍스트 인식**: 문서의 내용과 맥락을 정확히 파악
- **우선순위 설정**: 중요한 문서를 우선적으로 검색
- **필터링**: 관련 없는 문서를 사전에 제외

### 2. 답변 품질 개선
- **완성도 보장**: 완전한 정보를 포함한 문서 우선 선택
- **관련성 강화**: 질의와 직접 관련된 문서 집중
- **컨텍스트 보존**: 문서 간의 관계 정보 유지

## 📊 메타데이터 구조

### 기본 메타데이터 필드

```python
metadata = {
    # 문서 식별
    'section_name': 'PNS 메시지 규격',
    'content_type': 'message_specification',
    
    # 관련성 표시
    'contains_pns': True,
    'contains_purchasestate': True,
    
    # 완성도 표시
    'is_complete_spec': True,
    
    # 기술적 정보
    'chunk_size': 150,
    'chunk_index': 0,
    
    # 계층 정보
    'title_hierarchy': 'PNS > 메시지 규격 > Payment Notification',
    'hierarchy_level': 'medium'
}
```

### 확장 메타데이터 필드

```python
enhanced_metadata = {
    # 기본 필드들...
    
    # 우선순위 정보
    'priority_level': 'high',
    'search_boost': 2.0,
    
    # 키워드 정보
    'keyword_density': {
        'pns': 0.05,
        'purchasestate': 0.03,
        'signature': 0.02
    },
    
    # 부스트 팩터
    'boost_factors': [
        'complete_specification',
        'purchase_state_related',
        'pns_related'
    ],
    
    # 컨텍스트 정보
    'context_relevance': 0.85,
    'completeness_score': 0.95
}
```

## 🛠️ 메타데이터 활용 방법

### 1. 메타데이터 분석기 (MetadataAnalyzer)

```python
from metadata_utilization_guide import MetadataAnalyzer

# 분석기 초기화
analyzer = MetadataAnalyzer()

# 문서 메타데이터 분석
doc = Document(page_content="...", metadata={...})
analysis = analyzer.analyze_document_metadata(doc)

print(f"내용 타입: {analysis['content_type']}")
print(f"우선순위: {analysis['priority_level']}")
print(f"완성도 점수: {analysis['completeness_score']:.2f}")
print(f"부스트 팩터: {analysis['search_boost_factors']}")
```

#### 분석 결과 예시
```
📊 문서 분석 결과:
  - 내용 타입: message_specification
  - 우선순위: high
  - 완성도 점수: 0.95
  - 컨텍스트 관련성: 0.85
  - 부스트 팩터: ['complete_specification', 'purchase_state_related']
```

### 2. 메타데이터 기반 검색기 (MetadataBasedRetriever)

```python
from metadata_utilization_guide import MetadataBasedRetriever

# 검색기 초기화
retriever = MetadataBasedRetriever(documents)

# 검색 기준 설정
search_criteria = {
    'boost_factors': ['purchase_state_related', 'complete_specification']
}

# 메타데이터 기반 검색
query = "PNS 메시지의 purchaseState 값은 무엇이 있나요?"
results = retriever.search_by_metadata(query, search_criteria)

for score, doc in results:
    print(f"점수: {score:.2f}, 내용: {doc.page_content[:50]}...")
```

#### 검색 결과 예시
```
🔍 질의: PNS 메시지의 purchaseState 값은 무엇이 있나요?

📋 검색 결과:
  1. 점수: 8.50
     내용: | purchaseState | String | COMPLETED: 결제완료 / CANCELED: 취소 |
     타입: message_specification
  
  2. 점수: 6.20
     내용: PNS 메시지의 purchaseState 필드는 COMPLETED 또는 CANCELED 값을 가집니다.
     타입: purchase_state_info
```

### 3. 메타데이터 강화기 (MetadataEnhancer)

```python
from metadata_utilization_guide import MetadataEnhancer

# 메타데이터 강화
additional_info = {
    'priority_level': 'high',
    'processing_method': 'notification_handler',
    'estimated_complexity': 'intermediate',
    'related_topics': ['message_processing', 'error_handling']
}

enhanced_doc = MetadataEnhancer.enhance_document_metadata(doc, additional_info)

# 검색용 메타데이터 생성
query = "PNS 메시지의 purchaseState 값은 무엇이 있나요?"
search_metadata = MetadataEnhancer.create_search_metadata(query)
```

### 4. 메타데이터 시각화기 (MetadataVisualizer)

```python
from metadata_utilization_guide import MetadataVisualizer

# 메타데이터 요약 출력
MetadataVisualizer.print_metadata_summary(documents)

# 개별 문서 메타데이터 출력
MetadataVisualizer.print_document_metadata(doc, show_content=True)
```

#### 시각화 결과 예시
```
📊 메타데이터 분석 요약
==================================================
📄 총 문서 수: 917
🔗 PNS 관련 문서: 36 (3.9%)
📋 완전한 메시지 규격: 4 (0.4%)
💰 purchaseState 포함: 27 (2.9%)

📂 내용 타입별 분포:
  - message_specification: 4개 (0.4%)
  - purchase_state_info: 23개 (2.5%)
  - general_pns: 9개 (1.0%)
```

## 🎯 고급 활용 기법

### 1. 동적 부스트 팩터

```python
def calculate_dynamic_boost(query: str, doc: Document) -> float:
    """동적 부스트 팩터 계산"""
    boost = 1.0
    
    # 질의 타입별 부스트
    if 'purchasestate' in query.lower():
        if doc.metadata.get('contains_purchasestate', False):
            boost *= 2.0
        if doc.metadata.get('is_complete_spec', False):
            boost *= 1.5
    
    # 완성도 기반 부스트
    if doc.metadata.get('is_complete_spec', False):
        boost *= 1.8
    
    # 우선순위 기반 부스트
    priority = doc.metadata.get('priority_level', 'low')
    if priority == 'high':
        boost *= 1.3
    
    return boost
```

### 2. 컨텍스트 기반 필터링

```python
def context_aware_filtering(docs: List[Document], query_context: str) -> List[Document]:
    """컨텍스트 인식 필터링"""
    filtered_docs = []
    
    for doc in docs:
        # 컨텍스트 관련성 계산
        context_score = calculate_context_relevance(doc, query_context)
        
        # 임계값 이상만 선택
        if context_score > 0.5:
            filtered_docs.append(doc)
    
    return filtered_docs
```

### 3. 메타데이터 기반 점수 계산

```python
def calculate_metadata_score(query: str, doc: Document) -> float:
    """메타데이터 기반 점수 계산"""
    score = 0.0
    
    # 1. 내용 타입 매칭 (가장 중요)
    query_type = identify_query_type(query)
    if doc.metadata.get('content_type') == query_type:
        score += 5.0
    
    # 2. 완성도 점수
    if doc.metadata.get('is_complete_spec', False):
        score += 3.0
    
    # 3. 키워드 밀도
    keyword_density = doc.metadata.get('keyword_density', {})
    for keyword, density in keyword_density.items():
        if keyword in query.lower():
            score += density * 10
    
    # 4. 우선순위 보너스
    priority = doc.metadata.get('priority_level', 'low')
    if priority == 'high':
        score += 2.0
    
    return score
```

## 📈 성능 최적화 팁

### 1. 메타데이터 인덱싱

```python
# 메타데이터 인덱스 생성
metadata_index = {
    'content_type': {},
    'priority_level': {},
    'boost_factors': {},
    'contains_pns': {'true': [], 'false': []},
    'is_complete_spec': {'true': [], 'false': []}
}

# 인덱스 구축
for i, doc in enumerate(documents):
    metadata = doc.metadata
    
    # 내용 타입별 인덱싱
    content_type = metadata.get('content_type', 'unknown')
    if content_type not in metadata_index['content_type']:
        metadata_index['content_type'][content_type] = []
    metadata_index['content_type'][content_type].append(i)
    
    # 부스트 팩터별 인덱싱
    boost_factors = metadata.get('boost_factors', [])
    for factor in boost_factors:
        if factor not in metadata_index['boost_factors']:
            metadata_index['boost_factors'][factor] = []
        metadata_index['boost_factors'][factor].append(i)
```

### 2. 캐싱 전략

```python
# 메타데이터 분석 결과 캐싱
metadata_cache = {}

def get_cached_analysis(doc_id: str) -> Dict[str, Any]:
    """캐시된 메타데이터 분석 결과 조회"""
    if doc_id not in metadata_cache:
        doc = get_document_by_id(doc_id)
        metadata_cache[doc_id] = analyzer.analyze_document_metadata(doc)
    
    return metadata_cache[doc_id]
```

### 3. 점수 정규화

```python
def normalize_metadata_scores(scores: List[float]) -> List[float]:
    """메타데이터 점수 정규화"""
    if not scores:
        return scores
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    normalized = []
    for score in scores:
        normalized_score = (score - min_score) / (max_score - min_score)
        normalized.append(normalized_score)
    
    return normalized
```

## 🔧 실제 구현 예시

### PNS 질의 최적화

```python
def optimize_pns_query(query: str, documents: List[Document]) -> List[Document]:
    """PNS 질의 최적화"""
    
    # 1. 메타데이터 분석기 초기화
    analyzer = MetadataAnalyzer()
    
    # 2. 질의 분석
    query_analysis = analyze_query_intent(query)
    
    # 3. 메타데이터 기반 사전 필터링
    filtered_docs = []
    for doc in documents:
        analysis = analyzer.analyze_document_metadata(doc)
        
        # 관련성 체크
        if not is_relevant_to_query(doc, query_analysis):
            continue
        
        # 완성도 체크
        if query_analysis['requires_complete_spec'] and not analysis['completeness_score'] > 0.8:
            continue
        
        filtered_docs.append(doc)
    
    # 4. 점수 계산 및 정렬
    scored_docs = []
    for doc in filtered_docs:
        score = calculate_comprehensive_score(doc, query_analysis)
        scored_docs.append((score, doc))
    
    # 5. 상위 결과 반환
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:10]]
```

## 📊 성능 측정

### 메타데이터 활용 효과 측정

```python
def measure_metadata_effectiveness():
    """메타데이터 활용 효과 측정"""
    
    # 테스트 질의
    test_queries = [
        "PNS 메시지의 purchaseState 값은 무엇이 있나요?",
        "signature 검증은 어떻게 하나요?",
        "PNS 메시지 규격을 알려주세요"
    ]
    
    results = {
        'with_metadata': [],
        'without_metadata': []
    }
    
    for query in test_queries:
        # 메타데이터 활용 검색
        metadata_results = search_with_metadata(query)
        results['with_metadata'].append(metadata_results)
        
        # 기본 검색
        basic_results = search_without_metadata(query)
        results['without_metadata'].append(basic_results)
    
    # 성능 비교
    compare_performance(results)
```

## 🎯 권장사항

### 1. 메타데이터 설계 원칙
- **명확성**: 메타데이터 필드의 의미가 명확해야 함
- **일관성**: 동일한 정보는 항상 같은 필드에 저장
- **확장성**: 새로운 메타데이터 필드 추가가 용이해야 함
- **성능**: 메타데이터 접근이 빠르고 효율적이어야 함

### 2. 구현 우선순위
1. **기본 메타데이터**: contains_pns, content_type, is_complete_spec
2. **우선순위 메타데이터**: priority_level, boost_factors
3. **고급 메타데이터**: keyword_density, context_relevance
4. **확장 메타데이터**: related_topics, complexity_level

### 3. 모니터링 및 개선
- 메타데이터 활용 효과 정기 측정
- 사용자 피드백 기반 메타데이터 개선
- 새로운 메타데이터 필드 추가 검토

## 📞 지원 및 문의

추가 질문이나 개선 사항이 있으시면 언제든지 문의해주세요.

---

**참고**: 이 가이드는 PNS 관련 질의 개선을 위한 것이며, 다른 도메인에도 유사한 메타데이터 활용 전략을 적용할 수 있습니다.
