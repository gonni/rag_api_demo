# 🎯 기술문서 특화 분할 및 검색 가이드

## 📋 개요

기술문서는 JSON 규격, 코드 블록, 표, API 엔드포인트 등 구조화된 콘텐츠로 구성되어 있어서, 일반적인 텍스트 분할 방식으로는 전체 맥락을 파악하기 어려운 문제가 발생합니다. 이 가이드는 기술문서에 특화된 분할 및 검색 전략을 제시합니다.

## 🔍 문제 상황

### 기존 방식의 한계
- **JSON 규격 분할**: 큰 JSON 객체가 여러 청크로 나뉘어 전체 구조 파악 어려움
- **코드 블록 분할**: 코드 예제가 중간에 잘려서 실행 불가능한 상태
- **표 분할**: 데이터 테이블이 행 단위로 분할되어 의미 파악 어려움
- **컨텍스트 손실**: 관련 정보들이 서로 다른 청크에 분산

### 기술문서의 특성
- **구조화된 데이터**: JSON, XML, YAML 등
- **코드 블록**: 프로그래밍 언어 예제
- **데이터 테이블**: 마크다운 표, HTML 테이블
- **API 엔드포인트**: HTTP 메서드, URL, 헤더 정보
- **에러 코드**: 상태 코드, 에러 메시지

## 🚀 해결 방안

### 1. 블록 기반 분할 전략

#### 핵심 아이디어
- **완전한 블록 유지**: JSON, 코드, 표 등을 하나의 완전한 단위로 보존
- **블록 타입 식별**: 각 콘텐츠의 특성에 맞는 분할 전략 적용
- **컨텍스트 보존**: 블록 간의 관계 정보 유지

#### 블록 타입 분류
```python
class ContentBlockType(Enum):
    JSON_SPECIFICATION = "json_specification"    # JSON 규격
    CODE_BLOCK = "code_block"                    # 코드 블록
    TABLE = "table"                              # 데이터 테이블
    API_ENDPOINT = "api_endpoint"                # API 엔드포인트
    ERROR_CODE = "error_code"                    # 에러 코드
    HEADER_SECTION = "header_section"            # 헤더 섹션
    TEXT_CONTENT = "text_content"                # 일반 텍스트
```

### 2. 블록 식별 알고리즘

#### JSON 규격 식별
```python
def _is_json_spec_start(self, line: str, line_num: int) -> bool:
    """JSON 규격 시작 여부 확인"""
    json_patterns = [
        r'^\s*\{.*\}\s*$',                    # 한 줄 JSON
        r'^\s*\{.*$',                         # JSON 시작
        r'^\s*"msgVersion"\s*:',              # PNS 메시지 시작
        r'^\s*"clientId"\s*:',                # 클라이언트 ID 시작
    ]
    
    for pattern in json_patterns:
        if re.match(pattern, line.strip()):
            return True
    
    return False
```

#### 코드 블록 식별
```python
def _is_code_block_start(self, line: str) -> bool:
    """코드 블록 시작 여부 확인"""
    line_stripped = line.strip()
    
    # 마크다운 코드 블록
    if line_stripped.startswith('```'):
        return True
    
    # 들여쓰기된 코드 블록
    if line_stripped and not line_stripped.startswith('#'):
        return True
    
    return False
```

#### 표 식별
```python
def _is_table_start(self, line: str) -> bool:
    """표 시작 여부 확인"""
    line_stripped = line.strip()
    
    # 마크다운 표 패턴
    if '|' in line_stripped and line_stripped.count('|') >= 2:
        return True
    
    # HTML 테이블 패턴
    if line_stripped.startswith('<table') or line_stripped.startswith('<tr'):
        return True
    
    return False
```

### 3. 완성도 검증

#### JSON 완성도 확인
```python
def _is_json_complete(self, content: str) -> bool:
    """JSON 완성 여부 확인"""
    try:
        # JSON 파싱 시도
        json.loads(content)
        return True
    except json.JSONDecodeError:
        # 중괄호 균형 확인
        open_braces = content.count('{')
        close_braces = content.count('}')
        return open_braces == close_braces and open_braces > 0
```

#### 코드 블록 완성도 확인
```python
def _is_code_block_complete(self, content: str) -> bool:
    """코드 블록 완성 여부 확인"""
    if content.startswith('```'):
        return content.endswith('```')
    return True  # 들여쓰기된 코드는 컨텍스트 기반으로 판단
```

## 📊 구현 방법

### 1. 기술문서 분할기 사용

```python
from technical_document_splitter import TechnicalDocumentSplitter

# 기술문서 분할기 초기화
splitter = TechnicalDocumentSplitter("data/technical_document.md")

# 블록 기반 분할 실행
documents = splitter.split_documents()

print(f"생성된 문서 수: {len(documents)}")

# 블록 타입별 통계
block_types = {}
for doc in documents:
    block_type = doc.metadata.get('block_type', 'unknown')
    block_types[block_type] = block_types.get(block_type, 0) + 1

for block_type, count in block_types.items():
    print(f"{block_type}: {count}개")
```

### 2. 기술문서 검색기 사용

```python
from technical_document_retriever import TechnicalDocumentSearchEngine

# 검색 엔진 초기화
search_engine = TechnicalDocumentSearchEngine(documents)
search_engine.setup()

# 통합 검색
query = "JSON 메시지 규격이 어떻게 됩니까?"
result = search_engine.search(query)

print(f"질의 타입: {result['query_analysis']['query_type']}")
print(f"완전한 규격 필요: {result['query_analysis']['requires_complete_spec']}")
print(f"검색 결과: {result['total_results']}개")
print(f"완전한 블록: {result['complete_specs']}개")
```

### 3. 블록 타입별 전용 검색

```python
# JSON 규격 전용 검색
json_results = search_engine.search_json_specifications("메시지 규격")

# 코드 예제 전용 검색
code_results = search_engine.search_code_examples("구현 예제")

# 데이터 테이블 전용 검색
table_results = search_engine.search_data_tables("응답 코드")

# API 엔드포인트 전용 검색
api_results = search_engine.search_api_endpoints("엔드포인트 정보")
```

## 🎯 검색 최적화 전략

### 1. 블록 타입별 점수 계산

```python
def _calculate_block_type_score(self, query: str, block_type: str) -> float:
    """블록 타입별 점수 계산"""
    score = 0.0
    
    # JSON 규격 관련 질의
    if 'json' in query or '규격' in query or '메시지' in query:
        if block_type == 'json_specification':
            score += 5.0
    
    # 코드 관련 질의
    if '코드' in query or '예제' in query or 'code' in query:
        if block_type == 'code_block':
            score += 4.0
    
    # 표 관련 질의
    if '표' in query or 'table' in query or '코드' in query:
        if block_type == 'table':
            score += 4.0
    
    return score
```

### 2. 완전한 블록 우선순위

```python
def _prioritize_complete_blocks(self, scored_results: List[Tuple[float, Document]]) -> List[Tuple[float, Document]]:
    """완전한 블록 우선 정렬"""
    complete_blocks = []
    incomplete_blocks = []
    
    for score, doc in scored_results:
        if doc.metadata.get('is_complete_block', False):
            complete_blocks.append((score, doc))
        else:
            incomplete_blocks.append((score, doc))
    
    # 완전한 블록을 우선 정렬
    complete_blocks.sort(key=lambda x: x[0], reverse=True)
    incomplete_blocks.sort(key=lambda x: x[0], reverse=True)
    
    return complete_blocks + incomplete_blocks
```

### 3. 질의 타입별 최적화

```python
def analyze_query(self, query: str) -> Dict[str, Any]:
    """질의 분석"""
    analysis = {
        'query_type': 'general',
        'target_block_types': [],
        'requires_complete_spec': False,
        'confidence': 0.0
    }
    
    # 완전한 규격 필요 여부 확인
    if '전체' in query or '모든' in query or '규격' in query:
        analysis['requires_complete_spec'] = True
    
    # 블록 타입별 패턴 매칭
    if 'json' in query or '규격' in query:
        analysis['target_block_types'] = ['json_specification']
    elif '코드' in query or '예제' in query:
        analysis['target_block_types'] = ['code_block']
    elif '표' in query or '데이터' in query:
        analysis['target_block_types'] = ['table']
    
    return analysis
```

## 📈 성능 개선 효과

### 1. 컨텍스트 보존
- **JSON 규격**: 완전한 메시지 구조 유지
- **코드 블록**: 실행 가능한 완전한 예제 제공
- **데이터 테이블**: 전체 테이블 구조 보존

### 2. 검색 정확도 향상
- **블록 타입 매칭**: 질의와 블록 타입의 정확한 매칭
- **완전성 우선순위**: 완전한 블록 우선 검색
- **키워드 밀도**: 기술 용어 기반 정확한 매칭

### 3. 답변 품질 개선
- **전체 구조 파악**: 분할되지 않은 완전한 정보 제공
- **실행 가능한 코드**: 완전한 코드 예제 제공
- **정확한 데이터**: 테이블의 전체 구조 제공

## 🔧 실제 사용 예시

### 예시 1: JSON 메시지 규격 질의

```python
# 질의: "JSON 메시지 규격이 어떻게 됩니까?"

# 기존 방식 결과
# - 검색된 문서: 10개
# - 완전한 JSON: 0개 (분할됨)
# - 답변 품질: 낮음 (부분적 정보만)

# 기술문서 특화 방식 결과
# - 검색된 문서: 3개
# - 완전한 JSON: 1개 (완전한 블록)
# - 답변 품질: 높음 (전체 구조 제공)
```

### 예시 2: 코드 예제 질의

```python
# 질의: "signature 검증 코드 예제를 보여주세요"

# 기존 방식 결과
# - 검색된 문서: 8개
# - 완전한 코드: 0개 (중간에 잘림)
# - 실행 가능: 아니오

# 기술문서 특화 방식 결과
# - 검색된 문서: 2개
# - 완전한 코드: 1개 (완전한 블록)
# - 실행 가능: 예
```

### 예시 3: 데이터 테이블 질의

```python
# 질의: "응답 코드 표를 보여주세요"

# 기존 방식 결과
# - 검색된 문서: 6개
# - 완전한 표: 0개 (행 단위 분할)
# - 데이터 완성도: 낮음

# 기술문서 특화 방식 결과
# - 검색된 문서: 1개
# - 완전한 표: 1개 (완전한 블록)
# - 데이터 완성도: 높음
```

## 📊 성능 측정

### 분석 리포트 예시

```
📊 기술문서 구조 분석 리포트
==================================================
📄 총 문서 수: 150

📂 블록 타입별 분포:
  - json_specification: 25개 (16.7%)
  - code_block: 30개 (20.0%)
  - table: 20개 (13.3%)
  - api_endpoint: 15개 (10.0%)
  - header_section: 40개 (26.7%)
  - text_content: 20개 (13.3%)

📋 콘텐츠 타입별 분포:
  - json_specification: 25개 (16.7%)
  - code_example: 30개 (20.0%)
  - data_table: 20개 (13.3%)
  - api_endpoint: 15개 (10.0%)
  - combined_section: 60개 (40.0%)

✅ 완성도 분석:
  - complete: 90개 (60.0%)
  - incomplete: 60개 (40.0%)

📏 크기별 분포:
  - large: 30개 (20.0%)
  - medium: 80개 (53.3%)
  - small: 40개 (26.7%)
```

## 🎯 권장사항

### 1. 구현 우선순위
1. **JSON 규격 분할**: 가장 중요한 구조화된 데이터
2. **코드 블록 분할**: 실행 가능한 예제 보존
3. **데이터 테이블 분할**: 전체 구조 유지
4. **API 엔드포인트 분할**: 완전한 정보 제공

### 2. 검색 최적화
1. **완전한 블록 우선순위**: `is_complete_block` 필드 활용
2. **블록 타입별 점수**: 질의와 블록 타입 매칭
3. **키워드 밀도**: 기술 용어 기반 정확한 매칭

### 3. 성능 모니터링
- 블록 타입별 검색 정확도 측정
- 완전한 블록 비율 모니터링
- 사용자 피드백 기반 개선

## 📞 지원 및 문의

추가 질문이나 개선 사항이 있으시면 언제든지 문의해주세요.

---

**참고**: 이 가이드는 기술문서의 특성을 고려한 분할 및 검색 전략을 제시하며, 다른 도메인의 구조화된 문서에도 유사한 전략을 적용할 수 있습니다.
