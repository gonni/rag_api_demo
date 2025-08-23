"""
Hierarchical Document Splitter for Technical Documentation
계층적 기술 문서 분할기

특징:
1. 출처 URL 기반 1차 분할 (각 페이지별 독립성 보장)
2. 헤더 기반 계층적 분할 (# → ## → ### → ####)
3. 테이블과 코드 블록의 구조적 보존
4. 섹션별 컨텍스트 메타데이터 강화
5. 전문용어 추출 및 태깅
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from langchain.docstore.document import Document


@dataclass
class DocumentSection:
    """문서 섹션 정보"""
    content: str
    header_path: List[str]  # ["h1", "h2", "h3", "h4"] 계층
    level: int  # 헤더 레벨 (1-4)
    source_url: str
    section_id: str
    start_pos: int
    end_pos: int


class TechnicalTermExtractor:
    """기술 문서 전용 용어 추출기"""
    
    def __init__(self):
        self.patterns = {
            'api_codes': re.compile(r'\b[A-Z]{2,}[A-Z0-9_]*\b'),  # API, SDK, PNS 등
            'camel_case': re.compile(r'\b[a-z]+[A-Z][a-zA-Z0-9]*\b'),  # purchaseState 등
            'snake_case': re.compile(r'\b[a-z]+_[a-z0-9_]+\b'),  # query_purchases 등
            'version_codes': re.compile(r'\b[Vv]\d+(\.\d+)*\b'),  # V21, v1.0 등
            'http_codes': re.compile(r'\b[1-5]\d{2}\b'),  # 200, 404 등
            'method_names': re.compile(r'\b\w+\(\)\b'),  # launchPurchaseFlow() 등
            'class_names': re.compile(r'\b[A-Z][a-zA-Z0-9]*(?:Client|Listener|Params|Data|Result)\b'),
            'korean_tech': re.compile(r'(?:인앱|결제|구매|구독|월정액|상품|토큰|라이선스)'),
        }
        
        # 제외할 일반적인 단어들 (노이즈 감소)
        self.exclude_terms = {
            'IF', 'OR', 'TO', 'IS', 'ON', 'IN', 'AT', 'BY', 'UP', 'NO', 'OK',
            'GET', 'SET', 'PUT', 'POST', 'NEW', 'OLD', 'MAX', 'MIN', 'END'
        }
    
    def extract_terms(self, text: str) -> Dict[str, List[str]]:
        """텍스트에서 기술 용어들을 카테고리별로 추출"""
        terms = {}
        for category, pattern in self.patterns.items():
            matches = pattern.findall(text)
            # 중복 제거 및 제외 단어 필터링
            unique_matches = list(set(match for match in matches 
                                   if match.upper() not in self.exclude_terms))
            if unique_matches:
                terms[category] = unique_matches
        return terms


class HierarchicalSplitter:
    """계층적 문서 분할기"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 overlap_ratio: float = 0.1,
                 preserve_tables: bool = True,
                 preserve_code: bool = True):
        self.chunk_size = chunk_size
        self.overlap_size = int(chunk_size * overlap_ratio)
        self.preserve_tables = preserve_tables
        self.preserve_code = preserve_code
        self.term_extractor = TechnicalTermExtractor()
        
        # 출처 URL 패턴
        self.source_pattern = re.compile(r'^출처:\s*(https?://[^\s]+)', re.MULTILINE)
        
        # 헤더 패턴 (마크다운)
        self.header_patterns = {
            1: re.compile(r'^#\s+(.+)$', re.MULTILINE),
            2: re.compile(r'^##\s+(.+)$', re.MULTILINE),
            3: re.compile(r'^###\s+(.+)$', re.MULTILINE),
            4: re.compile(r'^####\s+(.+)$', re.MULTILINE),
        }
        
        # 테이블 패턴
        self.table_pattern = re.compile(r'^\|.+\|$', re.MULTILINE)
        
        # 코드 블록 패턴
        self.code_pattern = re.compile(r'```(\w+)?\n(.*?)\n```', re.DOTALL)
    
    def split_by_source(self, content: str) -> List[Tuple[str, str]]:
        """출처 URL 기준으로 문서를 1차 분할"""
        sources = list(self.source_pattern.finditer(content))
        sections = []
        
        for i, source_match in enumerate(sources):
            start_pos = source_match.start()
            end_pos = sources[i + 1].start() if i + 1 < len(sources) else len(content)
            
            source_url = source_match.group(1)
            section_content = content[start_pos:end_pos].strip()
            sections.append((source_url, section_content))
        
        return sections
    
    def extract_headers(self, content: str) -> List[Dict]:
        """텍스트에서 헤더 정보 추출"""
        headers = []
        
        for level in range(1, 5):  # h1 ~ h4
            pattern = self.header_patterns[level]
            for match in pattern.finditer(content):
                headers.append({
                    'level': level,
                    'title': match.group(1).strip(),
                    'position': match.start(),
                    'end_position': match.end()
                })
        
        # 위치 순으로 정렬
        headers.sort(key=lambda x: x['position'])
        return headers
    
    def build_hierarchy_path(self, headers: List[Dict], current_pos: int) -> List[str]:
        """현재 위치까지의 헤더 계층 경로 구성"""
        path = [''] * 4  # h1, h2, h3, h4
        
        for header in headers:
            if header['position'] > current_pos:
                break
            level = header['level']
            path[level - 1] = header['title']
            # 하위 레벨 초기화
            for i in range(level, 4):
                if i > level - 1:
                    path[i] = ''
        
        # 빈 문자열 제거하고 실제 경로만 반환
        return [p for p in path if p]
    
    def extract_tables(self, content: str) -> List[Dict]:
        """테이블 구조 추출 및 파싱"""
        tables = []
        lines = content.split('\n')
        table_start = None
        table_rows = []
        
        for i, line in enumerate(lines):
            if self.table_pattern.match(line):
                if table_start is None:
                    table_start = i
                table_rows.append(line)
            else:
                if table_start is not None:
                    # 테이블 종료
                    if len(table_rows) >= 2:  # 최소 헤더 + 데이터 1행
                        tables.append({
                            'start_line': table_start,
                            'end_line': i,
                            'rows': table_rows,
                            'content': '\n'.join(table_rows)
                        })
                    table_start = None
                    table_rows = []
        
        # 마지막 테이블 처리
        if table_start is not None and len(table_rows) >= 2:
            tables.append({
                'start_line': table_start,
                'end_line': len(lines),
                'rows': table_rows,
                'content': '\n'.join(table_rows)
            })
        
        return tables
    
    def extract_code_blocks(self, content: str) -> List[Dict]:
        """코드 블록 추출"""
        blocks = []
        for match in self.code_pattern.finditer(content):
            language = match.group(1) or 'text'
            code_content = match.group(2)
            blocks.append({
                'language': language,
                'content': code_content,
                'full_match': match.group(0),
                'start_pos': match.start(),
                'end_pos': match.end()
            })
        return blocks
    
    def split_content_by_headers(self, content: str, source_url: str) -> List[DocumentSection]:
        """헤더 기준으로 내용을 계층적으로 분할"""
        headers = self.extract_headers(content)
        sections = []
        
        if not headers:
            # 헤더가 없는 경우 전체를 하나의 섹션으로
            section = DocumentSection(
                content=content,
                header_path=[],
                level=0,
                source_url=source_url,
                section_id=f"section_0",
                start_pos=0,
                end_pos=len(content)
            )
            sections.append(section)
            return sections
        
        # 첫 번째 헤더 이전 내용
        if headers[0]['position'] > 0:
            intro_content = content[:headers[0]['position']].strip()
            if intro_content:
                section = DocumentSection(
                    content=intro_content,
                    header_path=[],
                    level=0,
                    source_url=source_url,
                    section_id=f"section_intro",
                    start_pos=0,
                    end_pos=headers[0]['position']
                )
                sections.append(section)
        
        # 헤더별 섹션 분할
        for i, header in enumerate(headers):
            start_pos = header['end_position']
            end_pos = headers[i + 1]['position'] if i + 1 < len(headers) else len(content)
            
            section_content = content[start_pos:end_pos].strip()
            if not section_content:
                continue
                
            hierarchy_path = self.build_hierarchy_path(headers, header['position'])
            
            section = DocumentSection(
                content=f"# {header['title']}\n\n{section_content}",
                header_path=hierarchy_path,
                level=header['level'],
                source_url=source_url,
                section_id=f"section_{i}",
                start_pos=start_pos,
                end_pos=end_pos
            )
            sections.append(section)
        
        return sections
    
    def chunk_large_section(self, section: DocumentSection) -> List[DocumentSection]:
        """큰 섹션을 청크로 분할"""
        if len(section.content) <= self.chunk_size:
            return [section]
        
        chunks = []
        content = section.content
        start_idx = 0
        chunk_id = 0
        
        while start_idx < len(content):
            end_idx = start_idx + self.chunk_size
            
            # 자연스러운 분할점 찾기 (문단, 문장 경계)
            if end_idx < len(content):
                # 문단 경계 찾기
                para_break = content.rfind('\n\n', start_idx, end_idx)
                if para_break > start_idx:
                    end_idx = para_break + 2
                else:
                    # 문장 경계 찾기
                    sent_break = content.rfind('. ', start_idx, end_idx)
                    if sent_break > start_idx:
                        end_idx = sent_break + 2
            
            chunk_content = content[start_idx:end_idx]
            
            chunk_section = DocumentSection(
                content=chunk_content,
                header_path=section.header_path,
                level=section.level,
                source_url=section.source_url,
                section_id=f"{section.section_id}_chunk_{chunk_id}",
                start_pos=section.start_pos + start_idx,
                end_pos=section.start_pos + end_idx
            )
            chunks.append(chunk_section)
            
            # 다음 청크 시작점 (오버랩 적용)
            start_idx = max(start_idx + 1, end_idx - self.overlap_size)
            chunk_id += 1
        
        return chunks
    
    def create_document_with_metadata(self, section: DocumentSection) -> Document:
        """DocumentSection을 LangChain Document로 변환 (메타데이터 포함)"""
        
        # 기술 용어 추출
        tech_terms = self.term_extractor.extract_terms(section.content)
        
        # 테이블 및 코드 블록 추출
        tables = self.extract_tables(section.content)
        code_blocks = self.extract_code_blocks(section.content)
        
        # 컨텐츠 타입 결정
        content_types = []
        if tables:
            content_types.append('table')
        if code_blocks:
            content_types.append('code')
        if not content_types:
            content_types.append('text')
        
        # 메타데이터 구성
        metadata = {
            'source_url': section.source_url,
            'header_path': section.header_path,
            'section_hierarchy': ' > '.join(section.header_path),
            'header_level': section.level,
            'section_id': section.section_id,
            'content_types': content_types,
            'technical_terms': tech_terms,
            'has_tables': len(tables) > 0,
            'has_code': len(code_blocks) > 0,
            'code_languages': [cb['language'] for cb in code_blocks],
            'table_count': len(tables),
            'code_block_count': len(code_blocks),
            'content_length': len(section.content),
            'start_pos': section.start_pos,
            'end_pos': section.end_pos
        }
        
        # 헤더 경로를 콘텐츠에 포함 (검색 개선)
        content_with_context = section.content
        if section.header_path:
            context_header = f"[{' > '.join(section.header_path)}]\n\n"
            content_with_context = context_header + content_with_context
        
        return Document(
            page_content=content_with_context,
            metadata=metadata
        )
    
    def split_document(self, content: str) -> List[Document]:
        """메인 분할 메소드"""
        documents = []
        
        # 1단계: 출처 URL 기준으로 분할
        source_sections = self.split_by_source(content)
        
        for source_url, section_content in source_sections:
            # 2단계: 헤더 기준으로 계층적 분할
            doc_sections = self.split_content_by_headers(section_content, source_url)
            
            for section in doc_sections:
                # 3단계: 큰 섹션은 청크로 분할
                chunks = self.chunk_large_section(section)
                
                # 4단계: Document 객체 생성
                for chunk in chunks:
                    doc = self.create_document_with_metadata(chunk)
                    documents.append(doc)
        
        return documents


# 사용 예제 및 테스트 함수
def test_splitter():
    """분할기 테스트"""
    
    # 샘플 텍스트 (실제 문서 구조 모방)
    sample_content = """
출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21.md
# 원스토어 인앱결제 API V7(SDK V21) 연동 안내

원스토어의 최신 인앱결제 API V7(SDK V21)이 출시되었습니다.

## 개요

원스토어 인앱결제(IAP)는 안드로이드 앱 내에서 상품을 판매하는 서비스입니다.

### PurchaseClient 초기화

```kotlin
val purchaseClient = PurchaseClient.getClient(activity)
```

### 상품 구매 요청

launchPurchaseFlow() 메소드를 사용합니다.

| 파라미터 | 타입 | 설명 |
|---------|-----|-----|
| productId | String | 상품 ID |
| purchaseParams | PurchaseParams | 구매 파라미터 |

출처: https://onestore-dev.gitbook.io/dev/tools/billing/v21/pns.md
# PNS(Payment Notification Service) 이용하기

PNS는 결제 상태를 서버로 전송하는 서비스입니다.

## PNS 설정

개발사 서버에서 API를 구현해야 합니다.
"""
    
    splitter = HierarchicalSplitter(chunk_size=500)
    documents = splitter.split_document(sample_content)
    
    print(f"총 {len(documents)}개의 문서로 분할됨\n")
    
    for i, doc in enumerate(documents):
        print(f"=== 문서 {i+1} ===")
        print(f"내용: {doc.page_content[:100]}...")
        print(f"출처: {doc.metadata['source_url']}")
        print(f"계층: {doc.metadata['section_hierarchy']}")
        print(f"타입: {doc.metadata['content_types']}")
        if doc.metadata['technical_terms']:
            print(f"기술용어: {doc.metadata['technical_terms']}")
        print()


if __name__ == "__main__":
    test_splitter()
