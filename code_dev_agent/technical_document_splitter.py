"""
기술문서 특화 분할 전략

이 모듈은 기술문서의 특성(JSON 규격, 코드 블록, 표 등)을 고려하여
전체 맥락을 보존하면서도 검색 효율성을 보장하는 분할 전략을 제공합니다.
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ContentBlockType(Enum):
    """콘텐츠 블록 타입"""
    JSON_SPECIFICATION = "json_specification"
    CODE_BLOCK = "code_block"
    TABLE = "table"
    API_ENDPOINT = "api_endpoint"
    ERROR_CODE = "error_code"
    HEADER_SECTION = "header_section"
    TEXT_CONTENT = "text_content"


@dataclass
class ContentBlock:
    """콘텐츠 블록 정보"""
    content: str
    block_type: ContentBlockType
    start_line: int
    end_line: int
    metadata: Dict[str, Any]
    is_complete: bool = True


class TechnicalDocumentSplitter:
    """기술문서 특화 분할기"""
    
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.raw_text = self._load_document()
        self.lines = self.raw_text.split('\n')
        
    def _load_document(self) -> str:
        """문서 로드"""
        with open(self.document_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def split_documents(self) -> List[Document]:
        """기술문서 최적화 분할"""
        print("🚀 기술문서 특화 분할 시작...")
        
        # 1단계: 콘텐츠 블록 식별
        content_blocks = self._identify_content_blocks()
        print(f"📋 식별된 콘텐츠 블록: {len(content_blocks)}개")
        
        # 2단계: 블록 타입별 분류
        block_groups = self._group_blocks_by_type(content_blocks)
        
        # 3단계: 블록별 최적화된 문서 생성
        documents = self._create_optimized_documents(block_groups)
        
        print(f"✅ 총 {len(documents)}개 최적화된 문서 생성")
        return documents
    
    def _identify_content_blocks(self) -> List[ContentBlock]:
        """콘텐츠 블록 식별"""
        blocks = []
        current_block = None
        block_start = 0
        
        for i, line in enumerate(self.lines):
            # JSON 규격 블록 시작
            if self._is_json_spec_start(line, i):
                if current_block:
                    blocks.append(current_block)
                current_block = ContentBlock(
                    content=line,
                    block_type=ContentBlockType.JSON_SPECIFICATION,
                    start_line=i,
                    end_line=i,
                    metadata={'json_depth': 0}
                )
                continue
            
            # 코드 블록 시작
            if self._is_code_block_start(line):
                if current_block:
                    blocks.append(current_block)
                current_block = ContentBlock(
                    content=line,
                    block_type=ContentBlockType.CODE_BLOCK,
                    start_line=i,
                    end_line=i,
                    metadata={'language': self._extract_language(line)}
                )
                continue
            
            # 표 시작
            if self._is_table_start(line):
                if current_block:
                    blocks.append(current_block)
                current_block = ContentBlock(
                    content=line,
                    block_type=ContentBlockType.TABLE,
                    start_line=i,
                    end_line=i,
                    metadata={'table_headers': self._extract_table_headers(line)}
                )
                continue
            
            # API 엔드포인트
            if self._is_api_endpoint(line):
                if current_block:
                    blocks.append(current_block)
                current_block = ContentBlock(
                    content=line,
                    block_type=ContentBlockType.API_ENDPOINT,
                    start_line=i,
                    end_line=i,
                    metadata={'method': self._extract_http_method(line)}
                )
                continue
            
            # 헤더 섹션
            if self._is_header_section(line):
                if current_block:
                    blocks.append(current_block)
                current_block = ContentBlock(
                    content=line,
                    block_type=ContentBlockType.HEADER_SECTION,
                    start_line=i,
                    end_line=i,
                    metadata={'header_level': self._extract_header_level(line)}
                )
                continue
            
            # 현재 블록에 라인 추가
            if current_block:
                current_block.content += '\n' + line
                current_block.end_line = i
                
                # 블록 완성 여부 확인
                if self._is_block_complete(current_block, line, i):
                    blocks.append(current_block)
                    current_block = None
        
        # 마지막 블록 처리
        if current_block:
            blocks.append(current_block)
        
        return blocks
    
    def _is_json_spec_start(self, line: str, line_num: int) -> bool:
        """JSON 규격 시작 여부 확인"""
        line_stripped = line.strip()
        
        # JSON 객체 시작 패턴
        json_patterns = [
            r'^\s*\{.*\}\s*$',  # 한 줄 JSON
            r'^\s*\{.*$',       # JSON 시작
            r'^\s*"msgVersion"\s*:',  # PNS 메시지 시작
            r'^\s*"clientId"\s*:',    # 클라이언트 ID 시작
        ]
        
        for pattern in json_patterns:
            if re.match(pattern, line_stripped):
                return True
        
        # 이전 라인에서 JSON 컨텍스트 확인
        if line_num > 0:
            prev_line = self.lines[line_num - 1].strip()
            if 'json' in prev_line.lower() or 'message' in prev_line.lower():
                if line_stripped.startswith('{') or line_stripped.startswith('"'):
                    return True
        
        return False
    
    def _is_code_block_start(self, line: str) -> bool:
        """코드 블록 시작 여부 확인"""
        line_stripped = line.strip()
        
        # 마크다운 코드 블록
        if line_stripped.startswith('```'):
            return True
        
        # 들여쓰기된 코드 블록
        if line_stripped and not line_stripped.startswith('#'):
            # 이전 라인이 헤더이고 현재 라인이 코드인 경우
            return True
        
        return False
    
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
    
    def _is_api_endpoint(self, line: str) -> bool:
        """API 엔드포인트 여부 확인"""
        line_stripped = line.strip()
        
        # HTTP 메서드 패턴
        http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
        for method in http_methods:
            if line_stripped.startswith(method):
                return True
        
        # URL 패턴
        if re.match(r'^\s*(https?://|/api/|/v\d+/).*', line_stripped):
            return True
        
        return False
    
    def _is_header_section(self, line: str) -> bool:
        """헤더 섹션 여부 확인"""
        line_stripped = line.strip()
        
        # 마크다운 헤더
        if line_stripped.startswith('#'):
            return True
        
        # HTML 헤더
        if re.match(r'^\s*<h[1-6]>.*</h[1-6]>\s*$', line_stripped):
            return True
        
        return False
    
    def _extract_language(self, line: str) -> str:
        """코드 블록 언어 추출"""
        if line.strip().startswith('```'):
            language = line.strip()[3:].strip()
            return language if language else 'text'
        return 'text'
    
    def _extract_table_headers(self, line: str) -> List[str]:
        """표 헤더 추출"""
        headers = []
        if '|' in line:
            parts = line.split('|')
            for part in parts[1:-1]:  # 첫 번째와 마지막 빈 부분 제외
                header = part.strip()
                if header:
                    headers.append(header)
        return headers
    
    def _extract_http_method(self, line: str) -> str:
        """HTTP 메서드 추출"""
        http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
        for method in http_methods:
            if line.strip().startswith(method):
                return method
        return 'UNKNOWN'
    
    def _extract_header_level(self, line: str) -> int:
        """헤더 레벨 추출"""
        line_stripped = line.strip()
        if line_stripped.startswith('#'):
            return line_stripped.count('#')
        return 1
    
    def _is_block_complete(self, block: ContentBlock, current_line: str, line_num: int) -> bool:
        """블록 완성 여부 확인"""
        current_line_stripped = current_line.strip()
        
        if block.block_type == ContentBlockType.JSON_SPECIFICATION:
            # JSON 블록 완성 확인
            return self._is_json_complete(block.content)
        
        elif block.block_type == ContentBlockType.CODE_BLOCK:
            # 코드 블록 완성 확인
            if block.content.startswith('```'):
                return current_line_stripped.endswith('```')
            return False
        
        elif block.block_type == ContentBlockType.TABLE:
            # 표 완성 확인
            return self._is_table_complete(current_line_stripped)
        
        elif block.block_type == ContentBlockType.HEADER_SECTION:
            # 헤더 섹션은 다음 헤더나 다른 블록 시작 시 완성
            if line_num + 1 < len(self.lines):
                next_line = self.lines[line_num + 1].strip()
                if (next_line.startswith('#') or 
                    self._is_json_spec_start(next_line, line_num + 1) or
                    self._is_code_block_start(next_line) or
                    self._is_table_start(next_line)):
                    return True
            return False
        
        return False
    
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
    
    def _is_table_complete(self, line: str) -> bool:
        """표 완성 여부 확인"""
        # 빈 줄이나 다른 콘텐츠 시작 시 표 완성
        if not line or not line.strip():
            return True
        
        # 다음 라인이 표가 아닌 경우
        if '|' not in line:
            return True
        
        return False
    
    def _group_blocks_by_type(self, blocks: List[ContentBlock]) -> Dict[ContentBlockType, List[ContentBlock]]:
        """블록 타입별 그룹핑"""
        groups: Dict[ContentBlockType, List[ContentBlock]] = {}
        for block_type in ContentBlockType:
            groups[block_type] = []
        
        for block in blocks:
            groups[block.block_type].append(block)
        
        return groups
    
    def _create_optimized_documents(self, block_groups: Dict[ContentBlockType, List[ContentBlock]]) -> List[Document]:
        """최적화된 문서 생성"""
        documents = []
        
        # 1. 완전한 블록들을 개별 문서로 생성
        for block_type, blocks in block_groups.items():
            for block in blocks:
                if block.is_complete:
                    doc = self._create_document_from_block(block)
                    documents.append(doc)
        
        # 2. 불완전한 블록들을 적절히 결합
        incomplete_blocks = []
        for block_type, blocks in block_groups.items():
            for block in blocks:
                if not block.is_complete:
                    incomplete_blocks.append(block)
        
        if incomplete_blocks:
            combined_docs = self._combine_incomplete_blocks(incomplete_blocks)
            documents.extend(combined_docs)
        
        return documents
    
    def _create_document_from_block(self, block: ContentBlock) -> Document:
        """블록에서 문서 생성"""
        # 블록 타입별 메타데이터 강화
        enhanced_metadata = {
            'block_type': block.block_type.value,
            'start_line': block.start_line,
            'end_line': block.end_line,
            'is_complete_block': block.is_complete,
            'content_length': len(block.content),
            **block.metadata
        }
        
        # 블록 타입별 특별 처리
        if block.block_type == ContentBlockType.JSON_SPECIFICATION:
            enhanced_metadata['content_type'] = 'json_specification'
            enhanced_metadata['is_complete_spec'] = True
            enhanced_metadata['contains_structured_data'] = True
        
        elif block.block_type == ContentBlockType.CODE_BLOCK:
            enhanced_metadata['content_type'] = 'code_example'
            enhanced_metadata['programming_language'] = block.metadata.get('language', 'text')
        
        elif block.block_type == ContentBlockType.TABLE:
            enhanced_metadata['content_type'] = 'data_table'
            enhanced_metadata['table_headers'] = block.metadata.get('table_headers', [])
        
        elif block.block_type == ContentBlockType.API_ENDPOINT:
            enhanced_metadata['content_type'] = 'api_endpoint'
            enhanced_metadata['http_method'] = block.metadata.get('method', 'UNKNOWN')
        
        # 컨텍스트 강화
        enhanced_content = self._enhance_block_context(block)
        
        return Document(
            page_content=enhanced_content,
            metadata=enhanced_metadata
        )
    
    def _enhance_block_context(self, block: ContentBlock) -> str:
        """블록 컨텍스트 강화"""
        context_info = f"[블록 타입]: {block.block_type.value}\n"
        context_info += f"[라인 범위]: {block.start_line + 1}-{block.end_line + 1}\n"
        
        if block.block_type == ContentBlockType.JSON_SPECIFICATION:
            context_info += "[설명]: 이 내용은 JSON 메시지 규격입니다. 전체 구조를 파악하기 위해 완전한 형태로 유지됩니다.\n\n"
        elif block.block_type == ContentBlockType.CODE_BLOCK:
            language = block.metadata.get('language', 'text')
            context_info += f"[설명]: 이 내용은 {language} 코드 예제입니다.\n\n"
        elif block.block_type == ContentBlockType.TABLE:
            context_info += "[설명]: 이 내용은 데이터 테이블입니다. 전체 구조를 파악하기 위해 완전한 형태로 유지됩니다.\n\n"
        elif block.block_type == ContentBlockType.API_ENDPOINT:
            method = block.metadata.get('method', 'UNKNOWN')
            context_info += f"[설명]: 이 내용은 {method} API 엔드포인트 정보입니다.\n\n"
        
        return context_info + block.content
    
    def _combine_incomplete_blocks(self, blocks: List[ContentBlock]) -> List[Document]:
        """불완전한 블록 결합"""
        documents = []
        
        # 연속된 블록들을 결합
        current_combined: List[ContentBlock] = []
        current_content = ""
        
        for block in blocks:
            # 새로운 헤더 섹션이 시작되면 이전 결합 완료
            if (block.block_type == ContentBlockType.HEADER_SECTION and 
                current_combined and 
                current_combined[-1].block_type != ContentBlockType.HEADER_SECTION):
                
                if current_content:
                    doc = self._create_combined_document(current_combined, current_content)
                    documents.append(doc)
                    current_combined = []
                    current_content = ""
            
            current_combined.append(block)
            current_content += block.content + "\n\n"
        
        # 마지막 결합 처리
        if current_content:
            doc = self._create_combined_document(current_combined, current_content)
            documents.append(doc)
        
        return documents
    
    def _create_combined_document(self, blocks: List[ContentBlock], content: str) -> Document:
        """결합된 문서 생성"""
        # 메타데이터 병합
        block_types: List[str] = [block.block_type.value for block in blocks]
        combined_metadata = {
            'block_types': block_types,
            'start_line': blocks[0].start_line,
            'end_line': blocks[-1].end_line,
            'is_combined_block': True,
            'content_length': len(content),
            'content_type': 'combined_section'
        }
        
        # 컨텍스트 강화
        context_info = "[결합된 섹션]: 여러 블록이 결합된 완전한 섹션입니다.\n"
        context_info += f"[포함된 블록]: {', '.join(block_types)}\n\n"
        
        enhanced_content = context_info + content
        
        return Document(
            page_content=enhanced_content,
            metadata=combined_metadata
        )


class TechnicalDocumentAnalyzer:
    """기술문서 분석기"""
    
    def __init__(self):
        self.block_patterns = {
            'json_spec': r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
            'code_block': r'```[\s\S]*?```',
            'table': r'\|.*\|.*\|',
            'api_endpoint': r'(GET|POST|PUT|DELETE|PATCH)\s+[^\s]+',
        }
    
    def analyze_document_structure(self, documents: List[Document]) -> Dict[str, Any]:
        """문서 구조 분석"""
        analysis: Dict[str, Any] = {
            'total_documents': len(documents),
            'block_type_distribution': {},
            'content_type_distribution': {},
            'completeness_analysis': {},
            'size_analysis': {}
        }
        
        for doc in documents:
            metadata = doc.metadata
            
            # 블록 타입 분포
            block_type = metadata.get('block_type', 'unknown')
            analysis['block_type_distribution'][block_type] = \
                analysis['block_type_distribution'].get(block_type, 0) + 1
            
            # 콘텐츠 타입 분포
            content_type = metadata.get('content_type', 'unknown')
            analysis['content_type_distribution'][content_type] = \
                analysis['content_type_distribution'].get(content_type, 0) + 1
            
            # 완성도 분석
            is_complete = metadata.get('is_complete_block', False)
            analysis['completeness_analysis']['complete' if is_complete else 'incomplete'] = \
                analysis['completeness_analysis'].get('complete' if is_complete else 'incomplete', 0) + 1
            
            # 크기 분석
            content_length = metadata.get('content_length', 0)
            if content_length > 1000:
                size_category = 'large'
            elif content_length > 500:
                size_category = 'medium'
            else:
                size_category = 'small'
            analysis['size_analysis'][size_category] = \
                analysis['size_analysis'].get(size_category, 0) + 1
        
        return analysis
    
    def print_analysis_report(self, analysis: Dict[str, Any]):
        """분석 리포트 출력"""
        print("📊 기술문서 구조 분석 리포트")
        print("=" * 50)
        
        print(f"📄 총 문서 수: {analysis['total_documents']}")
        
        print(f"\n📂 블록 타입별 분포:")
        for block_type, count in analysis['block_type_distribution'].items():
            percentage = count / analysis['total_documents'] * 100
            print(f"  - {block_type}: {count}개 ({percentage:.1f}%)")
        
        print(f"\n📋 콘텐츠 타입별 분포:")
        for content_type, count in analysis['content_type_distribution'].items():
            percentage = count / analysis['total_documents'] * 100
            print(f"  - {content_type}: {count}개 ({percentage:.1f}%)")
        
        print(f"\n✅ 완성도 분석:")
        for completeness, count in analysis['completeness_analysis'].items():
            percentage = count / analysis['total_documents'] * 100
            print(f"  - {completeness}: {count}개 ({percentage:.1f}%)")
        
        print(f"\n📏 크기별 분포:")
        for size, count in analysis['size_analysis'].items():
            percentage = count / analysis['total_documents'] * 100
            print(f"  - {size}: {count}개 ({percentage:.1f}%)")


# 사용 예시
def demonstrate_technical_splitting():
    """기술문서 분할 데모"""
    print("🚀 기술문서 특화 분할 데모")
    print("=" * 50)
    
    # 샘플 기술문서 생성
    sample_doc = """
# API 메시지 규격

## 요청 메시지

다음은 API 요청 메시지의 JSON 규격입니다:

```json
{
  "msgVersion": "3.1.0",
  "clientId": "0000000001",
  "productId": "0900001234",
  "messageType": "SINGLE_PAYMENT_TRANSACTION",
  "purchaseId": "SANDBOX3000000004564",
  "developerPayload": "OS_000211234",
  "purchaseTimeMillis": 24431212233,
  "purchaseState": "COMPLETED",
  "price": "10000",
  "priceCurrencyCode": "KRW",
  "productName": "GOLD100(+20)",
  "paymentTypeList": [
    {
      "paymentMethod": "DCB",
      "amount": "3000"
    },
    {
      "paymentMethod": "ONESTORECASH",
      "amount": "7000"
    }
  ],
  "billingKey": "36FED4C6E4AC9E29ADAF356057DB98B5CB92126B1D52E8757701E3A261AF49CCFBFC49F5FEF6E277A7A10E9076B523D839E9D84CE9225498155C5065529E22F5",
  "isTestMdn": true,
  "purchaseToken": "TOKEN...",
  "environment": "SANDBOX",
  "marketCode": "MKT_ONE",
  "signature": "SIGNATURE..."
}
```

## 응답 코드

| 코드 | 설명 |
|------|------|
| 200 | 성공 |
| 400 | 잘못된 요청 |
| 500 | 서버 오류 |

## API 엔드포인트

POST /api/v1/payment/notification
Content-Type: application/json

이 엔드포인트는 결제 알림을 처리합니다.
"""
    
    # 임시 파일로 저장
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(sample_doc)
        temp_file = f.name
    
    try:
        # 기술문서 분할기 사용
        splitter = TechnicalDocumentSplitter(temp_file)
        documents = splitter.split_documents()
        
        # 분석기 사용
        analyzer = TechnicalDocumentAnalyzer()
        analysis = analyzer.analyze_document_structure(documents)
        analyzer.print_analysis_report(analysis)
        
        # 개별 문서 확인
        print(f"\n📄 생성된 문서들:")
        for i, doc in enumerate(documents, 1):
            print(f"\n문서 {i}:")
            print(f"  - 타입: {doc.metadata.get('block_type', 'unknown')}")
            print(f"  - 완성도: {doc.metadata.get('is_complete_block', False)}")
            print(f"  - 크기: {doc.metadata.get('content_length', 0)}자")
            print(f"  - 내용 미리보기: {doc.page_content[:100]}...")
    
    finally:
        # 임시 파일 삭제
        import os
        os.unlink(temp_file)


if __name__ == "__main__":
    demonstrate_technical_splitting()
