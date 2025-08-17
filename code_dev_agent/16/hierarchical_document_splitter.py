"""
계층별 헤더 기반 문서 분할기
기술 문서의 헤더 계층에 따라 문서를 분할하여 RAG에 최적화된 청크를 생성합니다.
"""

import re
from typing import List, Dict
from dataclasses import dataclass
import json

@dataclass
class DocumentChunk:
    """문서 청크를 나타내는 클래스"""
    content: str
    level: int  # 헤더 레벨 (1=H1, 2=H2, 3=H3, 4=H4)
    title: str
    full_path: List[str]  # 계층적 경로 (예: ["07. PNS", "PNS 상세", "Payment Notification"])
    metadata: Dict
    start_line: int
    end_line: int

class HierarchicalDocumentSplitter:
    """계층별 헤더 기반 문서 분할기"""
    
    def __init__(self, include_parent_context: bool = True, max_chunk_size: int = 2000):
        """
        Args:
            include_parent_context: 상위 계층의 컨텍스트를 포함할지 여부
            max_chunk_size: 최대 청크 크기 (문자 수)
        """
        self.include_parent_context = include_parent_context
        self.max_chunk_size = max_chunk_size
        self.header_pattern = re.compile(r'^(#{1,4})\s+(.+?)(?:\s+<.*)?$', re.MULTILINE)
        
    def split_document(self, text: str) -> List[DocumentChunk]:
        """
        문서를 계층별로 분할합니다.
        
        Args:
            text: 분할할 마크다운 텍스트
            
        Returns:
            List[DocumentChunk]: 분할된 문서 청크들
        """
        lines = text.split('\n')
        chunks = []
        
        # 헤더 정보 추출
        headers = self._extract_headers(text)
        
        # 각 헤더별로 문서 청크 생성
        for i, header in enumerate(headers):
            start_line = header['line_num']
            end_line = headers[i + 1]['line_num'] - 1 if i + 1 < len(headers) else len(lines)
            
            # 해당 섹션의 내용 추출
            section_content = '\n'.join(lines[start_line:end_line])
            
            # 상위 컨텍스트 추가 (옵션)
            if self.include_parent_context:
                parent_context = self._get_parent_context(header, headers)
                if parent_context:
                    section_content = parent_context + '\n\n' + section_content
            
            # 청크 생성
            chunk = DocumentChunk(
                content=section_content,
                level=header['level'],
                title=header['title'],
                full_path=header['path'],
                metadata={
                    'level': header['level'],
                    'section_id': f"section_{i}",
                    'parent_titles': header['path'][:-1],
                    'char_count': len(section_content),
                    'line_range': f"{start_line}-{end_line}"
                },
                start_line=start_line,
                end_line=end_line
            )
            
            chunks.append(chunk)
            
            # 하위 레벨별로도 청크 생성 (계층적 접근)
            sub_chunks = self._create_hierarchical_chunks(header, section_content, start_line)
            chunks.extend(sub_chunks)
        
        return chunks
    
    def _extract_headers(self, text: str) -> List[Dict]:
        """텍스트에서 헤더 정보를 추출합니다."""
        lines = text.split('\n')
        headers = []
        path_stack: List[str] = []
        
        for line_num, line in enumerate(lines):
            match = self.header_pattern.match(line)
            if match:
                level = len(match.group(1))  # # 개수
                title = match.group(2).strip()
                
                # 경로 스택 관리
                while len(path_stack) >= level:
                    path_stack.pop()
                
                path_stack.append(title)
                
                headers.append({
                    'level': level,
                    'title': title,
                    'path': path_stack.copy(),
                    'line_num': line_num
                })
        
        return headers
    
    def _get_parent_context(self, current_header: Dict, all_headers: List[Dict]) -> str:
        """현재 헤더의 상위 컨텍스트를 가져옵니다."""
        parent_context = []
        
        # 상위 레벨 헤더들의 제목을 컨텍스트로 추가
        for parent_title in current_header['path'][:-1]:
            parent_context.append(f"상위 섹션: {parent_title}")
        
        return '\n'.join(parent_context) if parent_context else ""
    
    def _create_hierarchical_chunks(self, header: Dict, content: str, start_line: int) -> List[DocumentChunk]:
        """계층적으로 청크를 생성합니다."""
        chunks = []
        
        # 긴 내용을 더 작은 청크로 분할
        if len(content) > self.max_chunk_size:
            sub_chunks = self._split_long_content(content, header, start_line)
            chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_long_content(self, content: str, header: Dict, start_line: int) -> List[DocumentChunk]:
        """긴 내용을 작은 청크로 분할합니다."""
        chunks = []
        paragraphs = content.split('\n\n')
        current_chunk = ""
        chunk_count = 0
        
        for para in paragraphs:
            if len(current_chunk + para) > self.max_chunk_size and current_chunk:
                # 현재 청크를 저장
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    level=header['level'] + 1,  # 하위 레벨로 설정
                    title=f"{header['title']} (Part {chunk_count + 1})",
                    full_path=header['path'] + [f"Part {chunk_count + 1}"],
                    metadata={
                        'level': header['level'] + 1,
                        'section_id': f"section_{header['title']}_{chunk_count}",
                        'parent_titles': header['path'],
                        'char_count': len(current_chunk),
                        'is_sub_chunk': True
                    },
                    start_line=start_line,
                    end_line=start_line
                )
                chunks.append(chunk)
                current_chunk = para
                chunk_count += 1
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # 마지막 청크 추가
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                level=header['level'] + 1,
                title=f"{header['title']} (Part {chunk_count + 1})",
                full_path=header['path'] + [f"Part {chunk_count + 1}"],
                metadata={
                    'level': header['level'] + 1,
                    'section_id': f"section_{header['title']}_{chunk_count}",
                    'parent_titles': header['path'],
                    'char_count': len(current_chunk),
                    'is_sub_chunk': True
                },
                start_line=start_line,
                end_line=start_line
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_chunks_by_level(self, chunks: List[DocumentChunk], level: int) -> List[DocumentChunk]:
        """특정 레벨의 청크들만 반환합니다."""
        return [chunk for chunk in chunks if chunk.level == level]
    
    def find_relevant_chunks(self, chunks: List[DocumentChunk], query: str) -> List[DocumentChunk]:
        """쿼리와 관련된 청크들을 찾습니다."""
        relevant_chunks = []
        query_lower = query.lower()
        
        for chunk in chunks:
            # 제목이나 내용에 쿼리 키워드가 포함된 청크 찾기
            if (query_lower in chunk.title.lower() or 
                query_lower in chunk.content.lower()):
                relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def export_chunks_to_json(self, chunks: List[DocumentChunk], filename: str):
        """청크들을 JSON 파일로 내보냅니다."""
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                'content': chunk.content,
                'level': chunk.level,
                'title': chunk.title,
                'full_path': chunk.full_path,
                'metadata': chunk.metadata,
                'start_line': chunk.start_line,
                'end_line': chunk.end_line
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    def print_chunk_summary(self, chunks: List[DocumentChunk]):
        """청크들의 요약 정보를 출력합니다."""
        print(f"총 {len(chunks)}개의 청크가 생성되었습니다.\n")
        
        for level in range(1, 5):
            level_chunks = self.get_chunks_by_level(chunks, level)
            if level_chunks:
                print(f"레벨 {level} 헤더: {len(level_chunks)}개")
                for chunk in level_chunks[:3]:  # 처음 3개만 표시
                    print(f"  - {chunk.title} ({len(chunk.content)}자)")
                if len(level_chunks) > 3:
                    print(f"  ... 및 {len(level_chunks) - 3}개 더")
                print()


def demo_pns_query_test(chunks: List[DocumentChunk]):
    """PNS purchaseState 질의에 대한 테스트"""
    print("=== PNS purchaseState 질의 테스트 ===\n")
    
    query = "purchaseState"
    splitter = HierarchicalDocumentSplitter()
    relevant_chunks = splitter.find_relevant_chunks(chunks, query)
    
    print(f"'{query}' 관련 청크 {len(relevant_chunks)}개 발견:\n")
    
    # 답변 생성을 위한 정보 수집
    answer_found = False
    purchase_state_info = []
    
    for i, chunk in enumerate(relevant_chunks):
        print(f"청크 {i+1}:")
        print(f"  제목: {chunk.title}")
        print(f"  경로: {' > '.join(chunk.full_path)}")
        print(f"  레벨: {chunk.level}")
        
        # purchaseState 관련 내용 추출 (개선된 로직)
        lines = chunk.content.split('\n')
        found_purchase_state = False
        
        for line in lines:
            if 'purchaseState' in line.lower():
                print(f"  관련 내용: {line.strip()}")
                found_purchase_state = True
                
                # 테이블 형태나 설명 형태에서 COMPLETED/CANCELED 추출
                if 'COMPLETED' in line and 'CANCELED' in line:
                    # 테이블 형태: | purchaseState | String | COMPLETED : 결제완료 / CANCELED : 취소 |
                    if '|' in line:
                        parts = line.split('|')
                        for part in parts:
                            if 'COMPLETED' in part and 'CANCELED' in part:
                                purchase_state_info.append(part.strip())
                                answer_found = True
                    else:
                        purchase_state_info.append(line.strip())
                        answer_found = True
        
        if found_purchase_state:
            print()
    
    # 개선된 답변 생성
    if answer_found:
        print("✅ 예상 답변:")
        print("PNS의 purchaseState는 다음과 같은 값이 있습니다:")
        
        # COMPLETED와 CANCELED 값 추출
        for info in purchase_state_info:
            if 'COMPLETED' in info and 'CANCELED' in info:
                print("  - COMPLETED : 결제완료")
                print("  - CANCELED : 취소")
                break
        else:
            for info in purchase_state_info:
                print(f"  - {info}")
    else:
        print("❌ 답변을 생성할 수 없습니다.")
        
        # 디버깅을 위한 추가 검색
        print("\n🔍 디버깅: 더 넓은 범위에서 검색...")
        pns_chunks = [chunk for chunk in chunks if 'pns' in chunk.title.lower() or 'payment notification' in chunk.title.lower()]
        
        for chunk in pns_chunks:
            lines = chunk.content.split('\n')
            for line in lines:
                if 'purchaseState' in line.lower() and ('COMPLETED' in line or 'CANCELED' in line):
                    print(f"🎯 발견: {line.strip()}")
                    print(f"   청크: {chunk.title}")
                    answer_found = True
        
        if not answer_found:
            print("   추가 검색에서도 찾을 수 없습니다.")
