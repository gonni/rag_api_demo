"""
PNS 문서 전용 계층적 분할 전략

이 모듈은 PNS 관련 문서를 효과적으로 분할하여 
컨텍스트 손실을 최소화하는 전략을 제공합니다.
"""

import re
from typing import List, Dict, Any, Tuple
from langchain.docstore.document import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class PNSHierarchicalSplitter:
    """PNS 문서 전용 계층적 분할기"""
    
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.raw_text = self._load_document()
        
    def _load_document(self) -> str:
        """문서 로드"""
        with open(self.document_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def split_documents(self) -> List[Document]:
        """PNS 최적화 문서 분할"""
        print("🚀 PNS 계층적 문서 분할 시작...")
        
        # 1단계: PNS 섹션 식별 및 추출
        pns_sections = self._extract_pns_sections()
        print(f"📋 PNS 섹션 수: {len(pns_sections)}")
        
        # 2단계: 각 섹션별 최적화된 분할
        all_documents = []
        for section_name, section_content in pns_sections.items():
            section_docs = self._split_pns_section(section_name, section_content)
            all_documents.extend(section_docs)
            print(f"  {section_name}: {len(section_docs)}개 청크")
        
        print(f"✅ 총 {len(all_documents)}개 PNS 최적화 문서 생성")
        return all_documents
    
    def _extract_pns_sections(self) -> Dict[str, str]:
        """PNS 관련 섹션 추출"""
        sections = {}
        
        # PNS 관련 헤더 패턴
        pns_patterns = [
            r'(### PNS Payment Notification 메시지 발송 규격.*?)(?=###|\Z)',
            r'(### PNS Subscription Notification 메시지 발송 규격.*?)(?=###|\Z)',
            r'(## PNS.*?)(?=##|\Z)',
            r'(#.*?PNS.*?)(?=#|\Z)'
        ]
        
        for pattern in pns_patterns:
            matches = re.findall(pattern, self.raw_text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                # 섹션 이름 추출
                lines = match.strip().split('\n')
                section_name = lines[0].strip()
                if section_name.startswith('#'):
                    section_name = section_name.lstrip('#').strip()
                
                sections[section_name] = match.strip()
        
        return sections
    
    def _split_pns_section(self, section_name: str, section_content: str) -> List[Document]:
        """PNS 섹션별 최적화된 분할"""
        documents = []
        
        # 1. 메시지 규격 테이블 식별
        table_sections = self._extract_message_specifications(section_content)
        
        # 2. 각 테이블 섹션을 하나의 완전한 문서로 생성
        for table_name, table_content in table_sections.items():
            # 컨텍스트 강화
            enhanced_content = self._enhance_pns_context(section_name, table_name, table_content)
            
            # 완전한 메시지 규격을 하나의 문서로 유지
            metadata = {
                'section_name': section_name,
                'table_name': table_name,
                'content_type': 'pns_message_specification',
                'contains_pns': True,
                'contains_purchasestate': 'purchasestate' in table_content.lower(),
                'is_complete_spec': True,
                'chunk_size': len(enhanced_content)
            }
            
            documents.append(Document(
                page_content=enhanced_content,
                metadata=metadata
            ))
        
        # 3. 나머지 내용도 적절히 분할
        remaining_content = self._extract_remaining_content(section_content, table_sections)
        if remaining_content:
            remaining_docs = self._split_remaining_content(section_name, remaining_content)
            documents.extend(remaining_docs)
        
        return documents
    
    def _extract_message_specifications(self, content: str) -> Dict[str, str]:
        """메시지 규격 테이블 추출"""
        specifications = {}
        
        # 테이블 패턴 찾기
        table_patterns = [
            r'(\|.*?Element Name.*?Description.*?\|.*?\|.*?\|.*?\|.*?)(?=\n\n|\Z)',
            r'(\|.*?Parameter Name.*?Data Type.*?Description.*?\|.*?\|.*?\|.*?\|.*?)(?=\n\n|\Z)'
        ]
        
        for pattern in table_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for i, match in enumerate(matches):
                # 테이블 제목 찾기
                lines_before = content[:content.find(match)].split('\n')
                title = ""
                for line in reversed(lines_before[-10:]):  # 최근 10줄에서 제목 찾기
                    if line.strip() and not line.startswith('|') and not line.startswith('-'):
                        title = line.strip()
                        break
                
                if not title:
                    title = f"Message Specification {i+1}"
                
                specifications[title] = match.strip()
        
        return specifications
    
    def _enhance_pns_context(self, section_name: str, table_name: str, table_content: str) -> str:
        """PNS 컨텍스트 강화"""
        enhanced = f"[PNS 섹션]: {section_name}\n"
        enhanced += f"[메시지 규격]: {table_name}\n"
        enhanced += f"[설명]: 이 내용은 PNS(Payment Notification Service) 결제알림서비스의 메시지 규격입니다.\n\n"
        enhanced += table_content
        
        # purchaseState 관련 정보가 있으면 강조
        if 'purchasestate' in table_content.lower():
            enhanced += "\n\n[중요]: 이 메시지에는 purchaseState 필드가 포함되어 있어 결제 상태를 확인할 수 있습니다."
        
        return enhanced
    
    def _extract_remaining_content(self, content: str, table_sections: Dict[str, str]) -> str:
        """테이블 외 나머지 내용 추출"""
        remaining = content
        
        # 테이블 내용 제거
        for table_content in table_sections.values():
            remaining = remaining.replace(table_content, '')
        
        # 빈 줄 정리
        remaining = re.sub(r'\n\s*\n\s*\n', '\n\n', remaining)
        return remaining.strip()
    
    def _split_remaining_content(self, section_name: str, content: str) -> List[Document]:
        """나머지 내용 분할"""
        if not content or len(content) < 100:
            return []
        
        # 적절한 크기로 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "? ", "! ", ", "]
        )
        
        chunks = splitter.split_text(content)
        documents = []
        
        for i, chunk in enumerate(chunks):
            metadata = {
                'section_name': section_name,
                'content_type': 'pns_related_content',
                'chunk_index': i,
                'contains_pns': True,
                'contains_purchasestate': 'purchasestate' in chunk.lower(),
                'chunk_size': len(chunk)
            }
            
            documents.append(Document(
                page_content=chunk,
                metadata=metadata
            ))
        
        return documents


class PNSContextualRetriever:
    """PNS 컨텍스트 인식 검색기"""
    
    def __init__(self, documents: List[Document], embedding_model_name: str = "bge-m3:latest"):
        self.documents = documents
        self.embedding_model_name = embedding_model_name
        self.vector_store = None
        self.bm25_retriever = None
        
    def build_retrievers(self):
        """검색기 구축"""
        print(f"🔧 PNS 컨텍스트 검색기 구축 중... (문서 수: {len(self.documents)})")
        
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_community.retrievers import BM25Retriever
        
        # Vector store 구축
        embeddings = OllamaEmbeddings(model=self.embedding_model_name)
        self.vector_store = FAISS.from_documents(self.documents, embeddings)
        
        # BM25 검색기 구축
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 30
        
        print("✅ PNS 컨텍스트 검색기 구축 완료")
    
    def retrieve_pns_context(self, query: str, k: int = 10) -> List[Document]:
        """PNS 컨텍스트 인식 검색"""
        if not self.vector_store or not self.bm25_retriever:
            raise ValueError("검색기가 구축되지 않았습니다.")
        
        # 1. 완전한 메시지 규격 우선 검색
        complete_specs = self._find_complete_specifications(query)
        
        # 2. 벡터 검색으로 관련 문서 찾기
        vector_results = self.vector_store.similarity_search_with_score(query, k=20)
        
        # 3. BM25 검색으로 키워드 매칭
        bm25_results = self.bm25_retriever.get_relevant_documents(query)[:20]
        
        # 4. 결과 통합 및 우선순위 적용
        all_candidates = self._merge_and_prioritize(
            complete_specs, vector_results, bm25_results, query
        )
        
        return [doc for score, doc in all_candidates[:k]]
    
    def _find_complete_specifications(self, query: str) -> List[Tuple[float, Document]]:
        """완전한 메시지 규격 찾기"""
        complete_specs = []
        
        for doc in self.documents:
            if doc.metadata.get('is_complete_spec', False):
                score = self._calculate_spec_relevance(query, doc)
                if score > 0:
                    complete_specs.append((score, doc))
        
        return sorted(complete_specs, key=lambda x: x[0], reverse=True)
    
    def _calculate_spec_relevance(self, query: str, doc: Document) -> float:
        """메시지 규격 관련성 점수 계산"""
        score = 0.0
        query_lower = query.lower()
        content_lower = doc.page_content.lower()
        
        # PNS 관련 키워드
        pns_keywords = ['pns', 'payment notification', '메시지', '규격']
        for keyword in pns_keywords:
            if keyword in query_lower:
                score += 10
        
        # purchaseState 관련
        if 'purchasestate' in query_lower and 'purchasestate' in content_lower:
            score += 15
        
        # 메시지 타입 관련
        if 'message' in query_lower or '메시지' in query:
            score += 8
        
        return score
    
    def _merge_and_prioritize(self, complete_specs, vector_results, bm25_results, query: str) -> List[Tuple[float, Document]]:
        """결과 통합 및 우선순위 적용"""
        all_candidates = {}
        
        # 완전한 메시지 규격 (최우선)
        for score, doc in complete_specs:
            all_candidates[doc.page_content] = (score * 2, doc)  # 가중치 2배
        
        # 벡터 검색 결과
        for doc, score in vector_results:
            if doc.page_content not in all_candidates:
                all_candidates[doc.page_content] = (score, doc)
        
        # BM25 검색 결과
        for doc in bm25_results:
            if doc.page_content not in all_candidates:
                all_candidates[doc.page_content] = (0.5, doc)  # 기본 점수
        
        # 점수순 정렬
        return sorted(all_candidates.values(), key=lambda x: x[0], reverse=True)
