# Enhanced RAG Pipeline for Long Documents and Complex Concept Systems
# - Hierarchical document analysis
# - Concept mapping and categorization  
# - Multi-stage retrieval strategy
# - Context window expansion
# - Overview-first approach for comprehensive understanding

import os, re, json
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import networkx as nx

from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    HAS_MD_SPLITTER = True
except Exception:
    HAS_MD_SPLITTER = False

# Configuration
DATA_FILE = "../data/dev_center_guide_allmd_touched.md"
EMBED_MODEL = "bge-m3:latest"
LLM_MODEL = "exaone3.5:latest"
FAISS_DIR = "../models/faiss_rag_enhanced_bge-m3_hierarchical"

print("Enhanced RAG Config:")
print(f"- DATA_FILE: {DATA_FILE}")
print(f"- EMBED_MODEL: {EMBED_MODEL}")
print(f"- LLM_MODEL: {LLM_MODEL}")
print(f"- FAISS_DIR:  {FAISS_DIR}")

# ============================================================================
# Hierarchical Document Analysis
# ============================================================================

@dataclass
class DocumentNode:
    """계층적 문서 구조를 표현하는 노드"""
    id: str
    title: str
    content: str
    level: int
    parent_id: Optional[str] = None
    children: List[str] = None
    metadata: Dict[str, Any] = None
    concept_type: str = "section"  # section, concept, overview, detail
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}

class HierarchicalDocumentAnalyzer:
    """긴 문서의 계층적 구조를 분석하고 개념을 매핑하는 클래스"""
    
    def __init__(self):
        self.concept_patterns = {
            'numbered_concept': re.compile(r'([A-Z]{2,})_(\d+)'),  # PNS_1, API_2 등
            'versioned_concept': re.compile(r'([A-Z]{2,})\s*V?(\d+(?:\.\d+)*)'),  # PNS V1, API V2.1 등
            'sectioned_concept': re.compile(r'(\d+)\.\s*([A-Z][A-Za-z\s]+)'),  # 1. PNS 개요, 2. API 설명 등
        }
        self.overview_keywords = {
            '개요', 'overview', 'introduction', '개념', 'concept', '정의', 'definition',
            '설명', 'description', '소개', 'introduction', '기본', 'basic', 'fundamental'
        }
        self.detail_keywords = {
            '상세', 'detail', '구체', 'specific', '예시', 'example', '사용법', 'usage',
            '구현', 'implementation', '코드', 'code', '매개변수', 'parameter'
        }
    
    def analyze_document_structure(self, docs: List[Document]) -> Dict[str, DocumentNode]:
        """문서들의 계층적 구조를 분석하여 노드 트리 생성"""
        nodes = {}
        
        for doc in docs:
            metadata = doc.metadata or {}
            full_path = metadata.get('full_path', [])
            
            if not full_path:
                continue
                
            # 노드 ID 생성
            node_id = "_".join(full_path)
            
            # 레벨 결정 (헤더 깊이)
            level = len(full_path)
            
            # 부모 노드 찾기
            parent_id = None
            if level > 1:
                parent_path = full_path[:-1]
                parent_id = "_".join(parent_path)
            
            # 개념 타입 결정
            concept_type = self._determine_concept_type(full_path[-1], doc.page_content)
            
            # 노드 생성
            node = DocumentNode(
                id=node_id,
                title=full_path[-1],
                content=doc.page_content,
                level=level,
                parent_id=parent_id,
                concept_type=concept_type,
                metadata=metadata
            )
            
            nodes[node_id] = node
        
        # 부모-자식 관계 설정
        for node_id, node in nodes.items():
            if node.parent_id and node.parent_id in nodes:
                nodes[node.parent_id].children.append(node_id)
        
        return nodes
    
    def _determine_concept_type(self, title: str, content: str) -> str:
        """제목과 내용을 기반으로 개념 타입 결정"""
        title_lower = title.lower()
        content_lower = content.lower()
        
        # 개요 타입 확인
        if any(keyword in title_lower for keyword in self.overview_keywords):
            return "overview"
        
        # 상세 타입 확인
        if any(keyword in title_lower for keyword in self.detail_keywords):
            return "detail"
        
        # 번호가 있는 개념 확인
        if self.concept_patterns['numbered_concept'].search(title):
            return "concept"
        
        # 버전이 있는 개념 확인
        if self.concept_patterns['versioned_concept'].search(title):
            return "concept"
        
        return "section"
    
    def extract_concept_hierarchy(self, nodes: Dict[str, DocumentNode]) -> Dict[str, List[str]]:
        """개념 계층 구조 추출 (예: PNS_1, PNS_2, ...)"""
        concept_groups = defaultdict(list)
        
        for node_id, node in nodes.items():
            # 번호가 있는 개념 찾기
            match = self.concept_patterns['numbered_concept'].search(node.title)
            if match:
                base_concept = match.group(1)  # PNS
                concept_groups[base_concept].append(node_id)
        
        # 각 그룹 내에서 번호 순으로 정렬
        for base_concept in concept_groups:
            concept_groups[base_concept].sort(
                key=lambda x: int(self.concept_patterns['numbered_concept'].search(nodes[x].title).group(2))
            )
        
        return dict(concept_groups)
    
    def find_overview_sections(self, nodes: Dict[str, DocumentNode]) -> List[str]:
        """개요 섹션들 찾기"""
        overview_nodes = []
        for node_id, node in nodes.items():
            if node.concept_type == "overview":
                overview_nodes.append(node_id)
        return overview_nodes
    
    def get_related_sections(self, nodes: Dict[str, DocumentNode], target_node_id: str, max_distance: int = 2) -> List[str]:
        """관련 섹션들을 찾기 (계층적 거리 기반)"""
        target_node = nodes.get(target_node_id)
        if not target_node:
            return []
        
        related = []
        for node_id, node in nodes.items():
            if node_id == target_node_id:
                continue
            
            # 계층적 거리 계산
            distance = self._calculate_hierarchical_distance(nodes, target_node_id, node_id)
            if distance <= max_distance:
                related.append(node_id)
        
        return related
    
    def _calculate_hierarchical_distance(self, nodes: Dict[str, DocumentNode], node1_id: str, node2_id: str) -> int:
        """두 노드 간의 계층적 거리 계산"""
        node1 = nodes.get(node1_id)
        node2 = nodes.get(node2_id)
        
        if not node1 or not node2:
            return float('inf')
        
        # 공통 조상 찾기
        ancestors1 = self._get_ancestors(nodes, node1_id)
        ancestors2 = self._get_ancestors(nodes, node2_id)
        
        # 공통 조상이 있으면 거리 계산
        for ancestor in ancestors1:
            if ancestor in ancestors2:
                return len(ancestors1) + len(ancestors2) - 2 * len([a for a in ancestors1 if a in ancestors2])
        
        # 공통 조상이 없으면 레벨 차이
        return abs(node1.level - node2.level)
    
    def _get_ancestors(self, nodes: Dict[str, DocumentNode], node_id: str) -> List[str]:
        """노드의 모든 조상들 찾기"""
        ancestors = []
        current = nodes.get(node_id)
        
        while current and current.parent_id:
            ancestors.append(current.parent_id)
            current = nodes.get(current.parent_id)
        
        return ancestors

# ============================================================================
# Multi-Stage Retrieval Strategy
# ============================================================================

class MultiStageRetriever:
    """다단계 검색 전략: 개요 → 개념 → 세부사항"""
    
    def __init__(self, analyzer: HierarchicalDocumentAnalyzer, embeddings, docs: List[Document]):
        self.analyzer = analyzer
        self.embeddings = embeddings
        self.docs = docs
        
        # 문서 구조 분석
        self.nodes = analyzer.analyze_document_structure(docs)
        self.concept_hierarchy = analyzer.extract_concept_hierarchy(self.nodes)
        self.overview_sections = analyzer.find_overview_sections(self.nodes)
        
        # 벡터 스토어 생성
        self.vectorstore = FAISS.from_documents(docs, embeddings)
        
        print(f"📊 문서 구조 분석 완료:")
        print(f"   - 총 노드: {len(self.nodes)}")
        print(f"   - 개념 그룹: {len(self.concept_hierarchy)}")
        print(f"   - 개요 섹션: {len(self.overview_sections)}")
    
    def retrieve_comprehensive_context(self, query: str, top_k: int = 15) -> List[Document]:
        """전체적인 맥락을 파악하기 위한 다단계 검색"""
        
        # 1단계: 개요 섹션에서 검색
        overview_docs = self._retrieve_overview_context(query)
        
        # 2단계: 관련 개념들 찾기
        concept_docs = self._retrieve_concept_context(query)
        
        # 3단계: 세부사항 검색
        detail_docs = self._retrieve_detail_context(query, top_k // 2)
        
        # 결과 통합 및 중복 제거
        all_docs = overview_docs + concept_docs + detail_docs
        unique_docs = self._deduplicate_docs(all_docs)
        
        return unique_docs[:top_k]
    
    def _retrieve_overview_context(self, query: str) -> List[Document]:
        """개요 섹션에서 검색"""
        if not self.overview_sections:
            return []
        
        overview_content = []
        for node_id in self.overview_sections:
            node = self.nodes[node_id]
            overview_content.append(node.content)
        
        # 개요 내용을 하나의 문서로 결합
        combined_overview = "\n\n".join(overview_content)
        overview_doc = Document(
            page_content=combined_overview,
            metadata={'type': 'overview_summary', 'sections': self.overview_sections}
        )
        
        return [overview_doc]
    
    def _retrieve_concept_context(self, query: str) -> List[Document]:
        """관련 개념들 찾기"""
        concept_docs = []
        
        # 쿼리에서 개념 키워드 추출
        query_keywords = self._extract_concept_keywords(query)
        
        for base_concept, concept_nodes in self.concept_hierarchy.items():
            if any(keyword.lower() in base_concept.lower() for keyword in query_keywords):
                # 해당 개념 그룹의 모든 노드 포함
                for node_id in concept_nodes:
                    node = self.nodes[node_id]
                    concept_docs.append(Document(
                        page_content=node.content,
                        metadata={'type': 'concept_group', 'concept': base_concept, 'node_id': node_id}
                    ))
        
        return concept_docs
    
    def _retrieve_detail_context(self, query: str, top_k: int) -> List[Document]:
        """세부사항 검색"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k * 2})
        docs = retriever.invoke(query)
        return docs[:top_k]
    
    def _extract_concept_keywords(self, query: str) -> List[str]:
        """쿼리에서 개념 키워드 추출"""
        keywords = []
        
        # 대문자 약어 패턴
        acronym_pattern = re.compile(r'\b[A-Z]{2,}\b')
        acronyms = acronym_pattern.findall(query)
        keywords.extend(acronyms)
        
        # 일반적인 기술 용어
        tech_terms = ['API', 'SDK', 'PNS', 'IAP', '결제', '인앱', '알림', '서비스']
        for term in tech_terms:
            if term.lower() in query.lower():
                keywords.append(term)
        
        return keywords
    
    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """중복 문서 제거"""
        seen = set()
        unique_docs = []
        
        for doc in docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs

# ============================================================================
# Enhanced Context Window Expansion
# ============================================================================

class ContextWindowExpander:
    """컨텍스트 윈도우 확장을 위한 클래스"""
    
    def __init__(self, analyzer: HierarchicalDocumentAnalyzer):
        self.analyzer = analyzer
    
    def expand_context(self, docs: List[Document], nodes: Dict[str, DocumentNode], 
                      expansion_level: int = 1) -> List[Document]:
        """검색된 문서들의 컨텍스트를 확장"""
        expanded_docs = []
        
        for doc in docs:
            expanded_docs.append(doc)
            
            # 문서의 노드 ID 찾기
            node_id = self._find_node_id_for_doc(doc, nodes)
            if not node_id:
                continue
            
            # 관련 섹션들 찾기
            related_sections = self.analyzer.get_related_sections(
                nodes, node_id, max_distance=expansion_level
            )
            
            # 관련 섹션들의 내용 추가
            for related_id in related_sections:
                related_node = nodes[related_id]
                related_doc = Document(
                    page_content=related_node.content,
                    metadata={'type': 'expanded_context', 'original_node': node_id, 'related_node': related_id}
                )
                expanded_docs.append(related_doc)
        
        return expanded_docs
    
    def _find_node_id_for_doc(self, doc: Document, nodes: Dict[str, DocumentNode]) -> Optional[str]:
        """문서에 해당하는 노드 ID 찾기"""
        doc_content = doc.page_content[:100]  # 첫 100자만 비교
        
        for node_id, node in nodes.items():
            if doc_content in node.content[:200]:
                return node_id
        
        return None

# ============================================================================
# Enhanced Prompt Templates
# ============================================================================

COMPREHENSIVE_OVERVIEW_PROMPT = PromptTemplate.from_template(
    """당신은 긴 기술 문서의 전체적인 맥락을 파악하는 전문가입니다.

주어진 컨텍스트를 분석하여 다음을 수행하세요:

1. **전체 구조 파악**: 문서의 주요 섹션들과 그들의 관계를 파악
2. **핵심 개념 정리**: 주요 개념들(PNS_1, PNS_2 등)을 식별하고 분류
3. **기능 요약**: 요청된 기능의 전반적인 개요 제공
4. **관련 섹션 안내**: 더 자세한 정보가 필요한 섹션들 제시

질문: {question}

컨텍스트:
{context}

**분석 결과:**

### 📋 전체 구조
(문서의 주요 섹션들과 계층 구조)

### 🎯 핵심 개념
(식별된 주요 개념들과 그들의 역할)

### 📖 기능 요약
(요청된 기능의 전반적인 설명)

### 🔍 추가 정보
(더 자세한 정보가 필요한 섹션들)

답변:"""
)

DETAILED_ANALYSIS_PROMPT = PromptTemplate.from_template(
    """당신은 기술 문서의 세부사항을 정확히 분석하는 전문가입니다.

주어진 컨텍스트에서 다음을 수행하세요:

1. **정확한 값 추출**: 필드값, 코드, 매개변수 등을 정확히 나열
2. **구체적 예시 제공**: 실제 사용 예시나 코드 샘플 제시
3. **관련 정보 연결**: 다른 섹션과의 연관성 설명
4. **실용적 가이드**: 실제 구현 시 주의사항이나 팁 제공

질문: {question}

컨텍스트:
{context}

**상세 분석:**

### 📊 정확한 값들
(필드값, 코드, 상수 등)

### 💡 구체적 예시
(실제 사용 예시)

### 🔗 관련 정보
(다른 섹션과의 연관성)

### ⚠️ 주의사항
(구현 시 고려사항)

답변:"""
)

# ============================================================================
# Main Enhanced RAG System
# ============================================================================

class EnhancedRAGSystem:
    """개선된 RAG 시스템 - 긴 문서와 복잡한 개념 체계 처리"""
    
    def __init__(self, data_file: str, embed_model: str, llm_model: str):
        self.data_file = data_file
        self.embeddings = OllamaEmbeddings(model=embed_model)
        self.llm = ChatOllama(model=llm_model)
        self.parser = StrOutputParser()
        
        # 문서 로드 및 분석
        self.docs = self._load_and_process_documents()
        self.analyzer = HierarchicalDocumentAnalyzer()
        self.multi_stage_retriever = MultiStageRetriever(self.analyzer, self.embeddings, self.docs)
        self.context_expander = ContextWindowExpander(self.analyzer)
        
        print("🚀 Enhanced RAG System 초기화 완료!")
    
    def _load_and_process_documents(self) -> List[Document]:
        """문서 로드 및 전처리"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            md_text = f.read()
        
        # 마크다운 헤더 기반 분할
        if HAS_MD_SPLITTER:
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
                ("#", "h1"), ("##","h2"), ("###","h3"), ("####","h4")
            ])
            splits = splitter.split_text(md_text)
            docs = []
            for s in splits:
                path = []
                for k in ['h1','h2','h3','h4']:
                    if s.metadata.get(k):
                        path.append(s.metadata[k])
                full_path = " > ".join(path) if path else ""
                content = (f"[{full_path}] " if full_path else "") + s.page_content
                docs.append(Document(page_content=content, metadata={'full_path': path, 'type': 'section'}))
        else:
            # 기본 분할
            parts = re.split(r'(?m)^#{1,4}\s+', md_text)
            docs = []
            for p in parts:
                p = p.strip()
                if not p: continue
                first_line = p.splitlines()[0] if '\n' in p else p
                content = f"[{first_line}] " + p
                docs.append(Document(page_content=content, metadata={'full_path': [first_line], 'type': 'section'}))
        
        return docs
    
    def answer_comprehensive(self, query: str, include_overview: bool = True) -> str:
        """전체적인 맥락을 고려한 종합적 답변"""
        
        # 1. 다단계 검색으로 관련 문서 수집
        relevant_docs = self.multi_stage_retriever.retrieve_comprehensive_context(query, top_k=20)
        
        # 2. 컨텍스트 확장
        expanded_docs = self.context_expander.expand_context(
            relevant_docs, self.multi_stage_retriever.nodes, expansion_level=1
        )
        
        # 3. 컨텍스트 포맷팅
        context = self._format_context(expanded_docs)
        
        # 4. 적절한 프롬프트 선택
        if include_overview or "전반적" in query or "개요" in query or "요약" in query:
            prompt = COMPREHENSIVE_OVERVIEW_PROMPT
        else:
            prompt = DETAILED_ANALYSIS_PROMPT
        
        # 5. LLM 호출
        chain = prompt | self.llm | self.parser
        response = chain.invoke({"question": query, "context": context})
        
        return response
    
    def _format_context(self, docs: List[Document]) -> str:
        """컨텍스트 포맷팅"""
        parts = []
        for i, d in enumerate(docs, 1):
            md = d.metadata or {}
            doc_type = md.get('type', 'section')
            concept = md.get('concept', '')
            
            prefix = f"[{doc_type}"
            if concept:
                prefix += f":{concept}"
            prefix += "] "
            
            parts.append(f"({i}) {prefix}{d.page_content}")
        
        return "\n\n".join(parts)

# ============================================================================
# Test the Enhanced System
# ============================================================================

if __name__ == "__main__":
    print("🧪 Enhanced RAG System 테스트")
    print("="*80)
    
    # 시스템 초기화
    enhanced_system = EnhancedRAGSystem(DATA_FILE, EMBED_MODEL, LLM_MODEL)
    
    # 테스트 쿼리들
    test_queries = [
        "PNS의 전반적인 기능을 요약해주세요",
        "PNS_1, PNS_2 등의 세부 개념들이 무엇인지 알려주세요",
        "원스토어 인앱결제의 전체적인 구조를 설명해주세요"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 테스트 {i}: {query}")
        print("-" * 60)
        
        try:
            answer = enhanced_system.answer_comprehensive(query, include_overview=True)
            print(f"답변:\n{answer}")
        except Exception as e:
            print(f"오류 발생: {e}")
        
        print("\n" + "="*80)
