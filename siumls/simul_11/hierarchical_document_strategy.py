"""
계층적 문서 분할 전략 - PNS 대제목과 하위 섹션 연결 문제 해결

이 모듈은 다음과 같은 전략들을 구현합니다:
1. 계층적 제목 포함 전략 (Title Hierarchy Inclusion)
2. 다중 레벨 분할 전략 (Multi-Level Splitting)
3. 컨텍스트 상속 전략 (Context Inheritance)
"""

import os
import re
import pickle
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


@dataclass
class HierarchicalSearchResult:
    """계층적 검색 결과 분석을 위한 데이터 클래스"""
    strategy_name: str
    query: str
    documents: List[Document]
    hierarchy_scores: List[float]
    context_scores: List[float]
    total_docs: int
    relevant_docs: int
    pns_related_docs: int


class HierarchicalTitleStrategy:
    """계층적 제목 포함 전략"""
    
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.raw_text = self._load_document()
        
    def _load_document(self) -> str:
        with open(self.document_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def split_documents(self) -> List[Document]:
        """계층적 제목을 포함하여 문서 분할"""
        documents = []
        
        # 헤더 기반 분할
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4")
            ]
        )
        header_docs = header_splitter.split_text(self.raw_text)
        
        for doc in header_docs:
            # 계층적 제목 경로 생성
            title_hierarchy = self._build_title_hierarchy(doc.metadata)
            
            # 원본 내용에 제목 계층 추가
            enhanced_content = self._add_title_context(doc.page_content, title_hierarchy)
            
            # 적절한 크기로 청킹
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "? ", "! ", ", "]
            )
            chunks = text_splitter.split_text(enhanced_content)
            
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata.update({
                    "chunk_index": i,
                    "title_hierarchy": title_hierarchy,
                    "source_strategy": "hierarchical_title",
                    "hierarchy_level": len([h for h in title_hierarchy.split(" > ") if h.strip()]),
                    "contains_pns": "PNS" in title_hierarchy.upper() or "PAYMENT NOTIFICATION" in title_hierarchy.upper()
                })
                documents.append(Document(page_content=chunk, metadata=metadata))
        
        return documents
    
    def _build_title_hierarchy(self, metadata: Dict) -> str:
        """메타데이터에서 제목 계층 구조 생성"""
        hierarchy_parts = []
        
        # 헤더 레벨별로 제목 수집
        for level in range(1, 5):
            header_key = f"Header {level}"
            if header_key in metadata and metadata[header_key]:
                title = metadata[header_key].strip()
                if title:
                    hierarchy_parts.append(title)
        
        return " > ".join(hierarchy_parts) if hierarchy_parts else "Unknown Section"
    
    def _add_title_context(self, content: str, title_hierarchy: str) -> str:
        """내용에 제목 계층 컨텍스트 추가"""
        context_header = f"[섹션 경로]: {title_hierarchy}\n\n"
        return context_header + content


class MultiLevelSplittingStrategy:
    """다중 레벨 분할 전략 - 큰 단위 + 작은 단위 문서 생성"""
    
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.raw_text = self._load_document()
        
    def _load_document(self) -> str:
        with open(self.document_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def split_documents(self) -> List[Document]:
        """다중 레벨로 문서 분할"""
        documents = []
        
        # 헤더 기반 분할
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4")
            ]
        )
        header_docs = header_splitter.split_text(self.raw_text)
        
        # 계층별로 문서 그룹핑
        hierarchy_groups = self._group_by_hierarchy(header_docs)
        
        # 각 레벨별 문서 생성
        for level, group_docs in hierarchy_groups.items():
            documents.extend(self._create_level_documents(group_docs, level))
        
        return documents
    
    def _group_by_hierarchy(self, header_docs: List[Document]) -> Dict[str, List[Document]]:
        """계층별로 문서 그룹핑"""
        groups: Dict[str, List[Document]] = {
            "major": [],     # 대제목 레벨 (H1)
            "medium": [],    # 중제목 레벨 (H2)
            "minor": [],     # 소제목 레벨 (H3, H4)
        }
        
        for doc in header_docs:
            # 가장 높은 헤더 레벨 확인
            if "Header 1" in doc.metadata:
                groups["major"].append(doc)
            elif "Header 2" in doc.metadata:
                groups["medium"].append(doc)
            else:
                groups["minor"].append(doc)
        
        return groups
    
    def _create_level_documents(self, docs: List[Document], level: str) -> List[Document]:
        """레벨별 문서 생성"""
        level_documents = []
        
        # 레벨별 청크 크기 설정
        chunk_sizes = {
            "major": 2000,    # 대제목: 큰 컨텍스트 보존
            "medium": 1200,   # 중제목: 중간 크기
            "minor": 800      # 소제목: 세밀한 분할
        }
        
        chunk_size = chunk_sizes.get(level, 1000)
        
        for doc in docs:
            # 계층적 제목 생성
            title_hierarchy = self._build_title_hierarchy(doc.metadata)
            
            # 컨텍스트 강화
            enhanced_content = self._enhance_with_context(doc.page_content, title_hierarchy, level)
            
            # 청킹
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "? ", "! ", ", "]
            )
            chunks = text_splitter.split_text(enhanced_content)
            
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata.update({
                    "chunk_index": i,
                    "hierarchy_level": level,
                    "title_hierarchy": title_hierarchy,
                    "source_strategy": f"multi_level_{level}",
                    "chunk_size": len(chunk),
                    "contains_pns": self._check_pns_context(chunk, title_hierarchy)
                })
                level_documents.append(Document(page_content=chunk, metadata=metadata))
        
        return level_documents
    
    def _build_title_hierarchy(self, metadata: Dict) -> str:
        """제목 계층 구조 생성"""
        hierarchy_parts = []
        for level in range(1, 5):
            header_key = f"Header {level}"
            if header_key in metadata and metadata[header_key]:
                hierarchy_parts.append(metadata[header_key].strip())
        return " > ".join(hierarchy_parts) if hierarchy_parts else "Unknown"
    
    def _enhance_with_context(self, content: str, title_hierarchy: str, level: str) -> str:
        """레벨별 컨텍스트 강화"""
        # PNS 관련 섹션인지 확인
        is_pns_section = "PNS" in title_hierarchy.upper() or "PAYMENT NOTIFICATION" in title_hierarchy.upper()
        
        context_info = f"[계층]: {title_hierarchy}\n"
        
        if is_pns_section:
            context_info += "[PNS 관련]: 이 내용은 PNS(Payment Notification Service) 결제알림서비스와 관련됩니다.\n"
        
        context_info += f"[레벨]: {level}\n\n"
        
        return context_info + content
    
    def _check_pns_context(self, content: str, title_hierarchy: str) -> bool:
        """PNS 컨텍스트 여부 확인"""
        content_upper = content.upper()
        hierarchy_upper = title_hierarchy.upper()
        
        return ("PNS" in hierarchy_upper or 
                "PAYMENT NOTIFICATION" in hierarchy_upper or
                "PNS" in content_upper or
                "PAYMENT NOTIFICATION" in content_upper)


class ContextInheritanceStrategy:
    """컨텍스트 상속 전략 - 상위 섹션 정보를 하위 섹션에 상속"""
    
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.raw_text = self._load_document()
        
    def _load_document(self) -> str:
        with open(self.document_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def split_documents(self) -> List[Document]:
        """컨텍스트 상속을 통한 문서 분할"""
        documents = []
        
        # 헤더 기반 분할
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4")
            ]
        )
        header_docs = header_splitter.split_text(self.raw_text)
        
        # 계층적 컨텍스트 구축
        context_map = self._build_context_map(header_docs)
        
        for doc in header_docs:
            # 상속된 컨텍스트 가져오기
            inherited_context = self._get_inherited_context(doc.metadata, context_map)
            
            # 컨텍스트 포함한 내용 생성
            enhanced_content = self._create_context_enhanced_content(
                doc.page_content, 
                doc.metadata, 
                inherited_context
            )
            
            # 청킹
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "? ", "! ", ", "]
            )
            chunks = text_splitter.split_text(enhanced_content)
            
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata.update({
                    "chunk_index": i,
                    "inherited_context": inherited_context,
                    "source_strategy": "context_inheritance",
                    "context_keywords": self._extract_context_keywords(inherited_context),
                    "pns_inheritance": self._check_pns_inheritance(inherited_context)
                })
                documents.append(Document(page_content=chunk, metadata=metadata))
        
        return documents
    
    def _build_context_map(self, header_docs: List[Document]) -> Dict[str, str]:
        """계층적 컨텍스트 맵 구축"""
        context_map = {}
        
        for doc in header_docs:
            # 각 헤더 레벨별 컨텍스트 저장
            for level in range(1, 5):
                header_key = f"Header {level}"
                if header_key in doc.metadata and doc.metadata[header_key]:
                    title = doc.metadata[header_key].strip()
                    
                    # 컨텍스트 정보 추출 (첫 문단)
                    first_paragraph = doc.page_content.split('\n\n')[0][:200]
                    context_map[title] = first_paragraph
        
        return context_map
    
    def _get_inherited_context(self, metadata: Dict, context_map: Dict[str, str]) -> str:
        """상속될 컨텍스트 수집"""
        inherited_contexts = []
        
        # 상위 레벨부터 컨텍스트 수집
        for level in range(1, 5):
            header_key = f"Header {level}"
            if header_key in metadata and metadata[header_key]:
                title = metadata[header_key].strip()
                if title in context_map:
                    context_info = f"[{title}]: {context_map[title]}"
                    inherited_contexts.append(context_info)
        
        return "\n".join(inherited_contexts) if inherited_contexts else ""
    
    def _create_context_enhanced_content(self, content: str, metadata: Dict, inherited_context: str) -> str:
        """컨텍스트가 강화된 내용 생성"""
        # 현재 섹션의 제목 경로
        title_path = self._build_title_path(metadata)
        
        enhanced_content = f"[상위 컨텍스트]:\n{inherited_context}\n\n"
        enhanced_content += f"[현재 섹션]: {title_path}\n\n"
        enhanced_content += f"[내용]:\n{content}"
        
        return enhanced_content
    
    def _build_title_path(self, metadata: Dict) -> str:
        """제목 경로 생성"""
        path_parts = []
        for level in range(1, 5):
            header_key = f"Header {level}"
            if header_key in metadata and metadata[header_key]:
                path_parts.append(metadata[header_key].strip())
        return " > ".join(path_parts) if path_parts else "Unknown"
    
    def _extract_context_keywords(self, context: str) -> List[str]:
        """컨텍스트에서 키워드 추출"""
        # 기술 용어 패턴
        tech_patterns = [
            r'\b[A-Z]{2,}\b',  # PNS, API 등
            r'\b[a-z]+[A-Z][a-zA-Z]*\b',  # purchaseState 등
        ]
        
        keywords = []
        for pattern in tech_patterns:
            keywords.extend(re.findall(pattern, context))
        
        # 한글 키워드
        korean_keywords = ['결제', '알림', '서비스', '메시지', '규격', '상태', '구매']
        for keyword in korean_keywords:
            if keyword in context:
                keywords.append(keyword)
        
        return list(set(keywords))
    
    def _check_pns_inheritance(self, context: str) -> bool:
        """PNS 상속 여부 확인"""
        context_upper = context.upper()
        return ("PNS" in context_upper or 
                "PAYMENT NOTIFICATION" in context_upper or
                "결제 알림" in context or
                "결제알림" in context)


class HierarchicalSmartRetriever:
    """계층적 스마트 검색기"""
    
    def __init__(self, documents: List[Document], embedding_model_name: str = "bge-m3:latest"):
        self.documents = documents
        self.embedding_model_name = embedding_model_name
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
    def build_retrievers(self):
        """검색기 구축"""
        print(f"🔧 계층적 검색기 구축 중... (문서 수: {len(self.documents)})")
        
        # Vector store 구축
        embeddings = OllamaEmbeddings(model=self.embedding_model_name)
        self.vector_store = FAISS.from_documents(self.documents, embeddings)
        
        # BM25 검색기 구축
        self.bm25_retriever = BM25Retriever.from_documents(
            self.documents,
            bm25_params={"k1": 1.5, "b": 0.75}
        )
        self.bm25_retriever.k = 20
        
        # Vector 검색기
        vector_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "fetch_k": 50, "lambda_mult": 0.7}
        )
        
        # 앙상블 검색기
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
        
        print("✅ 계층적 검색기 구축 완료")
    
    def hierarchical_search(self, query: str, max_results: int = 10) -> List[Document]:
        """계층적 검색"""
        if not self.ensemble_retriever:
            raise ValueError("검색기가 구축되지 않았습니다.")
        
        # 1. 기본 검색
        raw_results = self.ensemble_retriever.invoke(query)
        
        # 2. 계층적 점수 계산
        scored_results = self._calculate_hierarchical_scores(query, raw_results)
        
        # 3. 점수순 정렬
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for score, doc in scored_results[:max_results]]
    
    def _calculate_hierarchical_scores(self, query: str, documents: List[Document]) -> List[Tuple[float, Document]]:
        """계층적 점수 계산"""
        scored_docs = []
        query_keywords = self._extract_query_keywords(query)
        
        for doc in documents:
            score = 0.0
            content_lower = doc.page_content.lower()
            
            # 1. 기본 키워드 매칭
            for keyword in query_keywords:
                if keyword.lower() in content_lower:
                    score += 10
            
            # 2. 계층적 제목 매칭 보너스
            title_hierarchy = doc.metadata.get('title_hierarchy', '')
            for keyword in query_keywords:
                if keyword.lower() in title_hierarchy.lower():
                    score += 15  # 제목에 있으면 더 높은 점수
            
            # 3. PNS 상속/포함 보너스
            if doc.metadata.get('contains_pns', False) or doc.metadata.get('pns_inheritance', False):
                if any(kw.lower() in ['pns', 'payment', 'notification'] for kw in query_keywords):
                    score += 20
            
            # 4. 컨텍스트 키워드 매칭
            context_keywords = doc.metadata.get('context_keywords', [])
            for keyword in query_keywords:
                if keyword in context_keywords:
                    score += 8
            
            # 5. 레벨별 가중치
            hierarchy_level = doc.metadata.get('hierarchy_level', 'minor')
            level_weights = {'major': 1.2, 'medium': 1.1, 'minor': 1.0}
            score *= level_weights.get(hierarchy_level, 1.0)
            
            doc.metadata['hierarchical_score'] = score
            scored_docs.append((score, doc))
        
        return scored_docs
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """쿼리에서 키워드 추출"""
        tech_patterns = [
            r'\b[A-Z]{2,}\b',
            r'\b[a-z]+[A-Z][a-zA-Z]*\b',
        ]
        
        keywords = []
        for pattern in tech_patterns:
            keywords.extend(re.findall(pattern, query))
        
        korean_keywords = ['메시지', '규격', '값', '구성', '상태', '결제', '서버']
        for keyword in korean_keywords:
            if keyword in query:
                keywords.append(keyword)
        
        return list(set(keywords))


class HierarchicalExperimentRunner:
    """계층적 실험 실행기"""
    
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.strategies: Dict[str, Union[HierarchicalTitleStrategy, MultiLevelSplittingStrategy, ContextInheritanceStrategy]] = {
            'hierarchical_title': HierarchicalTitleStrategy(document_path),
            'multi_level': MultiLevelSplittingStrategy(document_path),
            'context_inheritance': ContextInheritanceStrategy(document_path)
        }
        self.test_queries = [
            "PNS 메시지 서버 규격의 purchaseState는 어떤 값으로 구성되나요?",
            "Payment Notification Service에서 purchaseState 값은 무엇인가요?",
            "PNS에서 사용되는 purchaseState의 COMPLETED와 CANCELED 의미는?",
            "결제 알림 서비스의 purchaseState 필드 설명해주세요",
            "원스토어 PNS 메시지 규격에서 구매 상태 정보는?"
        ]
    
    def run_experiments(self) -> Dict[str, List[HierarchicalSearchResult]]:
        """계층적 실험 실행"""
        results = {}
        
        for strategy_name, splitter in self.strategies.items():
            print(f"\n🧪 계층적 실험 시작: {strategy_name}")
            strategy_results = []
            
            # 문서 분할
            documents = splitter.split_documents()
            print(f"📄 분할된 문서 수: {len(documents)}")
            
            # PNS 관련 문서 수 확인
            pns_docs = [doc for doc in documents 
                       if doc.metadata.get('contains_pns', False) or 
                          doc.metadata.get('pns_inheritance', False) or
                          'PNS' in doc.page_content.upper()]
            print(f"🎯 PNS 관련 문서 수: {len(pns_docs)}")
            
            # 검색기 구축
            retriever = HierarchicalSmartRetriever(documents)
            retriever.build_retrievers()
            
            # 각 쿼리에 대해 테스트
            for query in self.test_queries:
                print(f"🔍 쿼리 테스트: {query[:40]}...")
                
                search_results = retriever.hierarchical_search(query, max_results=10)
                analysis = self._analyze_hierarchical_results(query, search_results, strategy_name)
                strategy_results.append(analysis)
            
            results[strategy_name] = strategy_results
            print(f"✅ {strategy_name} 전략 실험 완료")
        
        return results
    
    def _analyze_hierarchical_results(self, query: str, documents: List[Document], strategy_name: str) -> HierarchicalSearchResult:
        """계층적 결과 분석"""
        query_keywords = self._extract_keywords(query)
        hierarchy_scores = []
        context_scores = []
        relevant_count = 0
        pns_related_count = 0
        
        for doc in documents:
            # 계층적 점수
            hierarchy_score = doc.metadata.get('hierarchical_score', 0)
            hierarchy_scores.append(hierarchy_score)
            
            # 컨텍스트 점수
            title_hierarchy = doc.metadata.get('title_hierarchy', '')
            context_matches = sum(1 for kw in query_keywords 
                                if kw.lower() in title_hierarchy.lower())
            context_score = context_matches / len(query_keywords) if query_keywords else 0
            context_scores.append(context_score)
            
            # 관련성 체크
            content_lower = doc.page_content.lower()
            content_matches = sum(1 for kw in query_keywords if kw.lower() in content_lower)
            total_matches = content_matches + context_matches
            
            if total_matches >= len(query_keywords) * 0.5:  # 50% 이상 매칭
                relevant_count += 1
            
            # PNS 관련성 체크
            if (doc.metadata.get('contains_pns', False) or 
                doc.metadata.get('pns_inheritance', False) or
                'PNS' in content_lower.upper()):
                pns_related_count += 1
        
        return HierarchicalSearchResult(
            strategy_name=strategy_name,
            query=query,
            documents=documents,
            hierarchy_scores=hierarchy_scores,
            context_scores=context_scores,
            total_docs=len(documents),
            relevant_docs=relevant_count,
            pns_related_docs=pns_related_count
        )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """키워드 추출"""
        tech_patterns = [r'\b[A-Z]{2,}\b', r'\b[a-z]+[A-Z][a-zA-Z]*\b']
        keywords = []
        
        for pattern in tech_patterns:
            keywords.extend(re.findall(pattern, query))
        
        korean_words = ['메시지', '규격', '값', '구성', '상태', '결제', '서버', '알림']
        for word in korean_words:
            if word in query:
                keywords.append(word)
        
        return list(set(keywords))
    
    def print_hierarchical_results(self, results: Dict[str, List[HierarchicalSearchResult]]):
        """계층적 결과 출력"""
        print("\n" + "="*80)
        print("🏆 계층적 RAG 검색 최적화 실험 결과")
        print("="*80)
        
        for strategy_name, strategy_results in results.items():
            print(f"\n📊 전략: {strategy_name.upper()}")
            print("-" * 60)
            
            total_hierarchy_score = sum(sum(r.hierarchy_scores[:3]) for r in strategy_results)
            total_context_score = sum(sum(r.context_scores[:3]) for r in strategy_results)
            total_relevant = sum(r.relevant_docs for r in strategy_results)
            total_pns_related = sum(r.pns_related_docs for r in strategy_results)
            
            avg_hierarchy = total_hierarchy_score / (len(strategy_results) * 3)
            avg_context = total_context_score / (len(strategy_results) * 3)
            
            print(f"평균 계층 점수 (상위3): {avg_hierarchy:.2f}")
            print(f"평균 컨텍스트 점수 (상위3): {avg_context:.3f}")
            print(f"전체 관련 문서 수: {total_relevant}")
            print(f"PNS 관련 문서 수: {total_pns_related}")
            
            # 핵심 쿼리 결과
            pns_query_result = strategy_results[0]  # 첫 번째 쿼리가 핵심 쿼리
            print(f"\n🎯 핵심 쿼리 결과 (PNS + purchaseState):")
            print(f"  관련 문서: {pns_query_result.relevant_docs}/10")
            print(f"  PNS 연결: {pns_query_result.pns_related_docs}/10")
            print(f"  평균 계층 점수: {sum(pns_query_result.hierarchy_scores[:3])/3:.2f}")


def main():
    """메인 실행 함수"""
    print("🚀 계층적 RAG 검색 최적화 실험 시작")
    print("💡 목표: PNS 대제목과 하위 섹션 purchaseState 연결 문제 해결")
    
    document_path = "../data/dev_center_guide_allmd_touched.md"
    
    if not os.path.exists(document_path):
        print(f"❌ 문서를 찾을 수 없습니다: {document_path}")
        return
    
    # 실험 실행
    runner = HierarchicalExperimentRunner(document_path)
    results = runner.run_experiments()
    
    # 결과 출력
    runner.print_hierarchical_results(results)
    
    # 결과 저장
    os.makedirs("results", exist_ok=True)
    output_path = "results/hierarchical_experiment_results.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n💾 계층적 실험 결과 저장됨: {output_path}")
    
    # 최적 전략 추천
    print(f"\n🎯 권장사항:")
    print(f"1. context_inheritance 전략이 PNS-purchaseState 연결에 가장 효과적일 것으로 예상")
    print(f"2. multi_level 전략으로 다양한 크기의 컨텍스트 제공")
    print(f"3. hierarchical_title 전략으로 명시적 제목 경로 포함")


if __name__ == "__main__":
    main()
