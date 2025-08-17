"""
RAG 검색 최적화를 위한 다양한 문서 분할 전략 테스트

이 모듈은 다음과 같은 전략들을 테스트합니다:
1. 키워드 기반 분할 (PNS, purchaseState 등 핵심 키워드 중심)
2. 의미 기반 분할 (문맥 보존)
3. 하이브리드 분할 (키워드 + 의미)
4. 우선순위 기반 검색 (키워드 매칭 점수)
"""

import os
import re
import pickle
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


@dataclass
class SearchResult:
    """검색 결과 분석을 위한 데이터 클래스"""
    strategy_name: str
    query: str
    documents: List[Document]
    keyword_scores: List[float]
    total_docs: int
    relevant_docs: int
    top_3_relevance: float


class DocumentSplittingStrategy:
    """문서 분할 전략의 기본 클래스"""
    
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.raw_text = self._load_document()
        
    def _load_document(self) -> str:
        """원본 문서 로드"""
        with open(self.document_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def split_documents(self) -> List[Document]:
        """문서 분할 (각 전략에서 구현)"""
        raise NotImplementedError
    
    def get_strategy_name(self) -> str:
        """전략 이름 반환"""
        raise NotImplementedError


class KeywordBasedSplitter(DocumentSplittingStrategy):
    """키워드 기반 분할 전략"""
    
    def __init__(self, document_path: str, target_keywords: Optional[List[str]] = None):
        super().__init__(document_path)
        self.target_keywords = target_keywords or [
            "PNS", "purchaseState", "Payment Notification", 
            "결제", "구매", "상태", "메시지", "규격"
        ]
    
    def get_strategy_name(self) -> str:
        return "keyword_based"
    
    def split_documents(self) -> List[Document]:
        """키워드 중심으로 문서 분할"""
        documents = []
        
        # 1. 헤더 기반 1차 분할
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"), 
                ("###", "Header 3"),
            ]
        )
        header_docs = header_splitter.split_text(self.raw_text)
        
        # 2. 키워드 기반 청킹 및 강화
        for doc in header_docs:
            enhanced_content = self._enhance_with_keywords(doc.page_content)
            
            # 키워드 밀도 계산
            keyword_density = self._calculate_keyword_density(enhanced_content)
            
            # 키워드가 많이 포함된 문서는 더 세분화
            if keyword_density > 0.02:  # 2% 이상
                sub_docs = self._split_keyword_rich_content(enhanced_content, doc.metadata)
                documents.extend(sub_docs)
            else:
                # 일반 청킹
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ". ", ", ", " "]
                )
                chunks = text_splitter.split_text(enhanced_content)
                
                for i, chunk in enumerate(chunks):
                    metadata = doc.metadata.copy()
                    metadata.update({
                        "chunk_index": i,
                        "keyword_density": self._calculate_keyword_density(chunk),
                        "source_strategy": "keyword_based"
                    })
                    documents.append(Document(page_content=chunk, metadata=metadata))
        
        return documents
    
    def _enhance_with_keywords(self, content: str) -> str:
        """키워드 확장으로 검색성 향상"""
        enhancements = {
            "PNS": "PNS Payment Notification Service 결제알림서비스",
            "purchaseState": "purchaseState 구매상태 결제상태",
            "COMPLETED": "COMPLETED 완료 구매완료 결제완료",
            "CANCELED": "CANCELED 취소 구매취소 결제취소"
        }
        
        enhanced = content
        for keyword, expansion in enhancements.items():
            if keyword in content:
                enhanced = enhanced.replace(keyword, expansion, 1)  # 첫 번째만 교체
        
        return enhanced
    
    def _calculate_keyword_density(self, content: str) -> float:
        """키워드 밀도 계산"""
        words = content.lower().split()
        keyword_count = sum(1 for word in words 
                          if any(kw.lower() in word for kw in self.target_keywords))
        return keyword_count / len(words) if words else 0
    
    def _split_keyword_rich_content(self, content: str, base_metadata: Dict) -> List[Document]:
        """키워드가 많은 콘텐츠를 더 세밀하게 분할"""
        documents = []
        
        # 키워드 중심으로 세그먼트 분할
        segments = self._segment_by_keywords(content)
        
        for i, segment in enumerate(segments):
            metadata = base_metadata.copy()
            metadata.update({
                "chunk_index": i,
                "keyword_density": self._calculate_keyword_density(segment),
                "source_strategy": "keyword_based_detailed",
                "keywords_found": self._extract_keywords_from_content(segment)
            })
            documents.append(Document(page_content=segment, metadata=metadata))
        
        return documents
    
    def _segment_by_keywords(self, content: str) -> List[str]:
        """키워드를 기준으로 세그먼트 분할"""
        # 문장 단위로 분할
        sentences = re.split(r'[.!?]\s+', content)
        segments = []
        current_segment = []
        
        for sentence in sentences:
            current_segment.append(sentence)
            
            # 키워드가 포함된 문장이면서 적정 길이가 되면 세그먼트 완성
            has_keyword = any(kw.lower() in sentence.lower() for kw in self.target_keywords)
            segment_text = '. '.join(current_segment)
            
            if has_keyword and len(segment_text) > 200:
                segments.append(segment_text + '.')
                current_segment = []
        
        # 남은 문장들 처리
        if current_segment:
            segments.append('. '.join(current_segment) + '.')
        
        return [seg for seg in segments if len(seg.strip()) > 50]
    
    def _extract_keywords_from_content(self, content: str) -> List[str]:
        """콘텐츠에서 발견된 키워드 추출"""
        found_keywords = []
        content_lower = content.lower()
        
        for keyword in self.target_keywords:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
        
        return found_keywords


class SemanticBasedSplitter(DocumentSplittingStrategy):
    """의미 기반 분할 전략"""
    
    def get_strategy_name(self) -> str:
        return "semantic_based"
    
    def split_documents(self) -> List[Document]:
        """의미 단위로 문서 분할"""
        documents = []
        
        # 헤더 기반 분할
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
        header_docs = header_splitter.split_text(self.raw_text)
        
        # 각 섹션을 의미 단위로 분할
        for doc in header_docs:
            semantic_chunks = self._split_by_semantic_units(doc.page_content)
            
            for i, chunk in enumerate(semantic_chunks):
                metadata = doc.metadata.copy()
                metadata.update({
                    "chunk_index": i,
                    "source_strategy": "semantic_based",
                    "semantic_score": self._calculate_semantic_score(chunk)
                })
                documents.append(Document(page_content=chunk, metadata=metadata))
        
        return documents
    
    def _split_by_semantic_units(self, content: str) -> List[str]:
        """의미 단위로 분할"""
        # 단락, 문장, 의미 구조를 고려한 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # 더 큰 청크로 문맥 보존
            chunk_overlap=200,
            separators=["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", ", "]
        )
        return text_splitter.split_text(content)
    
    def _calculate_semantic_score(self, content: str) -> float:
        """의미적 완성도 점수 계산"""
        # 간단한 휴리스틱으로 의미적 완성도 측정
        sentences = re.split(r'[.!?]', content)
        complete_sentences = [s for s in sentences if len(s.strip()) > 10]
        
        # 완성된 문장 비율, 길이 적정성 등을 고려
        completeness = len(complete_sentences) / len(sentences) if sentences else 0
        length_score = min(1.0, len(content) / 500)  # 500자 기준 정규화
        
        return (completeness + length_score) / 2


class HybridSplitter(DocumentSplittingStrategy):
    """하이브리드 분할 전략 (키워드 + 의미)"""
    
    def __init__(self, document_path: str, target_keywords: Optional[List[str]] = None):
        super().__init__(document_path)
        self.keyword_splitter = KeywordBasedSplitter(document_path, target_keywords)
        self.semantic_splitter = SemanticBasedSplitter(document_path)
    
    def get_strategy_name(self) -> str:
        return "hybrid"
    
    def split_documents(self) -> List[Document]:
        """키워드와 의미를 모두 고려한 분할"""
        documents = []
        
        # 1차: 키워드 기반 분할
        keyword_docs = self.keyword_splitter.split_documents()
        
        # 2차: 키워드 밀도가 낮은 문서는 의미 기반으로 재분할
        for doc in keyword_docs:
            keyword_density = doc.metadata.get('keyword_density', 0)
            
            if keyword_density < 0.01:  # 키워드 밀도가 낮으면
                # 의미 기반으로 재분할
                semantic_chunks = self.semantic_splitter._split_by_semantic_units(doc.page_content)
                
                for i, chunk in enumerate(semantic_chunks):
                    metadata = doc.metadata.copy()
                    metadata.update({
                        "chunk_index": f"{doc.metadata.get('chunk_index', 0)}_{i}",
                        "source_strategy": "hybrid_semantic",
                        "semantic_score": self.semantic_splitter._calculate_semantic_score(chunk)
                    })
                    documents.append(Document(page_content=chunk, metadata=metadata))
            else:
                # 키워드 밀도가 높으면 그대로 유지
                doc.metadata["source_strategy"] = "hybrid_keyword"
                documents.append(doc)
        
        return documents


class SmartRetriever:
    """스마트 검색기 - 키워드 우선순위 기반"""
    
    def __init__(self, documents: List[Document], embedding_model_name: str = "bge-m3:latest"):
        self.documents = documents
        self.embedding_model_name = embedding_model_name
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
    def build_retrievers(self):
        """검색기 구축"""
        print(f"🔧 검색기 구축 중... (문서 수: {len(self.documents)})")
        
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
            weights=[0.6, 0.4]  # BM25에 더 높은 가중치
        )
        
        print("✅ 검색기 구축 완료")
    
    def smart_search(self, query: str, max_results: int = 10) -> List[Document]:
        """스마트 검색 - 키워드 우선순위 적용"""
        if not self.ensemble_retriever:
            raise ValueError("검색기가 구축되지 않았습니다. build_retrievers()를 먼저 호출하세요.")
        
        # 1. 앙상블 검색으로 더 많은 후보 검색
        raw_results = self.ensemble_retriever.invoke(query)
        
        # 2. 키워드 기반 필터링 및 점수 계산
        scored_results = self._score_documents(query, raw_results)
        
        # 3. 점수순 정렬
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # 4. 상위 결과 반환
        return [doc for score, doc in scored_results[:max_results]]
    
    def _score_documents(self, query: str, documents: List[Document]) -> List[Tuple[float, Document]]:
        """문서 점수 계산"""
        scored_docs = []
        query_keywords = self._extract_query_keywords(query)
        
        for doc in documents:
            score = 0.0
            content_lower = doc.page_content.lower()
            
            # 1. 키워드 매칭 점수 (가장 중요)
            keyword_matches = 0
            for keyword in query_keywords:
                if keyword.lower() in content_lower:
                    keyword_matches += 1
                    # 정확한 매칭에 높은 점수
                    if keyword.lower() == keyword.lower():  # 완전 일치
                        score += 10
                    else:
                        score += 5
            
            # 2. 키워드 밀도 점수
            density = doc.metadata.get('keyword_density', 0)
            score += density * 20
            
            # 3. 전략별 보너스 점수
            strategy = doc.metadata.get('source_strategy', '')
            if 'keyword' in strategy:
                score += 5
            
            # 4. 위치 점수 (문서 앞부분에 키워드가 있으면 가점)
            first_half = content_lower[:len(content_lower)//2]
            if any(kw.lower() in first_half for kw in query_keywords):
                score += 8
            
            # 5. 길이 적정성 점수
            doc_length = len(doc.page_content.split())
            if 50 <= doc_length <= 300:  # 적정 길이
                score += 3
            
            doc.metadata['search_score'] = score
            doc.metadata['keyword_matches'] = keyword_matches
            scored_docs.append((score, doc))
        
        return scored_docs
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """쿼리에서 중요 키워드 추출"""
        # 기술 용어 패턴
        tech_patterns = [
            r'\b[A-Z]{2,}\b',  # PNS, API 등
            r'\b[a-z]+[A-Z][a-zA-Z]*\b',  # purchaseState 등
        ]
        
        keywords = []
        for pattern in tech_patterns:
            keywords.extend(re.findall(pattern, query))
        
        # 한글 키워드 추가
        korean_keywords = ['메시지', '규격', '값', '구성', '상태', '결제', '서버']
        for keyword in korean_keywords:
            if keyword in query:
                keywords.append(keyword)
        
        return list(set(keywords))


class RAGExperimentRunner:
    """RAG 실험 실행기"""
    
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.strategies = {
            'keyword': KeywordBasedSplitter(document_path),
            'semantic': SemanticBasedSplitter(document_path), 
            'hybrid': HybridSplitter(document_path)
        }
        self.test_queries = [
            "PNS 메시지 서버 규격의 purchaseState는 어떤 값으로 구성되나요?",
            "Payment Notification Service의 메시지 구조는 어떻게 되나요?",
            "purchaseState COMPLETED CANCELED 값은 무엇을 의미하나요?",
            "결제 상태 정보를 서버에서 받는 방법은?",
            "PNS 설정 방법과 URL 구성은?"
        ]
    
    def run_experiments(self) -> Dict[str, List[SearchResult]]:
        """모든 전략에 대해 실험 실행"""
        results = {}
        
        for strategy_name, splitter in self.strategies.items():
            print(f"\n🧪 실험 시작: {strategy_name}")
            strategy_results = []
            
            # 문서 분할
            documents = splitter.split_documents()
            print(f"📄 분할된 문서 수: {len(documents)}")
            
            # 검색기 구축
            retriever = SmartRetriever(documents)
            retriever.build_retrievers()
            
            # 각 쿼리에 대해 테스트
            for query in self.test_queries:
                print(f"🔍 쿼리 테스트: {query[:30]}...")
                
                search_results = retriever.smart_search(query, max_results=10)
                analysis = self._analyze_results(query, search_results, strategy_name)
                strategy_results.append(analysis)
            
            results[strategy_name] = strategy_results
            print(f"✅ {strategy_name} 전략 실험 완료")
        
        return results
    
    def _analyze_results(self, query: str, documents: List[Document], strategy_name: str) -> SearchResult:
        """검색 결과 분석"""
        query_keywords = self._extract_keywords(query)
        keyword_scores = []
        relevant_count = 0
        
        for doc in documents:
            # 키워드 매칭 점수 계산
            content_lower = doc.page_content.lower()
            matches = sum(1 for kw in query_keywords if kw.lower() in content_lower)
            score = matches / len(query_keywords) if query_keywords else 0
            keyword_scores.append(score)
            
            if score > 0.3:  # 30% 이상 키워드 매칭
                relevant_count += 1
        
        # 상위 3개 문서의 평균 관련성
        top_3_relevance = sum(keyword_scores[:3]) / 3 if len(keyword_scores) >= 3 else 0
        
        return SearchResult(
            strategy_name=strategy_name,
            query=query,
            documents=documents,
            keyword_scores=keyword_scores,
            total_docs=len(documents),
            relevant_docs=relevant_count,
            top_3_relevance=top_3_relevance
        )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """쿼리에서 키워드 추출"""
        keywords = []
        
        # 영문 패턴
        tech_patterns = [r'\b[A-Z]{2,}\b', r'\b[a-z]+[A-Z][a-zA-Z]*\b']
        for pattern in tech_patterns:
            keywords.extend(re.findall(pattern, query))
        
        # 한글 키워드
        korean_words = ['메시지', '규격', '값', '구성', '상태', '결제', '서버']
        for word in korean_words:
            if word in query:
                keywords.append(word)
        
        return list(set(keywords))
    
    def print_results_summary(self, results: Dict[str, List[SearchResult]]):
        """결과 요약 출력"""
        print("\n" + "="*80)
        print("🏆 RAG 검색 최적화 실험 결과 요약")
        print("="*80)
        
        for strategy_name, strategy_results in results.items():
            print(f"\n📊 전략: {strategy_name.upper()}")
            print("-" * 50)
            
            total_relevance = sum(r.top_3_relevance for r in strategy_results)
            avg_relevance = total_relevance / len(strategy_results)
            total_relevant_docs = sum(r.relevant_docs for r in strategy_results)
            
            print(f"평균 상위3 관련성: {avg_relevance:.3f}")
            print(f"전체 관련 문서 수: {total_relevant_docs}")
            print(f"평균 관련 문서 비율: {total_relevant_docs/len(strategy_results):.1f}")
            
            # 각 쿼리별 상세 결과
            for result in strategy_results:
                print(f"  • {result.query[:40]}... -> 관련성: {result.top_3_relevance:.3f}")


def main():
    """메인 실행 함수"""
    print("🚀 RAG 검색 최적화 실험 시작")
    
    document_path = "../data/dev_center_guide_allmd_touched.md"
    
    if not os.path.exists(document_path):
        print(f"❌ 문서를 찾을 수 없습니다: {document_path}")
        return
    
    # 실험 실행
    runner = RAGExperimentRunner(document_path)
    results = runner.run_experiments()
    
    # 결과 출력
    runner.print_results_summary(results)
    
    # 결과 저장
    os.makedirs("results", exist_ok=True)
    output_path = "results/experiment_results.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n💾 결과 저장됨: {output_path}")


if __name__ == "__main__":
    main()
