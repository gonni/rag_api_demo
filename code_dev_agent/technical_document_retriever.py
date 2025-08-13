"""
기술문서 특화 검색기

이 모듈은 기술문서의 특성(JSON 규격, 코드 블록, 표 등)을 고려하여
전체 맥락을 보존하면서도 정확한 검색을 제공하는 검색기를 구현합니다.
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


class TechnicalDocumentRetriever:
    """기술문서 특화 검색기"""
    
    def __init__(self, documents: List[Document], embedding_model: str = "bge-m3:latest"):
        self.documents = documents
        self.embedding_model = embedding_model
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
    def build_retrievers(self):
        """검색기 구축"""
        print(f"🔧 기술문서 특화 검색기 구축 중... (문서 수: {len(self.documents)})")
        
        # Vector store 구축
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.vector_store = FAISS.from_documents(self.documents, embeddings)
        
        # BM25 검색기 구축
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 30
        
        # 앙상블 검색기
        vector_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "fetch_k": 50, "lambda_mult": 0.7}
        )
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[0.6, 0.4]
        )
        
        print("✅ 기술문서 특화 검색기 구축 완료")
    
    def retrieve_technical_content(self, query: str, k: int = 10) -> List[Document]:
        """기술 콘텐츠 검색"""
        if not self.ensemble_retriever:
            raise ValueError("검색기가 구축되지 않았습니다.")
        
        # 1. 기본 앙상블 검색
        base_results = self.ensemble_retriever.invoke(query)
        
        # 2. 기술 콘텐츠 특화 필터링 및 점수 계산
        scored_results = self._score_technical_content(query, base_results)
        
        # 3. 완전한 블록 우선 정렬
        prioritized_results = self._prioritize_complete_blocks(scored_results)
        
        return [doc for score, doc in prioritized_results[:k]]
    
    def _score_technical_content(self, query: str, documents: List[Document]) -> List[Tuple[float, Document]]:
        """기술 콘텐츠 점수 계산"""
        scored_docs = []
        query_lower = query.lower()
        
        for doc in documents:
            score = 0.0
            metadata = doc.metadata
            content = doc.page_content
            
            # 1. 블록 타입별 점수
            block_type = metadata.get('block_type', 'unknown')
            score += self._calculate_block_type_score(query_lower, block_type)
            
            # 2. 완성도 점수
            if metadata.get('is_complete_block', False):
                score += 3.0
            
            # 3. 콘텐츠 타입별 점수
            content_type = metadata.get('content_type', 'unknown')
            score += self._calculate_content_type_score(query_lower, content_type)
            
            # 4. 키워드 매칭 점수
            score += self._calculate_keyword_score(query_lower, content)
            
            # 5. 구조적 데이터 포함 점수
            if metadata.get('contains_structured_data', False):
                score += 2.0
            
            # 6. 크기 적정성 점수
            content_length = metadata.get('content_length', 0)
            score += self._calculate_size_score(content_length)
            
            scored_docs.append((score, doc))
        
        return scored_docs
    
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
        
        # API 관련 질의
        if 'api' in query or 'endpoint' in query or '요청' in query:
            if block_type == 'api_endpoint':
                score += 4.0
        
        return score
    
    def _calculate_content_type_score(self, query: str, content_type: str) -> float:
        """콘텐츠 타입별 점수 계산"""
        score = 0.0
        
        # JSON 규격 관련
        if 'json' in query or '규격' in query:
            if content_type == 'json_specification':
                score += 3.0
        
        # 데이터 테이블 관련
        if '표' in query or '데이터' in query:
            if content_type == 'data_table':
                score += 3.0
        
        # 코드 예제 관련
        if '코드' in query or '예제' in query:
            if content_type == 'code_example':
                score += 3.0
        
        # API 엔드포인트 관련
        if 'api' in query or 'endpoint' in query:
            if content_type == 'api_endpoint':
                score += 3.0
        
        return score
    
    def _calculate_keyword_score(self, query: str, content: str) -> float:
        """키워드 매칭 점수 계산"""
        score = 0.0
        content_lower = content.lower()
        
        # 기술 용어 패턴
        tech_patterns = [
            r'\b[A-Z]{2,}\b',  # API, JSON 등
            r'\b[a-z]+[A-Z][a-zA-Z]*\b',  # purchaseState 등
            r'\b\d{3,}\b',  # HTTP 상태 코드
        ]
        
        # 쿼리에서 기술 용어 추출
        query_keywords = []
        for pattern in tech_patterns:
            query_keywords.extend(re.findall(pattern, query))
        
        # 매칭 점수 계산
        for keyword in query_keywords:
            if keyword.lower() in content_lower:
                score += 2.0
        
        # 일반 키워드 매칭
        general_keywords = ['메시지', '규격', '요청', '응답', '코드', '표', '예제']
        for keyword in general_keywords:
            if keyword in query and keyword in content_lower:
                score += 1.0
        
        return score
    
    def _calculate_size_score(self, content_length: int) -> float:
        """크기 적정성 점수 계산"""
        if 100 <= content_length <= 2000:
            return 2.0  # 적정 크기
        elif content_length > 2000:
            return 1.0  # 큰 크기 (완전성 보장)
        else:
            return 0.5  # 작은 크기
    
    def _prioritize_complete_blocks(self, scored_results: List[Tuple[float, Document]]) -> List[Tuple[float, Document]]:
        """완전한 블록 우선 정렬"""
        # 완전한 블록과 불완전한 블록 분리
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
        
        # 완전한 블록을 먼저, 그 다음 불완전한 블록
        return complete_blocks + incomplete_blocks
    
    def search_by_block_type(self, query: str, block_type: str, k: int = 5) -> List[Document]:
        """블록 타입별 검색"""
        if not self.ensemble_retriever:
            raise ValueError("검색기가 구축되지 않았습니다.")
        
        # 기본 검색 결과
        base_results = self.ensemble_retriever.invoke(query)
        
        # 블록 타입 필터링
        filtered_results = []
        for doc in base_results:
            if doc.metadata.get('block_type') == block_type:
                filtered_results.append(doc)
        
        # 점수 계산 및 정렬
        scored_results = self._score_technical_content(query, filtered_results)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for score, doc in scored_results[:k]]
    
    def search_complete_specifications(self, query: str, k: int = 5) -> List[Document]:
        """완전한 규격 검색"""
        if not self.ensemble_retriever:
            raise ValueError("검색기가 구축되지 않았습니다.")
        
        # 기본 검색 결과
        base_results = self.ensemble_retriever.invoke(query)
        
        # 완전한 블록만 필터링
        complete_blocks = []
        for doc in base_results:
            if doc.metadata.get('is_complete_block', False):
                complete_blocks.append(doc)
        
        # 점수 계산 및 정렬
        scored_results = self._score_technical_content(query, complete_blocks)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for score, doc in scored_results[:k]]


class TechnicalQueryAnalyzer:
    """기술 질의 분석기"""
    
    def __init__(self):
        self.query_patterns = {
            'json_specification': [
                r'json.*?규격',
                r'메시지.*?규격',
                r'요청.*?body',
                r'응답.*?형식'
            ],
            'code_example': [
                r'코드.*?예제',
                r'예제.*?코드',
                r'구현.*?방법',
                r'사용.*?방법'
            ],
            'data_table': [
                r'표.*?정보',
                r'데이터.*?표',
                r'코드.*?표',
                r'상태.*?코드'
            ],
            'api_endpoint': [
                r'api.*?endpoint',
                r'엔드포인트',
                r'요청.*?url',
                r'http.*?메서드'
            ]
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """질의 분석"""
        query_lower = query.lower()
        
        analysis = {
            'query_type': 'general',
            'target_block_types': [],
            'requires_complete_spec': False,
            'confidence': 0.0
        }
        
        # 블록 타입별 패턴 매칭
        max_matches = 0
        best_type = 'general'
        
        for block_type, patterns in self.query_patterns.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    matches += 1
            
            if matches > max_matches:
                max_matches = matches
                best_type = block_type
        
        analysis['query_type'] = best_type
        
        # 완전한 규격 필요 여부 확인
        if '전체' in query or '모든' in query or '규격' in query:
            analysis['requires_complete_spec'] = True
        
        # 신뢰도 계산
        analysis['confidence'] = max_matches / len(self.query_patterns[best_type]) if best_type != 'general' else 0.0
        
        # 타겟 블록 타입 설정
        if best_type == 'json_specification':
            analysis['target_block_types'] = ['json_specification']
        elif best_type == 'code_example':
            analysis['target_block_types'] = ['code_block']
        elif best_type == 'data_table':
            analysis['target_block_types'] = ['table']
        elif best_type == 'api_endpoint':
            analysis['target_block_types'] = ['api_endpoint']
        
        return analysis


class TechnicalDocumentSearchEngine:
    """기술문서 검색 엔진"""
    
    def __init__(self, documents: List[Document], embedding_model: str = "bge-m3:latest"):
        self.documents = documents
        self.retriever = TechnicalDocumentRetriever(documents, embedding_model)
        self.query_analyzer = TechnicalQueryAnalyzer()
        
    def setup(self):
        """검색 엔진 초기화"""
        print("🚀 기술문서 검색 엔진 초기화...")
        self.retriever.build_retrievers()
        print("✅ 기술문서 검색 엔진 초기화 완료")
    
    def search(self, query: str, k: int = 10) -> Dict[str, Any]:
        """통합 검색"""
        # 질의 분석
        query_analysis = self.query_analyzer.analyze_query(query)
        
        # 검색 실행
        if query_analysis['requires_complete_spec']:
            results = self.retriever.search_complete_specifications(query, k)
        elif query_analysis['target_block_types']:
            # 특정 블록 타입 검색
            block_type = query_analysis['target_block_types'][0]
            results = self.retriever.search_by_block_type(query, block_type, k)
        else:
            # 일반 검색
            results = self.retriever.retrieve_technical_content(query, k)
        
        return {
            'query': query,
            'query_analysis': query_analysis,
            'results': results,
            'total_results': len(results),
            'complete_specs': sum(1 for doc in results if doc.metadata.get('is_complete_block', False))
        }
    
    def search_json_specifications(self, query: str, k: int = 5) -> List[Document]:
        """JSON 규격 전용 검색"""
        return self.retriever.search_by_block_type(query, 'json_specification', k)
    
    def search_code_examples(self, query: str, k: int = 5) -> List[Document]:
        """코드 예제 전용 검색"""
        return self.retriever.search_by_block_type(query, 'code_block', k)
    
    def search_data_tables(self, query: str, k: int = 5) -> List[Document]:
        """데이터 테이블 전용 검색"""
        return self.retriever.search_by_block_type(query, 'table', k)
    
    def search_api_endpoints(self, query: str, k: int = 5) -> List[Document]:
        """API 엔드포인트 전용 검색"""
        return self.retriever.search_by_block_type(query, 'api_endpoint', k)


# 사용 예시
def demonstrate_technical_search():
    """기술문서 검색 데모"""
    print("🚀 기술문서 특화 검색 데모")
    print("=" * 50)
    
    # 샘플 문서 생성
    sample_docs = [
        Document(
            page_content="""
[블록 타입]: json_specification
[라인 범위]: 1-25
[설명]: 이 내용은 JSON 메시지 규격입니다. 전체 구조를 파악하기 위해 완전한 형태로 유지됩니다.

{
  "msgVersion": "3.1.0",
  "clientId": "0000000001",
  "purchaseState": "COMPLETED",
  "price": "10000"
}
            """,
            metadata={
                'block_type': 'json_specification',
                'content_type': 'json_specification',
                'is_complete_block': True,
                'contains_structured_data': True,
                'content_length': 200
            }
        ),
        Document(
            page_content="""
[블록 타입]: table
[라인 범위]: 1-5
[설명]: 이 내용은 데이터 테이블입니다. 전체 구조를 파악하기 위해 완전한 형태로 유지됩니다.

| 코드 | 설명 |
|------|------|
| 200 | 성공 |
| 400 | 잘못된 요청 |
            """,
            metadata={
                'block_type': 'table',
                'content_type': 'data_table',
                'is_complete_block': True,
                'table_headers': ['코드', '설명'],
                'content_length': 150
            }
        )
    ]
    
    # 검색 엔진 초기화
    search_engine = TechnicalDocumentSearchEngine(sample_docs)
    search_engine.setup()
    
    # 다양한 질의로 테스트
    test_queries = [
        "JSON 메시지 규격이 어떻게 됩니까?",
        "응답 코드 표를 보여주세요",
        "전체 메시지 구조를 알려주세요"
    ]
    
    for query in test_queries:
        print(f"\n🔍 질의: {query}")
        result = search_engine.search(query)
        
        print(f"  - 질의 타입: {result['query_analysis']['query_type']}")
        print(f"  - 완전한 규격 필요: {result['query_analysis']['requires_complete_spec']}")
        print(f"  - 검색 결과: {result['total_results']}개")
        print(f"  - 완전한 블록: {result['complete_specs']}개")
        
        for i, doc in enumerate(result['results'][:3], 1):
            print(f"    {i}. {doc.metadata.get('block_type', 'unknown')} - {doc.page_content[:50]}...")


if __name__ == "__main__":
    demonstrate_technical_search()
