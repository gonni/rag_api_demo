"""
Context-Aware Retriever for Technical Documentation
컨텍스트 인식 기술문서 검색기

특징:
1. 전문용어의 맥락적 의미 고려
2. 섹션별 컨텍스트 보존
3. 다단계 리랭킹 (관련성, 맥락, 기술정확성)
4. 쿼리 확장 (동의어, 약어)
5. 노이즈 필터링 (중요도 기반)
"""

import re
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun


@dataclass
class SearchContext:
    """검색 컨텍스트 정보"""
    query: str
    query_terms: List[str]
    tech_terms: Dict[str, List[str]]
    query_type: str  # 'how_to', 'what_is', 'api_spec', 'error_debug'
    context_keywords: Set[str]


class QueryAnalyzer:
    """쿼리 분석 및 확장"""
    
    def __init__(self):
        # 기술 용어 동의어 사전
        self.tech_synonyms = {
            'IAP': ['인앱결제', '인앱', '결제'],
            'SDK': ['소프트웨어개발키트', '개발킷'],
            'API': ['인터페이스', '연동'],
            'PNS': ['결제알림서비스', 'Payment Notification Service'],
            'PurchaseClient': ['구매클라이언트', '결제클라이언트'],
            'purchaseState': ['구매상태', '결제상태'],
            'purchaseToken': ['구매토큰', '결제토큰'],
            'acknowledge': ['구매확인', '승인'],
            'consume': ['소비', '사용완료'],
            'subscription': ['구독', '정기결제'],
            'recurring': ['월정액', '자동결제', '반복결제']
        }
        
        # 쿼리 타입 패턴
        self.query_patterns = {
            'how_to': re.compile(r'(?:어떻게|방법|사용|구현|적용|설정|연동)'),
            'what_is': re.compile(r'(?:무엇|뭔가요|란|이란|설명|개념)'),
            'api_spec': re.compile(r'(?:파라미터|리턴|응답|요청|스펙|명세)'),
            'error_debug': re.compile(r'(?:에러|오류|문제|실패|안됨|해결)')
        }
        
        # 한국어 조사 패턴
        self.particle_pattern = re.compile(r'([가-힣]+)(?:은|는|이|가|을|를|의|에|와|과|도|로|으로|부터|까지|만)')
    
    def expand_query(self, query: str) -> List[str]:
        """쿼리 확장 (동의어, 약어)"""
        expanded = [query]
        
        for term, synonyms in self.tech_synonyms.items():
            if term.lower() in query.lower():
                for synonym in synonyms:
                    expanded.append(query.replace(term, synonym))
            else:
                for synonym in synonyms:
                    if synonym in query:
                        expanded.append(query.replace(synonym, term))
        
        return list(set(expanded))
    
    def extract_query_terms(self, query: str) -> List[str]:
        """쿼리에서 핵심 용어 추출"""
        # 조사 제거
        query_cleaned = self.particle_pattern.sub(r'\1', query)
        
        terms = []
        
        # 기술 용어 추출
        for term in self.tech_synonyms.keys():
            if term.lower() in query_cleaned.lower():
                terms.append(term)
        
        # 영어 용어 추출
        english_terms = re.findall(r'\b[A-Za-z][A-Za-z0-9_]*\b', query_cleaned)
        terms.extend([t for t in english_terms if len(t) > 2])
        
        # 한글 명사 추출 (간단한 휴리스틱)
        korean_terms = re.findall(r'[가-힣]{2,}', query_cleaned)
        terms.extend(korean_terms)
        
        return list(set(terms))
    
    def classify_query_type(self, query: str) -> str:
        """쿼리 타입 분류"""
        for query_type, pattern in self.query_patterns.items():
            if pattern.search(query):
                return query_type
        return 'general'
    
    def analyze_query(self, query: str) -> SearchContext:
        """쿼리 종합 분석"""
        terms = self.extract_query_terms(query)
        query_type = self.classify_query_type(query)
        
        # 기술 용어 분류
        tech_terms: Dict[str, List[str]] = {
            'api_terms': [],
            'method_terms': [],
            'concept_terms': []
        }
        
        for term in terms:
            if term in self.tech_synonyms or term.upper() in self.tech_synonyms:
                tech_terms['api_terms'].append(term)
            elif '()' in term or 'Client' in term or 'Listener' in term:
                tech_terms['method_terms'].append(term)
            else:
                tech_terms['concept_terms'].append(term)
        
        context_keywords = set(terms)
        
        return SearchContext(
            query=query,
            query_terms=terms,
            tech_terms=tech_terms,
            query_type=query_type,
            context_keywords=context_keywords
        )


class ContextScorer:
    """문맥 기반 점수 계산"""
    
    def __init__(self):
        # 섹션 타입별 가중치
        self.section_weights = {
            'api_spec': {'how_to': 0.8, 'what_is': 0.6, 'api_spec': 1.0, 'error_debug': 0.7},
            'code': {'how_to': 1.0, 'what_is': 0.5, 'api_spec': 0.8, 'error_debug': 0.9},
            'table': {'how_to': 0.7, 'what_is': 0.8, 'api_spec': 1.0, 'error_debug': 0.6},
            'text': {'how_to': 0.9, 'what_is': 1.0, 'api_spec': 0.7, 'error_debug': 0.8}
        }
        
        # 기술 용어 가중치
        self.term_weights = {
            'api_terms': 1.0,
            'method_terms': 0.9,
            'concept_terms': 0.7
        }
    
    def calculate_term_relevance(self, doc: Document, search_context: SearchContext) -> float:
        """용어 관련성 점수"""
        content = doc.page_content.lower()
        metadata = doc.metadata
        
        score = 0.0
        total_weight = 0.0
        
        # 쿼리 용어 매칭
        for term in search_context.query_terms:
            if term.lower() in content:
                score += 1.0
                total_weight += 1.0
        
        # 기술 용어 매칭 (가중치 적용)
        for term_type, terms in search_context.tech_terms.items():
            weight = self.term_weights.get(term_type, 0.5)
            for term in terms:
                if term.lower() in content:
                    score += weight
                    total_weight += weight
        
        # 메타데이터의 기술 용어 매칭
        doc_tech_terms = metadata.get('technical_terms', {})
        for term_type, terms in doc_tech_terms.items():
            for term in terms:
                if term.lower() in search_context.query.lower():
                    score += 0.5
                    total_weight += 0.5
        
        return score / max(total_weight, 1.0)
    
    def calculate_context_relevance(self, doc: Document, search_context: SearchContext) -> float:
        """문맥 관련성 점수"""
        metadata = doc.metadata
        content_types = metadata.get('content_types', ['text'])
        query_type = search_context.query_type
        
        # 컨텐츠 타입과 쿼리 타입 매칭
        type_score = 0.0
        for content_type in content_types:
            if content_type in self.section_weights:
                type_score += self.section_weights[content_type].get(query_type, 0.5)
        
        type_score = type_score / len(content_types)
        
        # 섹션 계층 관련성
        hierarchy = metadata.get('section_hierarchy', '')
        hierarchy_score = 0.0
        
        for keyword in search_context.context_keywords:
            if keyword.lower() in hierarchy.lower():
                hierarchy_score += 0.2
        
        hierarchy_score = min(hierarchy_score, 1.0)
        
        return (type_score + hierarchy_score) / 2.0
    
    def calculate_technical_accuracy(self, doc: Document, search_context: SearchContext) -> float:
        """기술적 정확성 점수"""
        metadata = doc.metadata
        content = doc.page_content.lower()
        
        # 코드 블록과 테이블의 가중치
        accuracy_score = 0.5  # 기본 점수
        
        if metadata.get('has_code'):
            accuracy_score += 0.2
        
        if metadata.get('has_tables'):
            accuracy_score += 0.15
        
        # API 관련 키워드 밀도
        api_keywords = ['api', 'sdk', 'client', 'params', 'response', 'request']
        keyword_count = sum(1 for kw in api_keywords if kw in content)
        accuracy_score += min(keyword_count * 0.05, 0.25)
        
        return min(accuracy_score, 1.0)


class NoiseFilter:
    """노이즈 필터링"""
    
    def __init__(self):
        # 일반적인 노이즈 용어들
        self.noise_terms = {
            'common_words': {'the', 'and', 'or', 'is', 'are', 'was', 'were', 'for', 'to', 'in', 'on'},
            'filler_korean': {'입니다', '있습니다', '합니다', '때문에', '그리고', '하지만', '또한'},
            'generic_tech': {'data', 'info', 'value', 'result', 'success', 'error'}
        }
    
    def calculate_noise_score(self, doc: Document, search_context: SearchContext) -> float:
        """노이즈 점수 계산 (낮을수록 좋음)"""
        content = doc.page_content.lower()
        content_length = len(content)
        
        if content_length == 0:
            return 1.0
        
        noise_count = 0
        total_terms = len(content.split())
        
        # 일반적인 노이즈 용어 카운트
        for noise_category, terms in self.noise_terms.items():
            for term in terms:
                noise_count += content.count(term.lower())
        
        # 반복되는 문구 패널티
        sentences = content.split('.')
        unique_sentences = set(sentences)
        repetition_penalty = 1 - (len(unique_sentences) / max(len(sentences), 1))
        
        # 노이즈 비율 계산
        noise_ratio = noise_count / max(total_terms, 1)
        total_noise_score = min((noise_ratio + repetition_penalty) / 2, 1.0)
        
        return total_noise_score
    
    def filter_by_relevance_threshold(self, docs: List[Document], 
                                    scores: List[float], 
                                    threshold: float = 0.3) -> Tuple[List[Document], List[float]]:
        """관련성 임계값으로 필터링"""
        filtered_docs = []
        filtered_scores = []
        
        for doc, score in zip(docs, scores):
            if score >= threshold:
                filtered_docs.append(doc)
                filtered_scores.append(score)
        
        return filtered_docs, filtered_scores


class ContextAwareRetriever(BaseRetriever):
    """컨텍스트 인식 검색기"""
    
    def __init__(self, 
                 base_retriever: BaseRetriever,
                 rerank_top_k: int = 20,
                 final_top_k: int = 5,
                 enable_query_expansion: bool = True,
                 noise_threshold: float = 0.7):
        
        self.base_retriever = base_retriever
        self.rerank_top_k = rerank_top_k
        self.final_top_k = final_top_k
        self.enable_query_expansion = enable_query_expansion
        self.noise_threshold = noise_threshold
        
        self.query_analyzer = QueryAnalyzer()
        self.context_scorer = ContextScorer()
        self.noise_filter = NoiseFilter()
    
    def _get_relevant_documents(self, 
                              query: str, 
                              *, 
                              run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """메인 검색 메소드"""
        
        # 1. 쿼리 분석
        search_context = self.query_analyzer.analyze_query(query)
        
        # 2. 쿼리 확장 (옵션)
        queries_to_search = [query]
        if self.enable_query_expansion:
            expanded_queries = self.query_analyzer.expand_query(query)
            queries_to_search.extend(expanded_queries[:3])  # 최대 3개 확장
        
        # 3. 기본 검색 수행
        all_docs = []
        for q in queries_to_search:
            docs = self.base_retriever.get_relevant_documents(q, run_manager=run_manager)
            all_docs.extend(docs)
        
        # 중복 제거 (page_content 기준)
        unique_docs = []
        seen_contents = set()
        for doc in all_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(content_hash)
        
        # 상위 K개만 리랭킹 대상으로
        docs_to_rerank = unique_docs[:self.rerank_top_k]
        
        # 4. 다단계 리랭킹
        reranked_docs = self._rerank_documents(docs_to_rerank, search_context)
        
        # 5. 최종 결과 반환
        return reranked_docs[:self.final_top_k]
    
    def _rerank_documents(self, docs: List[Document], search_context: SearchContext) -> List[Document]:
        """다단계 리랭킹"""
        
        scored_docs = []
        
        for doc in docs:
            # 점수 계산
            term_score = self.context_scorer.calculate_term_relevance(doc, search_context)
            context_score = self.context_scorer.calculate_context_relevance(doc, search_context)
            accuracy_score = self.context_scorer.calculate_technical_accuracy(doc, search_context)
            noise_score = self.noise_filter.calculate_noise_score(doc, search_context)
            
            # 종합 점수 (노이즈는 패널티)
            final_score = (term_score * 0.4 + 
                          context_score * 0.3 + 
                          accuracy_score * 0.2 + 
                          (1 - noise_score) * 0.1)
            
            scored_docs.append((doc, final_score))
        
        # 점수순 정렬
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 노이즈 임계값으로 필터링
        filtered_docs = []
        for doc, score in scored_docs:
            noise_score = self.noise_filter.calculate_noise_score(doc, search_context)
            if noise_score <= self.noise_threshold:
                filtered_docs.append(doc)
        
        return filtered_docs if filtered_docs else [doc for doc, _ in scored_docs]
    
    def get_search_analytics(self, query: str) -> Dict[str, Any]:
        """검색 분석 정보 반환 (디버깅용)"""
        search_context = self.query_analyzer.analyze_query(query)
        
        return {
            'query_terms': search_context.query_terms,
            'query_type': search_context.query_type,
            'tech_terms': search_context.tech_terms,
            'context_keywords': list(search_context.context_keywords)
        }


# 사용 예제
def create_context_aware_retriever_example():
    """컨텍스트 인식 검색기 생성 예제"""
    
    # 가상의 기본 검색기 (실제로는 FAISS, BM25 등을 사용)
    class MockRetriever(BaseRetriever):
        def __init__(self, documents):
            self.documents = documents
        
        def _get_relevant_documents(self, query, *, run_manager):
            # 간단한 키워드 매칭 검색
            results = []
            for doc in self.documents:
                if any(term.lower() in doc.page_content.lower() 
                      for term in query.split()):
                    results.append(doc)
            return results[:10]
    
    # 샘플 문서들
    sample_docs = [
        Document(
            page_content="PurchaseClient를 사용하여 인앱결제를 구현하는 방법",
            metadata={'content_types': ['text'], 'technical_terms': {'api_terms': ['PurchaseClient']}}
        ),
        Document(
            page_content="purchaseState 값으로 구매 상태를 확인할 수 있습니다",
            metadata={'content_types': ['text'], 'technical_terms': {'method_terms': ['purchaseState']}}
        )
    ]
    
    base_retriever = MockRetriever(sample_docs)
    context_retriever = ContextAwareRetriever(base_retriever)
    
    return context_retriever


# 테스트 함수
def test_retriever():
    """검색기 테스트"""
    retriever = create_context_aware_retriever_example()
    
    queries = [
        "PurchaseClient 사용법이 뭔가요?",
        "purchaseState 값은 무엇인가요?",
        "인앱결제 에러 해결 방법"
    ]
    
    for query in queries:
        print(f"쿼리: {query}")
        analytics = retriever.get_search_analytics(query)
        print(f"분석: {analytics}")
        
        # results = retriever.get_relevant_documents(query)
        # print(f"결과: {len(results)}개 문서")
        print("-" * 50)


if __name__ == "__main__":
    test_retriever()
