"""
메타데이터 활용 가이드 및 유틸리티

이 모듈은 RAG 시스템에서 메타데이터를 효과적으로 활용하는 방법을
제시하고, 구체적인 구현 예시를 제공합니다.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from langchain.docstore.document import Document
from dataclasses import dataclass
from enum import Enum


class ContentType(Enum):
    """문서 내용 타입"""
    MESSAGE_SPECIFICATION = "message_specification"
    PURCHASE_STATE_INFO = "purchase_state_info"
    SIGNATURE_VERIFICATION = "signature_verification"
    GENERAL_PNS = "general_pns"
    CODE_EXAMPLE = "code_example"
    ERROR_HANDLING = "error_handling"


class PriorityLevel(Enum):
    """우선순위 레벨"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class MetadataScore:
    """메타데이터 점수"""
    relevance_score: float
    completeness_score: float
    context_score: float
    total_score: float


class MetadataAnalyzer:
    """메타데이터 분석기"""
    
    def __init__(self):
        self.keyword_patterns = {
            'pns': [r'\bPNS\b', r'Payment Notification', r'결제알림'],
            'purchasestate': [r'\bpurchaseState\b', r'purchase.*?state', r'결제.*?상태'],
            'signature': [r'\bsignature\b', r'서명', r'검증'],
            'message': [r'\bmessage\b', r'메시지', r'규격'],
            'api': [r'\bAPI\b', r'endpoint', r'요청'],
            'error': [r'\berror\b', r'오류', r'에러', r'exception']
        }
    
    def analyze_document_metadata(self, doc: Document) -> Dict[str, Any]:
        """문서 메타데이터 분석"""
        content = doc.page_content
        metadata = doc.metadata
        
        analysis = {
            'content_type': self._determine_content_type(content, metadata),
            'priority_level': self._determine_priority_level(metadata),
            'keyword_density': self._calculate_keyword_density(content),
            'completeness_score': self._calculate_completeness_score(metadata),
            'context_relevance': self._calculate_context_relevance(content, metadata),
            'search_boost_factors': self._identify_boost_factors(metadata)
        }
        
        return analysis
    
    def _determine_content_type(self, content: str, metadata: Dict) -> str:
        """내용 타입 결정"""
        content_lower = content.lower()
        
        # 메타데이터에서 확인
        if metadata.get('is_complete_spec', False):
            return ContentType.MESSAGE_SPECIFICATION.value
        
        # 내용 기반 판단
        if 'purchasestate' in content_lower:
            return ContentType.PURCHASE_STATE_INFO.value
        elif 'signature' in content_lower:
            return ContentType.SIGNATURE_VERIFICATION.value
        elif '```' in content or 'code' in content_lower:
            return ContentType.CODE_EXAMPLE.value
        elif 'error' in content_lower or 'exception' in content_lower:
            return ContentType.ERROR_HANDLING.value
        else:
            return ContentType.GENERAL_PNS.value
    
    def _determine_priority_level(self, metadata: Dict) -> str:
        """우선순위 레벨 결정"""
        if metadata.get('is_complete_spec', False):
            return PriorityLevel.HIGH.value
        elif metadata.get('contains_purchasestate', False):
            return PriorityLevel.HIGH.value
        elif metadata.get('contains_pns', False):
            return PriorityLevel.MEDIUM.value
        else:
            return PriorityLevel.LOW.value
    
    def _calculate_keyword_density(self, content: str) -> Dict[str, float]:
        """키워드 밀도 계산"""
        content_lower = content.lower()
        total_words = len(content.split())
        
        keyword_density = {}
        for keyword, patterns in self.keyword_patterns.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, content_lower, re.IGNORECASE))
            keyword_density[keyword] = count / total_words if total_words > 0 else 0
        
        return keyword_density
    
    def _calculate_completeness_score(self, metadata: Dict) -> float:
        """완성도 점수 계산"""
        score = 0.0
        
        # 완전한 메시지 규격
        if metadata.get('is_complete_spec', False):
            score += 1.0
        
        # PNS 관련성
        if metadata.get('contains_pns', False):
            score += 0.3
        
        # purchaseState 포함
        if metadata.get('contains_purchasestate', False):
            score += 0.4
        
        # 섹션 정보
        if metadata.get('section_name'):
            score += 0.2
        
        # 테이블 정보
        if metadata.get('table_name'):
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_context_relevance(self, content: str, metadata: Dict) -> float:
        """컨텍스트 관련성 점수"""
        score = 0.0
        
        # 계층 정보
        if metadata.get('title_hierarchy'):
            hierarchy = metadata['title_hierarchy']
            if 'PNS' in hierarchy:
                score += 0.4
            if '메시지' in hierarchy or 'message' in hierarchy.lower():
                score += 0.3
        
        # 섹션 정보
        if metadata.get('section_name'):
            section = metadata['section_name']
            if 'PNS' in section:
                score += 0.3
        
        return min(score, 1.0)
    
    def _identify_boost_factors(self, metadata: Dict) -> List[str]:
        """부스트 팩터 식별"""
        boost_factors = []
        
        if metadata.get('is_complete_spec', False):
            boost_factors.append('complete_specification')
        
        if metadata.get('contains_purchasestate', False):
            boost_factors.append('purchase_state_related')
        
        if metadata.get('contains_pns', False):
            boost_factors.append('pns_related')
        
        if metadata.get('content_type') == 'message_specification':
            boost_factors.append('message_specification')
        
        return boost_factors


class MetadataBasedRetriever:
    """메타데이터 기반 검색기"""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.metadata_analyzer = MetadataAnalyzer()
        self.document_analyses = self._analyze_all_documents()
    
    def _analyze_all_documents(self) -> Dict[str, Dict[str, Any]]:
        """모든 문서 메타데이터 분석"""
        analyses = {}
        for i, doc in enumerate(self.documents):
            doc_id = f"doc_{i}"
            analyses[doc_id] = {
                'document': doc,
                'analysis': self.metadata_analyzer.analyze_document_metadata(doc)
            }
        return analyses
    
    def search_by_metadata(self, query: str, search_criteria: Dict[str, Any]) -> List[Tuple[float, Document]]:
        """메타데이터 기반 검색"""
        query_analysis = self._analyze_query(query)
        
        scored_docs = []
        for doc_id, doc_info in self.document_analyses.items():
            doc = doc_info['document']
            analysis = doc_info['analysis']
            
            # 메타데이터 매칭 점수 계산
            metadata_score = self._calculate_metadata_matching_score(
                query_analysis, analysis, search_criteria
            )
            
            if metadata_score > 0:
                scored_docs.append((metadata_score, doc))
        
        # 점수순 정렬
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return scored_docs
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """질의 분석"""
        query_lower = query.lower()
        
        analysis = {
            'target_content_type': self._identify_target_content_type(query_lower),
            'required_keywords': self._extract_required_keywords(query_lower),
            'priority_level': self._determine_query_priority(query_lower)
        }
        
        return analysis
    
    def _identify_target_content_type(self, query: str) -> str:
        """목표 내용 타입 식별"""
        if 'purchasestate' in query or 'purchase state' in query:
            return ContentType.PURCHASE_STATE_INFO.value
        elif 'signature' in query:
            return ContentType.SIGNATURE_VERIFICATION.value
        elif 'message' in query or '메시지' in query:
            return ContentType.MESSAGE_SPECIFICATION.value
        elif 'code' in query or '예제' in query:
            return ContentType.CODE_EXAMPLE.value
        else:
            return ContentType.GENERAL_PNS.value
    
    def _extract_required_keywords(self, query: str) -> List[str]:
        """필요한 키워드 추출"""
        required_keywords = []
        
        for keyword, patterns in self.metadata_analyzer.keyword_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    required_keywords.append(keyword)
                    break
        
        return list(set(required_keywords))
    
    def _determine_query_priority(self, query: str) -> str:
        """질의 우선순위 결정"""
        if 'purchasestate' in query or 'signature' in query:
            return PriorityLevel.HIGH.value
        elif 'message' in query or '메시지' in query:
            return PriorityLevel.MEDIUM.value
        else:
            return PriorityLevel.LOW.value
    
    def _calculate_metadata_matching_score(
        self, 
        query_analysis: Dict[str, Any], 
        doc_analysis: Dict[str, Any], 
        search_criteria: Dict[str, Any]
    ) -> float:
        """메타데이터 매칭 점수 계산"""
        score = 0.0
        
        # 1. 내용 타입 매칭
        if query_analysis['target_content_type'] == doc_analysis['content_type']:
            score += 2.0
        
        # 2. 우선순위 매칭
        if query_analysis['priority_level'] == doc_analysis['priority_level']:
            score += 1.5
        
        # 3. 키워드 매칭
        for keyword in query_analysis['required_keywords']:
            if keyword in doc_analysis['keyword_density']:
                density = doc_analysis['keyword_density'][keyword]
                score += density * 10  # 키워드 밀도에 비례한 점수
        
        # 4. 완성도 점수
        score += doc_analysis['completeness_score'] * 2
        
        # 5. 컨텍스트 관련성
        score += doc_analysis['context_relevance'] * 1.5
        
        # 6. 부스트 팩터 적용
        for boost_factor in doc_analysis['search_boost_factors']:
            if boost_factor in search_criteria.get('boost_factors', []):
                score *= 1.5
        
        return score


class MetadataEnhancer:
    """메타데이터 강화기"""
    
    @staticmethod
    def enhance_document_metadata(doc: Document, additional_info: Dict[str, Any]) -> Document:
        """문서 메타데이터 강화"""
        enhanced_metadata = doc.metadata.copy()
        
        # 추가 정보 병합
        enhanced_metadata.update(additional_info)
        
        # 계산된 필드 추가
        enhanced_metadata['enhanced_at'] = 'metadata_enhancer'
        enhanced_metadata['total_metadata_fields'] = len(enhanced_metadata)
        
        return Document(
            page_content=doc.page_content,
            metadata=enhanced_metadata
        )
    
    @staticmethod
    def create_search_metadata(query: str) -> Dict[str, Any]:
        """검색용 메타데이터 생성"""
        query_lower = query.lower()
        
        search_metadata = {
            'query_type': 'unknown',
            'target_keywords': [],
            'boost_factors': [],
            'priority_level': 'low'
        }
        
        # 질의 타입 식별
        if 'purchasestate' in query_lower:
            search_metadata['query_type'] = 'purchase_state'
            search_metadata['target_keywords'].append('purchasestate')
            search_metadata['boost_factors'].append('purchase_state_related')
            search_metadata['priority_level'] = 'high'
        
        elif 'signature' in query_lower:
            search_metadata['query_type'] = 'signature_verification'
            search_metadata['target_keywords'].append('signature')
            search_metadata['boost_factors'].append('signature_related')
            search_metadata['priority_level'] = 'high'
        
        elif 'message' in query_lower or '메시지' in query:
            search_metadata['query_type'] = 'message_specification'
            search_metadata['target_keywords'].extend(['message', '메시지'])
            search_metadata['boost_factors'].append('message_specification')
            search_metadata['priority_level'] = 'medium'
        
        return search_metadata


class MetadataVisualizer:
    """메타데이터 시각화기"""
    
    @staticmethod
    def print_metadata_summary(documents: List[Document]):
        """메타데이터 요약 출력"""
        print("📊 메타데이터 분석 요약")
        print("=" * 50)
        
        # 통계 계산
        total_docs = len(documents)
        pns_docs = sum(1 for doc in documents if doc.metadata.get('contains_pns', False))
        complete_specs = sum(1 for doc in documents if doc.metadata.get('is_complete_spec', False))
        purchase_state_docs = sum(1 for doc in documents if doc.metadata.get('contains_purchasestate', False))
        
        print(f"📄 총 문서 수: {total_docs}")
        print(f"🔗 PNS 관련 문서: {pns_docs} ({pns_docs/total_docs*100:.1f}%)")
        print(f"📋 완전한 메시지 규격: {complete_specs} ({complete_specs/total_docs*100:.1f}%)")
        print(f"💰 purchaseState 포함: {purchase_state_docs} ({purchase_state_docs/total_docs*100:.1f}%)")
        
        # 내용 타입별 분포
        content_types = {}
        for doc in documents:
            content_type = doc.metadata.get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        print(f"\n📂 내용 타입별 분포:")
        for content_type, count in content_types.items():
            print(f"  - {content_type}: {count}개 ({count/total_docs*100:.1f}%)")
    
    @staticmethod
    def print_document_metadata(doc: Document, show_content: bool = False):
        """개별 문서 메타데이터 출력"""
        print(f"📄 문서 메타데이터:")
        print(f"  - 섹션: {doc.metadata.get('section_name', 'N/A')}")
        print(f"  - 내용 타입: {doc.metadata.get('content_type', 'N/A')}")
        print(f"  - PNS 관련: {doc.metadata.get('contains_pns', False)}")
        print(f"  - purchaseState 포함: {doc.metadata.get('contains_purchasestate', False)}")
        print(f"  - 완전한 메시지 규격: {doc.metadata.get('is_complete_spec', False)}")
        print(f"  - 청크 크기: {doc.metadata.get('chunk_size', 'N/A')}")
        
        if show_content:
            print(f"  - 내용 미리보기: {doc.page_content[:100]}...")


# 사용 예시
def demonstrate_metadata_utilization():
    """메타데이터 활용 데모"""
    print("🚀 메타데이터 활용 데모")
    print("=" * 50)
    
    # 샘플 문서 생성
    sample_docs = [
        Document(
            page_content="PNS Payment Notification 메시지의 purchaseState 필드는 COMPLETED 또는 CANCELED 값을 가집니다.",
            metadata={
                'section_name': 'PNS 메시지 규격',
                'content_type': 'purchase_state_info',
                'contains_pns': True,
                'contains_purchasestate': True,
                'is_complete_spec': False
            }
        ),
        Document(
            page_content="| Element Name | Data Type | Description |\n| purchaseState | String | COMPLETED: 결제완료 / CANCELED: 취소 |",
            metadata={
                'section_name': 'PNS 메시지 규격',
                'content_type': 'message_specification',
                'contains_pns': True,
                'contains_purchasestate': True,
                'is_complete_spec': True
            }
        )
    ]
    
    # 메타데이터 분석기 사용
    analyzer = MetadataAnalyzer()
    for i, doc in enumerate(sample_docs):
        print(f"\n📊 문서 {i+1} 분석:")
        analysis = analyzer.analyze_document_metadata(doc)
        print(f"  - 내용 타입: {analysis['content_type']}")
        print(f"  - 우선순위: {analysis['priority_level']}")
        print(f"  - 완성도 점수: {analysis['completeness_score']:.2f}")
        print(f"  - 부스트 팩터: {analysis['search_boost_factors']}")
    
    # 메타데이터 기반 검색기 사용
    retriever = MetadataBasedRetriever(sample_docs)
    query = "PNS 메시지의 purchaseState 값은 무엇이 있나요?"
    
    print(f"\n🔍 질의: {query}")
    search_criteria = {'boost_factors': ['purchase_state_related', 'complete_specification']}
    results = retriever.search_by_metadata(query, search_criteria)
    
    for score, doc in results:
        print(f"  - 점수: {score:.2f}, 내용: {doc.page_content[:50]}...")
    
    # 메타데이터 시각화
    print(f"\n📊 메타데이터 요약:")
    MetadataVisualizer.print_metadata_summary(sample_docs)


if __name__ == "__main__":
    demonstrate_metadata_utilization()
