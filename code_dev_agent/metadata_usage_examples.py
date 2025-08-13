"""
메타데이터 활용 구체적 예시

이 스크립트는 메타데이터를 실제로 어떻게 활용할 수 있는지
구체적인 예시를 통해 보여줍니다.
"""

import os
import sys
from typing import List, Dict, Any
from langchain.docstore.document import Document

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metadata_utilization_guide import (
    MetadataAnalyzer, 
    MetadataBasedRetriever, 
    MetadataEnhancer, 
    MetadataVisualizer
)


class MetadataUsageExamples:
    """메타데이터 활용 예시 클래스"""
    
    def __init__(self):
        self.analyzer = MetadataAnalyzer()
        self.enhancer = MetadataEnhancer()
    
    def example_1_basic_metadata_analysis(self):
        """예시 1: 기본 메타데이터 분석"""
        print("🔍 예시 1: 기본 메타데이터 분석")
        print("=" * 50)
        
        # 샘플 문서 생성
        doc = Document(
            page_content="""
            | Element Name | Data Type | Description |
            | purchaseState | String | COMPLETED: 결제완료 / CANCELED: 취소 |
            | signature | String | 본 메시지에 대한 signature |
            """,
            metadata={
                'section_name': 'PNS Payment Notification 메시지 발송 규격',
                'content_type': 'message_specification',
                'contains_pns': True,
                'contains_purchasestate': True,
                'is_complete_spec': True,
                'chunk_size': 150
            }
        )
        
        # 메타데이터 분석
        analysis = self.analyzer.analyze_document_metadata(doc)
        
        print("📄 문서 정보:")
        MetadataVisualizer.print_document_metadata(doc)
        
        print("\n📊 분석 결과:")
        print(f"  - 내용 타입: {analysis['content_type']}")
        print(f"  - 우선순위: {analysis['priority_level']}")
        print(f"  - 완성도 점수: {analysis['completeness_score']:.2f}")
        print(f"  - 컨텍스트 관련성: {analysis['context_relevance']:.2f}")
        print(f"  - 부스트 팩터: {analysis['search_boost_factors']}")
        
        # 키워드 밀도 분석
        print(f"\n🔤 키워드 밀도:")
        for keyword, density in analysis['keyword_density'].items():
            if density > 0:
                print(f"  - {keyword}: {density:.4f}")
    
    def example_2_metadata_based_search(self):
        """예시 2: 메타데이터 기반 검색"""
        print("\n🔍 예시 2: 메타데이터 기반 검색")
        print("=" * 50)
        
        # 다양한 문서 생성
        documents = [
            Document(
                page_content="PNS 메시지의 purchaseState 필드는 COMPLETED 또는 CANCELED 값을 가집니다.",
                metadata={
                    'section_name': 'PNS 개요',
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
            ),
            Document(
                page_content="signature 검증을 위해서는 PublicKey를 사용하여 메시지의 무결성을 확인합니다.",
                metadata={
                    'section_name': '서명 검증',
                    'content_type': 'signature_verification',
                    'contains_pns': False,
                    'contains_purchasestate': False,
                    'is_complete_spec': False
                }
            )
        ]
        
        # 메타데이터 기반 검색기 생성
        retriever = MetadataBasedRetriever(documents)
        
        # 다양한 질의로 테스트
        test_queries = [
            "PNS 메시지의 purchaseState 값은 무엇이 있나요?",
            "signature 검증은 어떻게 하나요?",
            "PNS 메시지 규격을 알려주세요"
        ]
        
        for query in test_queries:
            print(f"\n❓ 질의: {query}")
            
            # 검색 기준 설정
            search_criteria = {
                'boost_factors': ['purchase_state_related', 'complete_specification', 'pns_related']
            }
            
            # 메타데이터 기반 검색
            results = retriever.search_by_metadata(query, search_criteria)
            
            print("📋 검색 결과:")
            for i, (score, doc) in enumerate(results[:3], 1):
                print(f"  {i}. 점수: {score:.2f}")
                print(f"     내용: {doc.page_content[:60]}...")
                print(f"     타입: {doc.metadata.get('content_type', 'N/A')}")
    
    def example_3_metadata_enhancement(self):
        """예시 3: 메타데이터 강화"""
        print("\n🔧 예시 3: 메타데이터 강화")
        print("=" * 50)
        
        # 기본 문서
        original_doc = Document(
            page_content="PNS 메시지 처리 방법에 대한 설명입니다.",
            metadata={
                'section_name': 'PNS 처리',
                'contains_pns': True
            }
        )
        
        print("📄 원본 메타데이터:")
        MetadataVisualizer.print_document_metadata(original_doc)
        
        # 메타데이터 강화
        additional_info = {
            'content_type': 'general_pns',
            'priority_level': 'medium',
            'processing_method': 'notification_handler',
            'estimated_complexity': 'intermediate',
            'related_topics': ['message_processing', 'error_handling'],
            'last_updated': '2024-01-15'
        }
        
        enhanced_doc = self.enhancer.enhance_document_metadata(original_doc, additional_info)
        
        print("\n📄 강화된 메타데이터:")
        MetadataVisualizer.print_document_metadata(enhanced_doc, show_content=True)
        
        # 검색용 메타데이터 생성
        query = "PNS 메시지의 purchaseState 값은 무엇이 있나요?"
        search_metadata = self.enhancer.create_search_metadata(query)
        
        print(f"\n🔍 질의 메타데이터: {query}")
        print(f"  - 질의 타입: {search_metadata['query_type']}")
        print(f"  - 목표 키워드: {search_metadata['target_keywords']}")
        print(f"  - 부스트 팩터: {search_metadata['boost_factors']}")
        print(f"  - 우선순위: {search_metadata['priority_level']}")
    
    def example_4_advanced_metadata_filtering(self):
        """예시 4: 고급 메타데이터 필터링"""
        print("\n🎯 예시 4: 고급 메타데이터 필터링")
        print("=" * 50)
        
        # 다양한 문서 생성
        documents = [
            Document(
                page_content="PNS Payment Notification 메시지 규격",
                metadata={
                    'content_type': 'message_specification',
                    'contains_pns': True,
                    'contains_purchasestate': True,
                    'is_complete_spec': True,
                    'section_name': 'PNS 메시지 규격',
                    'priority': 'high'
                }
            ),
            Document(
                page_content="일반적인 API 사용법",
                metadata={
                    'content_type': 'general_api',
                    'contains_pns': False,
                    'contains_purchasestate': False,
                    'is_complete_spec': False,
                    'section_name': 'API 가이드',
                    'priority': 'low'
                }
            ),
            Document(
                page_content="purchaseState 필드 설명",
                metadata={
                    'content_type': 'purchase_state_info',
                    'contains_pns': True,
                    'contains_purchasestate': True,
                    'is_complete_spec': False,
                    'section_name': '필드 설명',
                    'priority': 'medium'
                }
            )
        ]
        
        # 고급 필터링 함수들
        def filter_by_content_type(docs: List[Document], content_type: str) -> List[Document]:
            """내용 타입별 필터링"""
            return [doc for doc in docs if doc.metadata.get('content_type') == content_type]
        
        def filter_by_priority(docs: List[Document], min_priority: str) -> List[Document]:
            """우선순위별 필터링"""
            priority_order = {'low': 1, 'medium': 2, 'high': 3}
            min_priority_level = priority_order.get(min_priority, 1)
            
            return [doc for doc in docs 
                   if priority_order.get(doc.metadata.get('priority', 'low'), 1) >= min_priority_level]
        
        def filter_by_completeness(docs: List[Document], require_complete: bool = True) -> List[Document]:
            """완성도별 필터링"""
            return [doc for doc in docs if doc.metadata.get('is_complete_spec', False) == require_complete]
        
        # 필터링 예시
        print("📋 전체 문서:")
        for i, doc in enumerate(documents, 1):
            print(f"  {i}. {doc.metadata.get('content_type')} - {doc.metadata.get('priority')}")
        
        print(f"\n🎯 메시지 규격만 필터링:")
        message_specs = filter_by_content_type(documents, 'message_specification')
        for doc in message_specs:
            print(f"  - {doc.metadata.get('section_name')}")
        
        print(f"\n🎯 높은 우선순위만 필터링:")
        high_priority = filter_by_priority(documents, 'high')
        for doc in high_priority:
            print(f"  - {doc.metadata.get('section_name')} ({doc.metadata.get('priority')})")
        
        print(f"\n🎯 완전한 메시지 규격만 필터링:")
        complete_specs = filter_by_completeness(documents, True)
        for doc in complete_specs:
            print(f"  - {doc.metadata.get('section_name')}")
    
    def example_5_metadata_statistics(self):
        """예시 5: 메타데이터 통계"""
        print("\n📊 예시 5: 메타데이터 통계")
        print("=" * 50)
        
        # 샘플 문서 생성
        documents = [
            Document(page_content="PNS 메시지 1", metadata={'content_type': 'message_specification', 'contains_pns': True, 'is_complete_spec': True}),
            Document(page_content="PNS 메시지 2", metadata={'content_type': 'message_specification', 'contains_pns': True, 'is_complete_spec': True}),
            Document(page_content="일반 API", metadata={'content_type': 'general_api', 'contains_pns': False, 'is_complete_spec': False}),
            Document(page_content="purchaseState 정보", metadata={'content_type': 'purchase_state_info', 'contains_pns': True, 'is_complete_spec': False}),
            Document(page_content="signature 검증", metadata={'content_type': 'signature_verification', 'contains_pns': False, 'is_complete_spec': False})
        ]
        
        # 통계 계산
        total_docs = len(documents)
        pns_docs = sum(1 for doc in documents if doc.metadata.get('contains_pns', False))
        complete_specs = sum(1 for doc in documents if doc.metadata.get('is_complete_spec', False))
        
        # 내용 타입별 통계
        content_type_stats = {}
        for doc in documents:
            content_type = doc.metadata.get('content_type', 'unknown')
            content_type_stats[content_type] = content_type_stats.get(content_type, 0) + 1
        
        print("📈 메타데이터 통계:")
        print(f"  - 총 문서 수: {total_docs}")
        print(f"  - PNS 관련 문서: {pns_docs} ({pns_docs/total_docs*100:.1f}%)")
        print(f"  - 완전한 메시지 규격: {complete_specs} ({complete_specs/total_docs*100:.1f}%)")
        
        print(f"\n📂 내용 타입별 분포:")
        for content_type, count in content_type_stats.items():
            percentage = count / total_docs * 100
            print(f"  - {content_type}: {count}개 ({percentage:.1f}%)")
        
        # 메타데이터 시각화 사용
        print(f"\n📊 시각화 결과:")
        MetadataVisualizer.print_metadata_summary(documents)


def run_all_examples():
    """모든 예시 실행"""
    examples = MetadataUsageExamples()
    
    print("🚀 메타데이터 활용 구체적 예시")
    print("=" * 60)
    
    examples.example_1_basic_metadata_analysis()
    examples.example_2_metadata_based_search()
    examples.example_3_metadata_enhancement()
    examples.example_4_advanced_metadata_filtering()
    examples.example_5_metadata_statistics()
    
    print("\n✅ 모든 예시 완료!")


if __name__ == "__main__":
    run_all_examples()
