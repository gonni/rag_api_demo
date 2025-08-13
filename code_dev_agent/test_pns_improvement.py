"""
PNS 질의 개선 테스트 스크립트

이 스크립트는 새로운 PNS 계층적 분할 및 컨텍스트 인식 RAG 시스템의
성능을 테스트하고 기존 시스템과 비교합니다.
"""

import os
import sys
from typing import List, Dict, Any
from langchain.docstore.document import Document

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code_dev'))

from hierarchical_pns_splitter import PNSHierarchicalSplitter, PNSContextualRetriever
from context_aware_rag import ContextAwareRAG, PNSQueryAnalyzer


class PNSImprovementTester:
    """PNS 개선 테스트 클래스"""
    
    def __init__(self, document_path: str = "data/dev_center_guide_allmd_touched.md"):
        self.document_path = document_path
        self.original_docs: List[Document] = []
        self.improved_docs: List[Document] = []
        
    def run_comparison_test(self):
        """개선 전후 비교 테스트"""
        print("🚀 PNS 질의 개선 테스트 시작")
        print("=" * 60)
        
        # 1. 기존 방식으로 문서 분할
        print("1️⃣ 기존 방식 문서 분할...")
        self._load_original_documents()
        
        # 2. 개선된 방식으로 문서 분할
        print("\n2️⃣ 개선된 PNS 계층적 분할...")
        self._load_improved_documents()
        
        # 3. 테스트 질의 정의
        test_queries = [
            "PNS(Payment Notification Service)는 무엇이고 메세지 규격은 어떻게 됩니까?",
            "PNS 메시지의 purchaseState 값은 무엇이 있나요?",
            "PNS 메시지에서 signature 검증은 어떻게 하나요?",
            "PNS Payment Notification 메시지의 구성 요소는 무엇인가요?"
        ]
        
        # 4. 각 질의별 성능 비교
        print("\n3️⃣ 질의별 성능 비교...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- 테스트 질의 {i}: {query} ---")
            self._compare_query_performance(query)
        
        # 5. 종합 분석
        print("\n4️⃣ 종합 분석...")
        self._generate_summary()
    
    def _load_original_documents(self):
        """기존 방식 문서 로드"""
        from code_dev.optimal_rag_pipeline import OptimalDocumentSplitter
        
        splitter = OptimalDocumentSplitter(self.document_path)
        self.original_docs = splitter.split_documents()
        
        # PNS 관련 문서 수 확인
        pns_count = sum(1 for doc in self.original_docs if doc.metadata.get('contains_pns', False))
        print(f"  기존 방식: 총 {len(self.original_docs)}개 문서, PNS 관련 {pns_count}개")
    
    def _load_improved_documents(self):
        """개선된 방식 문서 로드"""
        splitter = PNSHierarchicalSplitter(self.document_path)
        self.improved_docs = splitter.split_documents()
        
        # 완전한 메시지 규격 문서 수 확인
        complete_specs = sum(1 for doc in self.improved_docs if doc.metadata.get('is_complete_spec', False))
        print(f"  개선된 방식: 총 {len(self.improved_docs)}개 문서, 완전한 메시지 규격 {complete_specs}개")
    
    def _compare_query_performance(self, query: str):
        """질의 성능 비교"""
        # 기존 방식 테스트
        print("  📊 기존 방식 결과:")
        original_result = self._test_original_approach(query)
        
        # 개선된 방식 테스트
        print("  📊 개선된 방식 결과:")
        improved_result = self._test_improved_approach(query)
        
        # 성능 비교
        self._analyze_performance_difference(original_result, improved_result, query)
    
    def _test_original_approach(self, query: str) -> Dict[str, Any]:
        """기존 방식 테스트"""
        try:
            # 기존 SmartRetriever 사용
            from code_common.common_rag_util import SmartRetriever
            
            retriever = SmartRetriever(documents=self.original_docs)
            retriever.build_retrievers()
            
            results = retriever.get_retriever().invoke(query)
            
            # 결과 분석
            pns_docs = [doc for doc in results if doc.metadata.get('contains_pns', False)]
            complete_specs = [doc for doc in results if doc.metadata.get('is_complete_spec', False)]
            
            print(f"    - 검색된 문서: {len(results)}개")
            print(f"    - PNS 관련: {len(pns_docs)}개")
            print(f"    - 완전한 메시지 규격: {len(complete_specs)}개")
            
            return {
                'total_docs': len(results),
                'pns_docs': len(pns_docs),
                'complete_specs': len(complete_specs),
                'documents': results
            }
            
        except Exception as e:
            print(f"    ❌ 오류: {str(e)}")
            return {'error': str(e)}
    
    def _test_improved_approach(self, query: str) -> Dict[str, Any]:
        """개선된 방식 테스트"""
        try:
            # 컨텍스트 인식 RAG 사용
            context_rag = ContextAwareRAG(documents=self.improved_docs)
            context_rag.setup()
            
            result = context_rag.query(query)
            
            # 결과 분석
            relevant_docs = result['relevant_docs']
            pns_docs = [doc for doc in relevant_docs if doc.metadata.get('contains_pns', False)]
            complete_specs = [doc for doc in relevant_docs if doc.metadata.get('is_complete_spec', False)]
            
            print(f"    - 검색된 문서: {len(relevant_docs)}개")
            print(f"    - PNS 관련: {len(pns_docs)}개")
            print(f"    - 완전한 메시지 규격: {len(complete_specs)}개")
            print(f"    - 컨텍스트 타입: {result['context_type']}")
            
            return {
                'total_docs': len(relevant_docs),
                'pns_docs': len(pns_docs),
                'complete_specs': len(complete_specs),
                'context_type': result['context_type'],
                'answer': result['answer'],
                'documents': relevant_docs
            }
            
        except Exception as e:
            print(f"    ❌ 오류: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_performance_difference(self, original: Dict, improved: Dict, query: str):
        """성능 차이 분석"""
        if 'error' in original or 'error' in improved:
            print("    ⚠️  오류로 인해 비교 불가")
            return
        
        print("  📈 성능 개선 분석:")
        
        # 완전한 메시지 규격 문서 개선도
        original_specs = original.get('complete_specs', 0)
        improved_specs = improved.get('complete_specs', 0)
        
        if original_specs == 0 and improved_specs > 0:
            print(f"    ✅ 완전한 메시지 규격 문서: 0개 → {improved_specs}개 (무한대 개선)")
        elif original_specs > 0:
            improvement = (improved_specs - original_specs) / original_specs * 100
            print(f"    📊 완전한 메시지 규격 문서: {original_specs}개 → {improved_specs}개 ({improvement:+.1f}%)")
        
        # PNS 관련 문서 개선도
        original_pns = original.get('pns_docs', 0)
        improved_pns = improved.get('pns_docs', 0)
        
        if original_pns > 0:
            pns_improvement = (improved_pns - original_pns) / original_pns * 100
            print(f"    📊 PNS 관련 문서: {original_pns}개 → {improved_pns}개 ({pns_improvement:+.1f}%)")
        
        # 컨텍스트 인식 결과
        if 'context_type' in improved:
            print(f"    🎯 컨텍스트 타입 인식: {improved['context_type']}")
        
        # 답변 품질 (간단한 키워드 기반 평가)
        if 'answer' in improved:
            answer_quality = self._evaluate_answer_quality(improved['answer'], query)
            print(f"    📝 답변 품질 점수: {answer_quality}/10")
    
    def _evaluate_answer_quality(self, answer: str, query: str) -> float:
        """답변 품질 평가 (간단한 키워드 기반)"""
        score = 0.0
        answer_lower = answer.lower()
        query_lower = query.lower()
        
        # PNS 관련 키워드
        if 'pns' in answer_lower or 'payment notification' in answer_lower:
            score += 2
        
        # 메시지 규격 관련
        if '메시지' in answer or 'message' in answer_lower:
            score += 2
        
        # purchaseState 관련
        if 'purchasestate' in query_lower and 'purchasestate' in answer_lower:
            score += 3
        
        # 구체적인 정보 포함
        if '|' in answer or 'table' in answer_lower or '요소' in answer:
            score += 2
        
        # 한국어 답변
        if any(char in answer for char in ['는', '은', '이', '가', '을', '를']):
            score += 1
        
        return min(score, 10.0)
    
    def _generate_summary(self):
        """종합 분석 결과"""
        print("\n" + "=" * 60)
        print("📋 종합 분석 결과")
        print("=" * 60)
        
        # 문서 분할 개선도
        original_pns = sum(1 for doc in self.original_docs if doc.metadata.get('contains_pns', False))
        improved_complete = sum(1 for doc in self.improved_docs if doc.metadata.get('is_complete_spec', False))
        
        print(f"📊 문서 분할 개선:")
        print(f"  - 기존 PNS 관련 문서: {original_pns}개")
        print(f"  - 개선된 완전한 메시지 규격: {improved_complete}개")
        
        if original_pns == 0:
            print("  ✅ 완전한 메시지 규격 문서 생성으로 컨텍스트 손실 해결")
        else:
            improvement = improved_complete / original_pns * 100
            print(f"  📈 완전한 메시지 규격 문서 비율: {improvement:.1f}%")
        
        print(f"\n🎯 주요 개선 사항:")
        print(f"  1. 계층적 문서 분할로 PNS 섹션 완전성 보장")
        print(f"  2. 메시지 규격 테이블을 하나의 문서로 유지")
        print(f"  3. 컨텍스트 인식 검색으로 관련성 향상")
        print(f"  4. 질의 타입별 최적화된 검색 전략")
        
        print(f"\n💡 권장사항:")
        print(f"  - PNS 관련 질의에는 개선된 방식을 사용")
        print(f"  - 메시지 규격 테이블은 분할하지 않고 통합 유지")
        print(f"  - 컨텍스트 타입별 우선순위 적용")


def main():
    """메인 실행 함수"""
    tester = PNSImprovementTester()
    tester.run_comparison_test()


if __name__ == "__main__":
    main()
