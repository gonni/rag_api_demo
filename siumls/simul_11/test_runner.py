"""
RAG 검색 최적화 테스트 실행기 (GPU 머신용)

이 스크립트는 실제 실행 전에 코드 검증을 위한 테스트입니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from document_splitting_strategies import (
    KeywordBasedSplitter, 
    SemanticBasedSplitter, 
    HybridSplitter,
    SmartRetriever,
    RAGExperimentRunner
)


def test_document_loading():
    """문서 로딩 테스트"""
    print("📄 문서 로딩 테스트...")
    
    document_path = "../data/dev_center_guide_allmd_touched.md"
    
    if not os.path.exists(document_path):
        print(f"❌ 문서 파일이 없습니다: {document_path}")
        return False
    
    try:
        with open(document_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✅ 문서 로딩 성공 (크기: {len(content):,} 자)")
        print(f"📊 예상 분할 수: ~{len(content) // 800} 청크")
        return True
        
    except Exception as e:
        print(f"❌ 문서 로딩 실패: {e}")
        return False


def test_splitting_strategies():
    """분할 전략 테스트"""
    print("\n🔧 분할 전략 테스트...")
    
    document_path = "../data/dev_center_guide_allmd_touched.md"
    
    strategies = {
        'keyword': KeywordBasedSplitter(document_path),
        'semantic': SemanticBasedSplitter(document_path),
        'hybrid': HybridSplitter(document_path)
    }
    
    results = {}
    
    for name, splitter in strategies.items():
        try:
            print(f"  🧪 {name} 전략 테스트...")
            
            # 작은 샘플로 테스트
            original_text = splitter.raw_text
            splitter.raw_text = original_text[:5000]  # 처음 5000자만 테스트
            
            documents = splitter.split_documents()
            results[name] = len(documents)
            
            print(f"    ✅ {name}: {len(documents)}개 문서 생성")
            
            # 첫 번째 문서 샘플 출력
            if documents:
                first_doc = documents[0]
                print(f"    📝 샘플: {first_doc.page_content[:100]}...")
                print(f"    🏷️  메타데이터: {list(first_doc.metadata.keys())}")
            
            # 원본 텍스트 복원
            splitter.raw_text = original_text
            
        except Exception as e:
            print(f"    ❌ {name} 전략 실패: {e}")
            results[name] = 0
    
    return results


def test_query_processing():
    """쿼리 처리 테스트"""
    print("\n🔍 쿼리 처리 테스트...")
    
    test_queries = [
        "PNS 메시지 서버 규격의 purchaseState는 어떤 값으로 구성되나요?",
        "Payment Notification Service 설정 방법은?",
        "결제 상태 정보 처리는 어떻게 하나요?"
    ]
    
    # 키워드 추출 테스트
    splitter = KeywordBasedSplitter("../data/dev_center_guide_allmd_touched.md")
    
    for query in test_queries:
        keywords = splitter._extract_keywords_from_content(query)
        print(f"  🔑 '{query[:30]}...' -> {keywords}")
    
    return True


def test_mock_retrieval():
    """모의 검색 테스트 (임베딩 없이)"""
    print("\n🎯 모의 검색 테스트...")
    
    # 샘플 문서 생성
    sample_docs = [
        {
            "content": "PNS(Payment Notification Service)는 결제 알림 서비스입니다. purchaseState 값으로 COMPLETED, CANCELED가 있습니다.",
            "keywords": ["PNS", "purchaseState", "COMPLETED", "CANCELED"]
        },
        {
            "content": "원스토어 인앱결제 API를 사용하여 결제 처리를 할 수 있습니다.",
            "keywords": ["API", "결제"]
        },
        {
            "content": "구매 상태는 purchaseState 필드로 확인할 수 있으며 여러 값이 있습니다.",
            "keywords": ["purchaseState", "구매", "상태"]
        }
    ]
    
    query = "PNS purchaseState 값"
    query_keywords = ["PNS", "purchaseState", "값"]
    
    # 간단한 키워드 매칭 점수 계산
    scores = []
    for doc in sample_docs:
        score = 0
        for keyword in query_keywords:
            if keyword in doc["content"]:
                score += 1
        scores.append((score, doc))
    
    # 점수순 정렬
    scores.sort(reverse=True)
    
    print("  📊 검색 결과 (점수순):")
    for i, (score, doc) in enumerate(scores[:3]):
        print(f"    {i+1}. 점수: {score} | {doc['content'][:50]}...")
    
    return True


def run_basic_tests():
    """기본 테스트 실행"""
    print("🧪 RAG 검색 최적화 기본 테스트 시작")
    print("=" * 60)
    
    tests = [
        ("문서 로딩", test_document_loading),
        ("분할 전략", test_splitting_strategies), 
        ("쿼리 처리", test_query_processing),
        ("모의 검색", test_mock_retrieval)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "✅ 성공" if result else "❌ 실패"
        except Exception as e:
            results[test_name] = f"❌ 오류: {str(e)}"
    
    print("\n" + "=" * 60)
    print("🏆 테스트 결과 요약")
    print("=" * 60)
    
    for test_name, result in results.items():
        print(f"{test_name:15}: {result}")
    
    # 모든 테스트 성공 여부 확인
    all_success = all("✅" in result for result in results.values())
    
    if all_success:
        print("\n🎉 모든 테스트 통과! GPU 머신에서 전체 실험을 실행할 수 있습니다.")
        print("\n💡 다음 단계:")
        print("   1. GPU 머신에 코드 복사")
        print("   2. python simul_11/document_splitting_strategies.py 실행")
        print("   3. 결과 분석 및 최적 전략 선택")
    else:
        print("\n⚠️  일부 테스트 실패. 코드를 확인해주세요.")
    
    return all_success


if __name__ == "__main__":
    run_basic_tests()
