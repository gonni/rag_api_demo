"""
원스토어 IAP 기술문서 최적화 RAG 시스템 데모
Demo for Optimized RAG System for OneStore IAP Technical Documentation

실행 방법:
python demo.py

또는 주피터 노트북에서:
%run demo.py
"""

import os
import sys
import time
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from optimized_rag_pipeline import OptimizedRAGPipeline, create_pipeline, interactive_mode
    print("✓ 모든 모듈 로드 성공")
except ImportError as e:
    print(f"❌ 모듈 로드 실패: {e}")
    print("필요한 의존성을 설치해주세요: pip install -r requirements.txt")
    sys.exit(1)


def demo_quick_test():
    """빠른 테스트 데모"""
    print("🚀 빠른 테스트 데모 시작")
    print("=" * 60)
    
    # 설정
    data_file = "../../data/dev_center_guide_allmd_touched.md"
    
    # 데이터 파일 존재 확인
    if not os.path.exists(data_file):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_file}")
        print("경로를 수정하거나 파일을 확인해주세요.")
        return
    
    try:
        # 파이프라인 생성
        print("파이프라인 초기화 중...")
        pipeline = OptimizedRAGPipeline(
            data_file=data_file,
            chunk_size=800,
            final_top_k=3
        )
        
        # 단계별 초기화
        pipeline.initialize_models()
        pipeline.load_and_process_documents(force_rebuild=False)
        pipeline.build_retriever()
        
        print("\n✅ 파이프라인 준비 완료!")
        
        # 테스트 질문들
        test_questions = [
            "PurchaseClient를 어떻게 초기화하나요?",
            "purchaseState 값의 종류는 무엇인가요?",
            "PNS 서비스란 무엇인가요?"
        ]
        
        print("\n📝 테스트 질문 실행:")
        print("-" * 40)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[질문 {i}] {question}")
            print("🤔 답변 생성 중...")
            
            start_time = time.time()
            result = pipeline.query(question, stream=False)
            elapsed = time.time() - start_time
            
            print("\n💡 답변:")
            print(result['answer'])
            print(f"\n⏱️  응답 시간: {elapsed:.2f}초")
            print("-" * 40)
        
        # 통계 출력
        stats = pipeline.get_statistics()
        print("\n📊 파이프라인 통계:")
        print(f"  - 총 문서 청크: {stats['total_chunks']}")
        print(f"  - 인덱싱 시간: {stats['index_build_time']:.2f}초")
        print(f"  - 처리된 쿼리: {stats['queries_processed']}")
        print(f"  - 마지막 쿼리 시간: {stats['last_query_time']:.2f}초")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def demo_document_analysis():
    """문서 구조 분석 데모"""
    print("📈 문서 구조 분석 데모")
    print("=" * 60)
    
    data_file = "../../data/dev_center_guide_allmd_touched.md"
    
    if not os.path.exists(data_file):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_file}")
        return
    
    try:
        pipeline = create_pipeline(data_file, force_rebuild=False)
        analysis = pipeline.analyze_document_structure()
        
        print("📋 문서 구조 분석 결과:")
        print("-" * 30)
        
        print(f"총 청크 수: {analysis['total_chunks']}")
        
        print("\n출처별 분포:")
        for source, count in analysis['source_distribution'].items():
            print(f"  - {source}: {count}개")
        
        print("\n콘텐츠 타입별 분포:")
        for content_type, count in analysis['content_type_distribution'].items():
            print(f"  - {content_type}: {count}개")
        
        print("\n주요 기술 용어:")
        for category, terms in analysis['tech_term_distribution'].items():
            if terms:
                print(f"  [{category}]")
                for term, count in list(terms.items())[:5]:
                    print(f"    - {term}: {count}회")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


def demo_search_debugging():
    """검색 디버깅 데모"""
    print("🔍 검색 디버깅 데모")
    print("=" * 60)
    
    data_file = "../../data/dev_center_guide_allmd_touched.md"
    
    if not os.path.exists(data_file):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_file}")
        return
    
    try:
        pipeline = create_pipeline(data_file, force_rebuild=False)
        
        test_queries = [
            "PurchaseClient 초기화",
            "PNS 설정 방법", 
            "purchaseState 값"
        ]
        
        for query in test_queries:
            print(f"\n쿼리: '{query}' 디버깅")
            print("-" * 40)
            
            debug_info = pipeline.debug_search(query)
            
            print("쿼리 분석:")
            analysis = debug_info['query_analysis']
            print(f"  - 추출된 용어: {analysis['query_terms']}")
            print(f"  - 쿼리 타입: {analysis['query_type']}")
            print(f"  - 기술 용어: {analysis['tech_terms']}")
            
            print(f"\n검색 결과: {debug_info['retrieved_count']}개")
            for result in debug_info['results'][:3]:
                print(f"  [{result['rank']}] {result['section_hierarchy']}")
                print(f"      {result['content_preview']}")
                print(f"      타입: {result['content_types']}")
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


def demo_interactive():
    """대화형 데모"""
    print("💬 대화형 모드 데모")
    print("=" * 60)
    
    data_file = "../../data/dev_center_guide_allmd_touched.md"
    
    if not os.path.exists(data_file):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_file}")
        print("경로를 수정하거나 파일을 확인해주세요.")
        return
    
    try:
        pipeline = create_pipeline(data_file, force_rebuild=False)
        interactive_mode(pipeline)
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


def demo_performance_test():
    """성능 테스트 데모"""
    print("⚡ 성능 테스트 데모")
    print("=" * 60)
    
    data_file = "../../data/dev_center_guide_allmd_touched.md"
    
    if not os.path.exists(data_file):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_file}")
        return
    
    try:
        # 파이프라인 생성
        print("파이프라인 초기화 중...")
        start_time = time.time()
        
        pipeline = create_pipeline(data_file, force_rebuild=False)
        init_time = time.time() - start_time
        
        print(f"✓ 초기화 완료: {init_time:.2f}초")
        
        # 배치 테스트
        test_questions = [
            "PurchaseClient 초기화 방법",
            "purchaseState의 값들",
            "PNS 서비스 개요",
            "인앱결제 테스트 방법",
            "구독형 상품 설정"
        ]
        
        print(f"\n배치 테스트 시작: {len(test_questions)}개 질문")
        start_time = time.time()
        
        results = pipeline.batch_query(test_questions)
        batch_time = time.time() - start_time
        
        print("\n📊 성능 결과:")
        print(f"  - 총 소요 시간: {batch_time:.2f}초")
        print(f"  - 질문당 평균 시간: {batch_time/len(test_questions):.2f}초")
        print(f"  - 처리 속도: {len(test_questions)/batch_time:.2f} queries/sec")
        
        # 개별 질문별 성능
        print("\n질문별 응답 시간:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['query_time']:.2f}초 - {result['question']}")
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


def print_menu():
    """메뉴 출력"""
    print("\n" + "="*60)
    print("🤖 원스토어 IAP RAG 시스템 데모 메뉴")
    print("="*60)
    print("1. 빠른 테스트 (Quick Test)")
    print("2. 문서 구조 분석 (Document Analysis)")
    print("3. 검색 디버깅 (Search Debugging)")
    print("4. 대화형 모드 (Interactive Mode)")
    print("5. 성능 테스트 (Performance Test)")
    print("0. 종료 (Exit)")
    print("-"*60)


def main():
    """메인 함수"""
    print("🚀 원스토어 IAP 기술문서 최적화 RAG 시스템")
    print("Optimized RAG System for OneStore IAP Documentation")
    
    # 환경 체크
    print("\n🔧 환경 체크:")
    print(f"Python 버전: {sys.version}")
    print(f"작업 디렉토리: {os.getcwd()}")
    print(f"모듈 경로: {current_dir}")
    
    while True:
        print_menu()
        
        try:
            choice = input("선택하세요 (0-5): ").strip()
            
            if choice == '0':
                print("👋 시스템을 종료합니다.")
                break
            
            elif choice == '1':
                demo_quick_test()
            
            elif choice == '2':
                demo_document_analysis()
            
            elif choice == '3':
                demo_search_debugging()
            
            elif choice == '4':
                demo_interactive()
            
            elif choice == '5':
                demo_performance_test()
            
            else:
                print("❌ 잘못된 선택입니다. 0-5 사이의 숫자를 입력하세요.")
        
        except KeyboardInterrupt:
            print("\n\n👋 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")


# Jupyter Notebook 지원
def notebook_demo():
    """주피터 노트북용 데모"""
    print("📓 Jupyter Notebook 모드")
    
    # 빠른 테스트만 실행
    demo_quick_test()
    
    print("\n💡 추가 기능을 사용하려면:")
    print("  - demo_document_analysis() : 문서 구조 분석")
    print("  - demo_search_debugging() : 검색 디버깅")
    print("  - demo_performance_test() : 성능 테스트")


if __name__ == "__main__":
    # 스크립트 직접 실행
    main()
else:
    # 주피터 노트북에서 import
    print("📓 Jupyter Notebook에서 로드됨")
    print("notebook_demo()를 실행하여 데모를 시작하세요.")
