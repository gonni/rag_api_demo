#!/usr/bin/env python3
"""
기본 기능 테스트 스크립트

이 스크립트는 의존성 없이도 작동하는 기본 파싱 기능을 테스트합니다.
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hierarchical_context_rag import HierarchicalContextRAG, create_sample_document


def test_basic_parsing():
    """기본 파싱 기능 테스트 (의존성 없이)"""
    print("🧪 기본 파싱 기능 테스트")
    print("="*50)
    
    try:
        # RAG 시스템 초기화
        rag = HierarchicalContextRAG()
        print("✅ RAG 시스템 초기화 완료")
        
        # 샘플 문서 생성
        sample_doc = create_sample_document()
        print("✅ 샘플 문서 생성 완료")
        
        # 마크다운 계층적 파싱
        sections = rag.parse_markdown_hierarchy(sample_doc, "test_doc")
        print(f"✅ 계층적 섹션 파싱 완료: {len(sections)}개 섹션")
        
        # 파싱된 섹션 정보 출력
        print("\n📂 파싱된 섹션 구조:")
        for i, section in enumerate(sections, 1):
            print(f"\n--- 섹션 {i} ---")
            print(f"  제목: {section.title}")
            print(f"  레벨: {section.level}")
            print(f"  전체 경로: {section.full_path}")
            print(f"  내용 길이: {len(section.content)} 문자")
            if section.content.strip():
                print(f"  내용 미리보기: {section.content[:100]}...")
        
        # 맥락 정보가 포함된 문서 생성
        contextual_docs = rag.create_contextual_documents(sections)
        print(f"\n✅ 맥락 문서 생성 완료: {len(contextual_docs)}개 문서")
        
        # 맥락 문서 정보 출력
        print("\n📄 맥락 문서 정보:")
        for i, doc in enumerate(contextual_docs[:3], 1):  # 처음 3개만 출력
            print(f"\n--- 맥락 문서 {i} ---")
            print(f"  섹션 경로: {doc.section_path}")
            print(f"  계층 구조: {doc.section_hierarchy}")
            print(f"  상위 맥락: {doc.parent_context[:100]}...")
        
        # 검색 문서 생성
        search_docs = rag.build_search_documents(contextual_docs)
        print(f"\n✅ 검색 문서 생성 완료: {len(search_docs)}개 문서")
        
        print("\n🎉 기본 파싱 기능 테스트 완료!")
        print("📝 모든 기본 기능이 정상적으로 작동합니다.")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


def test_search_without_dependencies():
    """의존성 없이 검색 기능 테스트"""
    print("\n🔍 의존성 없는 검색 기능 테스트")
    print("="*50)
    
    try:
        rag = HierarchicalContextRAG()
        sample_doc = create_sample_document()
        sections = rag.parse_markdown_hierarchy(sample_doc, "test_doc")
        contextual_docs = rag.create_contextual_documents(sections)
        search_docs = rag.build_search_documents(contextual_docs)
        
        # 검색기 구축 시도 (의존성 없으면 에러 발생)
        try:
            rag.build_retrievers(search_docs)
            print("✅ 검색기 구축 완료 (모든 의존성 설치됨)")
        except ImportError as e:
            print(f"⚠️  검색기 구축 실패 (의존성 부족): {e}")
            print("📝 파싱 및 문서 생성은 정상 작동합니다.")
            print("🔧 의존성 설치 후 검색 기능을 사용할 수 있습니다:")
            print("   pip install -r requirements.txt")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    print("🚀 계층적 맥락 RAG 시스템 - 기본 기능 테스트")
    print("="*60)
    
    # 기본 파싱 테스트
    success1 = test_basic_parsing()
    
    # 의존성 없는 검색 테스트
    success2 = test_search_without_dependencies()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("🎉 모든 테스트 통과!")
        print("✅ 기본 기능이 정상적으로 작동합니다.")
    else:
        print("⚠️  일부 테스트 실패")
        print("🔧 문제를 확인하고 수정해주세요.")
    
    print("\n📋 다음 단계:")
    print("1. 의존성 설치: pip install -r requirements.txt")
    print("2. 전체 데모 실행: python hierarchical_context_rag.py")
    print("3. Jupyter Notebook 실행: jupyter notebook hierarchical_context_rag_demo.ipynb")
