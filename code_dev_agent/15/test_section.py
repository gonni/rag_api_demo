#!/usr/bin/env python3
"""
간단한 HierarchicalSection 테스트 스크립트
"""

from hierarchical_context_rag import HierarchicalSection

def test_hierarchical_section():
    """HierarchicalSection 클래스 테스트"""
    print("🧪 HierarchicalSection 클래스 테스트")
    print("="*40)
    
    try:
        # 기본 생성 테스트
        section = HierarchicalSection(
            id="test-1",
            level=1,
            title="SDK",
            full_path="SDK",
            content="SDK 내용입니다.",
            start_line=1
        )
        print("✅ 기본 생성 성공")
        print(f"  ID: {section.id}")
        print(f"  제목: {section.title}")
        print(f"  레벨: {section.level}")
        print(f"  전체 경로: {section.full_path}")
        print(f"  시작 라인: {section.start_line}")
        
        # 하위 섹션 생성 테스트
        subsection = HierarchicalSection(
            id="test-2",
            level=2,
            title="API Specification",
            full_path="SDK > API Specification",
            content="API 명세 내용입니다.",
            start_line=10,
            parent_id="test-1"
        )
        print("\n✅ 하위 섹션 생성 성공")
        print(f"  제목: {subsection.title}")
        print(f"  전체 경로: {subsection.full_path}")
        print(f"  부모 ID: {subsection.parent_id}")
        
        print("\n🎉 모든 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    test_hierarchical_section()
