#!/usr/bin/env python3
"""
PNS purchaseState 질의 테스트 스크립트
실제 문서 분할 및 질의 테스트를 수행합니다.
"""

import os
import sys
from pathlib import Path

# 현재 디렉토리에서 모듈 import
from hierarchical_document_splitter import HierarchicalDocumentSplitter, demo_pns_query_test

def main():
    """메인 테스트 함수"""
    print("=== PNS purchaseState 질의 테스트 시작 ===\n")
    
    # 문서 파일 경로
    document_path = "../../data/dev_center_guide_allmd_touched.md"
    
    if not os.path.exists(document_path):
        print(f"❌ 문서 파일을 찾을 수 없습니다: {document_path}")
        return
    
    # 문서 읽기
    print("📖 문서 로드 중...")
    try:
        with open(document_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
        print(f"✅ 문서 로드 완료: {len(document_text):,}자")
    except Exception as e:
        print(f"❌ 문서 로드 실패: {e}")
        return
    
    # 분할기 초기화
    print("\n🔧 문서 분할기 초기화...")
    splitter = HierarchicalDocumentSplitter(
        include_parent_context=True,
        max_chunk_size=2000
    )
    
    # 문서 분할
    print("⚡ 문서 분할 실행...")
    try:
        chunks = splitter.split_document(document_text)
        print(f"✅ 문서 분할 완료: {len(chunks)}개 청크 생성\n")
    except Exception as e:
        print(f"❌ 문서 분할 실패: {e}")
        return
    
    # 청크 요약 출력
    print("📊 청크 요약:")
    splitter.print_chunk_summary(chunks)
    
    # PNS 관련 테스트
    print("\n" + "="*60)
    demo_pns_query_test(chunks)
    
    # 추가 검증: 직접 purchaseState 검색
    print("\n" + "="*60)
    print("=== 직접 purchaseState 검색 테스트 ===\n")
    
    purchase_state_chunks = splitter.find_relevant_chunks(chunks, "purchaseState")
    print(f"purchaseState 관련 청크: {len(purchase_state_chunks)}개")
    
    found_values = False
    for chunk in purchase_state_chunks:
        lines = chunk.content.split('\n')
        for line in lines:
            if 'purchaseState' in line.lower() and ('COMPLETED' in line or 'CANCELED' in line):
                print(f"🎯 발견: {line.strip()}")
                print(f"   청크: {chunk.title}")
                found_values = True
    
    # PNS 관련 청크에서 더 자세히 검색
    if not found_values:
        print("\n🔍 PNS 관련 청크에서 상세 검색...")
        pns_chunks = [chunk for chunk in chunks if 'pns' in chunk.title.lower() or 'payment notification' in chunk.title.lower()]
        
        for chunk in pns_chunks:
            if 'PNS Payment Notification' in chunk.title:
                print(f"\n📋 청크 확인: {chunk.title}")
                print(f"   경로: {' > '.join(chunk.full_path)}")
                print(f"   내용 길이: {len(chunk.content)}자")
                
                lines = chunk.content.split('\n')
                purchase_state_lines = []
                for i, line in enumerate(lines):
                    if 'purchaseState' in line.lower():
                        purchase_state_lines.append((i+1, line.strip()))
                        print(f"   라인 {i+1}: {line.strip()}")
                        # 대소문자 구분 없이 검색하고, 공백과 특수문자 무시
                        line_clean = line.upper().replace(' ', '').replace('|', '')
                        if 'COMPLETED' in line_clean and 'CANCELED' in line_clean:
                            print(f"   🎯 정답 발견!")
                            found_values = True
                
                if not purchase_state_lines:
                    print(f"   ⚠️  이 청크에서 purchaseState 라인을 찾을 수 없습니다.")
                    # 전체 내용에서 purchaseState 검색 (디버깅)
                    if 'purchasestate' in chunk.content.lower():
                        print(f"   💡 내용에는 purchaseState가 포함되어 있습니다.")
                        
                        # 실제로 포함된 라인들 찾기
                        for i, line in enumerate(lines):
                            if 'purchasestate' in line.lower():
                                print(f"      디버그 라인 {i+1}: {line.strip()}")
                                if 'completed' in line.lower() and 'canceled' in line.lower():
                                    print(f"      🎯 디버그에서 정답 발견!")
                                    found_values = True
    
    if found_values:
        print("\n✅ 테스트 성공: purchaseState 값들이 정확히 추출되었습니다!")
        print("   예상 답변: 'COMPLETED : 결제완료 / CANCELED : 취소'")
    else:
        print("\n❌ 테스트 실패: purchaseState 값을 찾을 수 없습니다.")
    
    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    main()
