#!/usr/bin/env python3
"""
특정 라인이 어느 청크에 포함되는지 확인하는 디버깅 스크립트
"""

import os
from hierarchical_document_splitter import HierarchicalDocumentSplitter

def main():
    # 문서 로드
    document_path = "../../data/dev_center_guide_allmd_touched.md"
    with open(document_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
    
    lines = document_text.split('\n')
    target_line = "|| purchaseState      | String        | COMPLETED : 결제완료 / CANCELED : 취소                                              |                                       |"
    
    # 타겟 라인 찾기
    target_line_num = None
    for i, line in enumerate(lines):
        if 'purchaseState' in line and 'COMPLETED' in line and 'CANCELED' in line:
            print(f"🎯 찾은 라인 {i+1}: {line}")
            target_line_num = i
            break
    
    if target_line_num is None:
        print("❌ 타겟 라인을 찾을 수 없습니다.")
        return
    
    # 문서 분할
    splitter = HierarchicalDocumentSplitter(include_parent_context=True, max_chunk_size=2000)
    chunks = splitter.split_document(document_text)
    
    print(f"\n총 {len(chunks)}개 청크 생성됨")
    
    # 타겟 라인이 포함된 청크 찾기
    found_chunks = []
    for chunk in chunks:
        if target_line_num >= chunk.start_line and target_line_num <= chunk.end_line:
            found_chunks.append(chunk)
    
    print(f"\n타겟 라인이 포함된 청크: {len(found_chunks)}개")
    
    for i, chunk in enumerate(found_chunks):
        print(f"\n청크 {i+1}:")
        print(f"  제목: {chunk.title}")
        print(f"  경로: {' > '.join(chunk.full_path)}")
        print(f"  레벨: {chunk.level}")
        print(f"  라인 범위: {chunk.start_line}-{chunk.end_line}")
        print(f"  내용 길이: {len(chunk.content)}자")
        
        # 해당 청크에서 purchaseState 라인 확인
        chunk_lines = chunk.content.split('\n')
        for j, line in enumerate(chunk_lines):
            if 'purchaseState' in line and 'COMPLETED' in line:
                print(f"  ★ 라인 {j+1}: {line}")
        
        # 전체 내용의 일부 출력 (디버깅용)
        if len(chunk.content) > 500:
            print(f"  미리보기: {chunk.content[:200]}...")
        else:
            print(f"  전체 내용:\n{chunk.content}")

if __name__ == "__main__":
    main()
