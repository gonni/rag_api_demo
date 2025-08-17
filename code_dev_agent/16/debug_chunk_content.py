#!/usr/bin/env python3
"""
PNS Payment Notification 청크의 실제 내용을 확인하는 스크립트
"""

import os
from hierarchical_document_splitter import HierarchicalDocumentSplitter

def main():
    # 문서 로드
    document_path = "../../data/dev_center_guide_allmd_touched.md"
    with open(document_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
    
    # 문서 분할
    splitter = HierarchicalDocumentSplitter(include_parent_context=True, max_chunk_size=2000)
    chunks = splitter.split_document(document_text)
    
    # PNS Payment Notification 청크 찾기
    target_chunks = [chunk for chunk in chunks if 'PNS Payment Notification' in chunk.title]
    
    print(f"PNS Payment Notification 관련 청크: {len(target_chunks)}개\n")
    
    for i, chunk in enumerate(target_chunks):
        print(f"=== 청크 {i+1}: {chunk.title} ===")
        print(f"경로: {' > '.join(chunk.full_path)}")
        print(f"레벨: {chunk.level}")
        print(f"라인 범위: {chunk.start_line}-{chunk.end_line}")
        print(f"내용 길이: {len(chunk.content)}자")
        print("\n내용:")
        print("-" * 80)
        
        # 전체 내용 출력
        lines = chunk.content.split('\n')
        for j, line in enumerate(lines):
            if 'purchaseState' in line.lower():
                print(f">>> 라인 {j+1}: {line}")
            elif j < 20 or 'COMPLETED' in line or 'CANCELED' in line:  # 처음 20줄이나 키워드 포함 라인
                print(f"{j+1:3}: {line}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
