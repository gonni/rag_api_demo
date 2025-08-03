#!/usr/bin/env python3
"""
노트북 함수들을 테스트하는 스크립트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

def test_notebook_functions():
    """노트북의 주요 함수들을 테스트합니다."""
    
    print("🚀 노트북 함수 테스트 시작...")
    
    # 1. 마크다운 파일 로드 테스트
    print("\n1. 마크다운 파일 로드 테스트")
    markdown_file_path = "../data/dev_center_guide_allmd.md"
    
    if not os.path.exists(markdown_file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {markdown_file_path}")
        return False
    
    with open(markdown_file_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    print(f"✅ 마크다운 파일 로드 완료: {len(markdown_content)} 문자")
    
    # 2. MarkdownHeaderTextSplitter 설정 테스트
    print("\n2. MarkdownHeaderTextSplitter 설정 테스트")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_line=False,
    )
    
    print("✅ MarkdownHeaderTextSplitter 설정 완료")
    
    # 3. 의미적 분할 테스트
    print("\n3. 의미적 분할 테스트")
    semantic_chunks = markdown_splitter.split_text(markdown_content)
    print(f"✅ 분할 완료: {len(semantic_chunks)}개의 청크 생성")
    
    # 4. Document 객체 처리 테스트
    print("\n4. Document 객체 처리 테스트")
    documents = semantic_chunks
    
    # 각 Document의 메타데이터에 추가 정보 설정
    for i, doc in enumerate(documents):
        doc.metadata.update({
            'source': markdown_file_path,
            'chunk_id': i,
            'chunk_type': 'semantic_markdown',
            'content_length': len(doc.page_content),
        })
    
    print(f"✅ Document 객체 처리 완료: {len(documents)}개")
    
    # 5. 통계 정보 테스트
    print("\n5. 통계 정보 테스트")
    if documents:
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chars = total_chars / len(documents)
        
        print(f"총 문서 수: {len(documents)}개")
        print(f"총 문자 수: {total_chars:,}자")
        print(f"평균 청크 크기: {avg_chars:.1f}자")
        
        # 헤더가 있는 문서 수
        docs_with_headers = [doc for doc in documents if any(key.startswith('Header') for key in doc.metadata.keys())]
        print(f"헤더가 포함된 문서: {len(docs_with_headers)}개")
        
        # 첫 번째 문서 샘플 출력
        first_doc = documents[0]
        print(f"첫 번째 문서 메타데이터: {first_doc.metadata}")
        print(f"첫 번째 문서 내용 (처음 100자): {first_doc.page_content[:100]}...")
    
    # 6. 헤더 정보 분석 테스트
    print("\n6. 헤더 정보 분석 테스트")
    header_types = {}
    for doc in documents:
        for key in doc.metadata.keys():
            if key.startswith('Header'):
                header_types[key] = header_types.get(key, 0) + 1
    
    for header_type, count in sorted(header_types.items()):
        print(f"  {header_type}: {count}개 문서")
    
    print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
    return True

if __name__ == "__main__":
    try:
        success = test_notebook_functions()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 