#!/usr/bin/env python3
"""
Semantic Chunking 테스트 스크립트
MarkdownHeaderTextSplitter를 사용하여 마크다운 파일을 의미적으로 분할하는 기능을 테스트합니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

def test_semantic_chunking():
    """마크다운 파일의 의미적 분할을 테스트합니다."""
    
    # 마크다운 파일 경로
    markdown_file_path = "../data/dev_center_guide_allmd.md"
    
    # 파일 존재 확인
    if not os.path.exists(markdown_file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {markdown_file_path}")
        return False
    
    try:
        # 파일 내용 읽기
        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        print(f"✅ 마크다운 파일 로드 완료: {len(markdown_content)} 문자")
        
        # MarkdownHeaderTextSplitter 설정
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
        
        # 의미적 분할 수행
        print("🔄 마크다운 문서를 의미적으로 분할 중...")
        semantic_chunks = markdown_splitter.split_text(markdown_content)
        
        print(f"✅ 분할 완료: {len(semantic_chunks)}개의 청크 생성")
        
        # Document 객체들은 이미 생성되어 있음
        documents = semantic_chunks
        
        # 각 Document의 메타데이터에 추가 정보 설정
        for i, doc in enumerate(documents):
            # 기존 메타데이터에 추가 정보 설정
            doc.metadata.update({
                'source': markdown_file_path,
                'chunk_id': i,
                'chunk_type': 'semantic_markdown',
                'content_length': len(doc.page_content),
            })
        
        print(f"✅ Document 객체 생성 완료: {len(documents)}개")
        
        # 통계 정보 출력
        if documents:
            total_chars = sum(len(doc.page_content) for doc in documents)
            avg_chars = total_chars / len(documents)
            
            print(f"\n📊 === 통계 정보 ===")
            print(f"총 문서 수: {len(documents)}개")
            print(f"총 문자 수: {total_chars:,}자")
            print(f"평균 청크 크기: {avg_chars:.1f}자")
            
            # 헤더가 있는 문서 수 (메타데이터에서 헤더 정보 확인)
            docs_with_headers = [doc for doc in documents if any(key.startswith('Header') for key in doc.metadata.keys())]
            print(f"헤더가 포함된 문서: {len(docs_with_headers)}개")
            
            # 첫 번째 문서 샘플 출력
            print(f"\n📄 === 첫 번째 Document 샘플 ===")
            first_doc = documents[0]
            print(f"메타데이터: {first_doc.metadata}")
            print(f"내용 (처음 150자): {first_doc.page_content[:150]}...")
            
            return True
            
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Semantic Chunking 테스트 시작...")
    success = test_semantic_chunking()
    
    if success:
        print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 테스트 중 오류가 발생했습니다.")
        sys.exit(1) 