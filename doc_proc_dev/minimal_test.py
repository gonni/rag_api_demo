#!/usr/bin/env python3
"""
Minimal test for semantic chunking
"""

import os
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

def minimal_test():
    """Minimal test for semantic chunking"""
    
    # Simple markdown content
    markdown_content = """# Header 1
This is content under header 1.

## Header 2
This is content under header 2.

### Header 3
This is content under header 3.
"""
    
    print("Testing with simple markdown content...")
    print(f"Content length: {len(markdown_content)} characters")
    
    # MarkdownHeaderTextSplitter 설정
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_line=False,
    )
    
    print("MarkdownHeaderTextSplitter configured")
    
    # 의미적 분할 수행
    print("Splitting markdown content...")
    semantic_chunks = markdown_splitter.split_text(markdown_content)
    
    print(f"Split into {len(semantic_chunks)} chunks")
    
    # Document 객체들은 이미 생성되어 있음
    documents = semantic_chunks
    
    print(f"Created {len(documents)} Document objects")
    
    # 각 Document의 메타데이터를 확인하고 추가 정보 설정
    for i, doc in enumerate(documents):
        print(f"Processing document {i}: {len(doc.page_content)} characters")
        
        # 기존 메타데이터에 추가 정보 설정
        doc.metadata.update({
            'chunk_id': i,
            'chunk_type': 'semantic_markdown',
            'content_length': len(doc.page_content),
        })
    
    print(f"Created {len(documents)} Document objects")
    
    # 첫 번째 문서 출력
    if documents:
        first_doc = documents[0]
        print(f"First document metadata: {first_doc.metadata}")
        print(f"First document content: {first_doc.page_content[:100]}...")
    
    return True

if __name__ == "__main__":
    print("🚀 Minimal test 시작...")
    try:
        success = minimal_test()
        if success:
            print("✅ Minimal test 성공!")
        else:
            print("❌ Minimal test 실패!")
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc() 