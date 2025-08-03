#!/usr/bin/env python3
"""
SemanticChunker 테스트 스크립트
SemanticChunker를 사용하여 마크다운 파일을 의미적으로 분할하는 기능을 테스트합니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

def test_semantic_chunker():
    """SemanticChunker를 사용한 마크다운 파일의 의미적 분할을 테스트합니다."""
    
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
        
        # 임베딩 모델 설정
        print("🔄 임베딩 모델 로딩 중...")
        embeddings = OllamaEmbeddings(model="exaone3.5:latest")
        
        # SemanticChunker 설정
        print("🔄 SemanticChunker 설정 중...")
        semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            min_chunk_size=100,
        )
        
        print("✅ SemanticChunker 설정 완료")
        
        # 의미적 분할 수행
        print("🔄 마크다운 문서를 의미적으로 분할 중...")
        print("⚠️  이 과정은 시간이 걸릴 수 있습니다...")
        semantic_chunks = semantic_chunker.split_text(markdown_content)
        
        print(f"✅ 분할 완료: {len(semantic_chunks)}개의 청크 생성")
        
        # Document 객체 생성
        documents = []
        for i, chunk in enumerate(semantic_chunks):
            # 메타데이터 구성
            metadata = {
                'source': markdown_file_path,
                'chunk_id': i,
                'chunk_type': 'semantic_chunker',
                'content_length': len(chunk),
                'chunker_model': 'SemanticChunker',
                'embedding_model': 'exaone3.5:latest',
            }
            
            # Document 객체 생성
            doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            
            documents.append(doc)
        
        print(f"✅ Document 객체 생성 완료: {len(documents)}개")
        
        # 통계 정보 출력
        if documents:
            total_chars = sum(len(doc.page_content) for doc in documents)
            avg_chars = total_chars / len(documents)
            
            print(f"\n📊 === 통계 정보 ===")
            print(f"총 문서 수: {len(documents)}개")
            print(f"총 문자 수: {total_chars:,}자")
            print(f"평균 청크 크기: {avg_chars:.1f}자")
            
            # 청크 크기 통계
            chunk_sizes = [len(doc.page_content) for doc in documents]
            print(f"최소 청크 크기: {min(chunk_sizes)}자")
            print(f"최대 청크 크기: {max(chunk_sizes)}자")
            
            # 첫 번째 문서 샘플 출력
            print(f"\n📄 === 첫 번째 Document 샘플 ===")
            first_doc = documents[0]
            print(f"메타데이터: {first_doc.metadata}")
            print(f"내용 (처음 150자): {first_doc.page_content[:150]}...")
            
            return True
            
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 SemanticChunker 테스트 시작...")
    success = test_semantic_chunker()
    
    if success:
        print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 테스트 중 오류가 발생했습니다.")
        sys.exit(1) 