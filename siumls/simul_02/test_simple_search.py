#!/usr/bin/env python3
"""
간단한 검색 테스트 스크립트
시스템이 제대로 작동하는지 빠르게 확인합니다.
"""

import os
import sys
from pathlib import Path

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings

def test_simple_search():
    """간단한 검색 테스트를 수행합니다."""
    print("간단한 검색 테스트 시작")
    print("=" * 40)
    
    # 문서 경로
    document_path = "../data/dev_center_guide_allmd_touched.md"
    
    if not os.path.exists(document_path):
        print(f"문서 파일을 찾을 수 없습니다: {document_path}")
        return False
    
    try:
        # 1. 문서 로드
        print("1. 문서 로드 중...")
        with open(document_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"   문서 크기: {len(text):,} 문자")
        
        # 2. 문서 분할
        print("2. 문서 분할 중...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk, metadata={"source": document_path}) for chunk in chunks]
        print(f"   분할된 청크 수: {len(documents)}")
        
        # 3. 임베딩 모델 로드 (Ollama 기반)
        print("3. 임베딩 모델 로드 중...")
        embeddings = OllamaEmbeddings(model="codellama")
        print("   임베딩 모델 로드 완료")
        
        # 4. 벡터 스토어 생성
        print("4. 벡터 스토어 생성 중...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("   벡터 스토어 생성 완료")
        
        # 5. 테스트 검색
        print("5. 테스트 검색 수행...")
        test_query = "PNS의 purchaseState의 값은 무엇이 있나요"
        results = vectorstore.similarity_search_with_score(test_query, k=3)
        
        print(f"\n검색 쿼리: {test_query}")
        print("\n검색 결과:")
        print("-" * 40)
        
        target_keywords = ["purchaseState", "COMPLETED", "CANCELED", "결제완료", "취소"]
        found_keywords = []
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n결과 {i} (유사도 점수: {score:.4f}):")
            print(f"내용: {doc.page_content[:200]}...")
            
            # 키워드 확인
            content_lower = doc.page_content.lower()
            for keyword in target_keywords:
                if keyword.lower() in content_lower:
                    found_keywords.append(keyword)
        
        print(f"\n발견된 키워드: {found_keywords}")
        print(f"키워드 매칭률: {len(found_keywords)}/{len(target_keywords)} = {len(found_keywords)/len(target_keywords)*100:.1f}%")
        
        # 6. 성공 여부 판단
        if len(found_keywords) >= 3:  # 최소 3개 키워드가 발견되면 성공
            print("\n✅ 테스트 성공! 시스템이 정상적으로 작동합니다.")
            return True
        else:
            print("\n❌ 테스트 실패. 검색 결과가 부족합니다.")
            return False
            
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {str(e)}")
        return False

def test_ollama_connection():
    """Ollama 연결을 테스트합니다."""
    print("\nOllama 연결 테스트")
    print("=" * 40)
    
    try:
        import ollama
        
        # 사용 가능한 모델 확인
        models = ollama.list()
        print(f"사용 가능한 모델 수: {len(models['models'])}")
        
        if models['models']:
            print("사용 가능한 모델들:")
            for model in models['models']:
                print(f"  - {model['name']}")
            
            # 간단한 테스트
            test_model = models['models'][0]['name']
            print(f"\n모델 '{test_model}'로 간단한 테스트 수행...")
            
            response = ollama.chat(model=test_model, messages=[
                {
                    'role': 'user',
                    'content': '안녕하세요. 간단한 테스트입니다.'
                }
            ])
            
            print(f"응답: {response['message']['content'][:100]}...")
            print("✅ Ollama 연결 성공!")
            return True
            
        else:
            print("❌ 사용 가능한 Ollama 모델이 없습니다.")
            print("다음 명령어로 모델을 다운로드하세요:")
            print("  ollama pull llama2")
            print("  ollama pull nomic-embed-text")
            return False
            
    except ImportError:
        print("❌ ollama 패키지가 설치되지 않았습니다.")
        print("pip install ollama 명령어로 설치하세요.")
        return False
    except Exception as e:
        print(f"❌ Ollama 연결 실패: {str(e)}")
        return False

def main():
    """메인 실행 함수"""
    print("RAG 시스템 간단 테스트")
    print("=" * 50)
    
    # 1. 검색 테스트
    search_success = test_simple_search()
    
    # 2. Ollama 연결 테스트
    ollama_success = test_ollama_connection()
    
    # 3. 최종 결과
    print("\n" + "=" * 50)
    print("테스트 결과 요약")
    print("=" * 50)
    
    if search_success and ollama_success:
        print("✅ 모든 테스트 통과! 시스템이 정상적으로 작동합니다.")
        print("\n이제 다음 명령어로 전체 테스트를 실행할 수 있습니다:")
        print("  python run_complete_test.py")
    elif search_success:
        print("⚠️  검색 시스템은 정상이지만 Ollama 연결에 문제가 있습니다.")
        print("Ollama 설치 및 모델 다운로드를 확인하세요.")
    elif ollama_success:
        print("⚠️  Ollama는 정상이지만 검색 시스템에 문제가 있습니다.")
        print("패키지 설치 및 문서 경로를 확인하세요.")
    else:
        print("❌ 모든 테스트 실패. 시스템 설정을 확인하세요.")
    
    print("\n문제 해결:")
    print("1. pip install -r requirements.txt")
    print("2. ollama 설치 및 모델 다운로드")
    print("3. 문서 파일 경로 확인")

if __name__ == "__main__":
    main() 