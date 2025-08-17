#!/usr/bin/env python3
"""
올바른 RAG 구현: 검색용 + 생성용 모델 분리
"""

import os
import re
import torch
from typing import List, Dict, Any
from pathlib import Path
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings, Ollama
from langchain_community.vectorstores import FAISS
from langchain.schema import BaseRetriever
from langchain.chains import RetrievalQA
import json
from datetime import datetime

class ProperRAGImplementation:
    def __init__(self, markdown_file_path: str):
        self.markdown_file_path = markdown_file_path
        self.raw_text = self.load_markdown_file(markdown_file_path)
        
    def load_markdown_file(self, file_path: str) -> str:
        """마크다운 파일을 로드합니다."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def extract_headers(self, text: str) -> List[Dict[str, Any]]:
        """마크다운 텍스트에서 헤더 정보를 추출합니다."""
        headers = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                headers.append({
                    'level': level,
                    'title': title,
                    'line_number': i,
                    'content_start': i + 1
                })
        
        return headers
    
    def get_content_between_headers(self, text: str, start_header: Dict, end_header: Dict[str, Any] | None = None) -> str:
        """두 헤더 사이의 내용을 추출합니다."""
        lines = text.split('\n')
        start_line = start_header['content_start']
        
        if end_header:
            end_line = end_header['line_number']
        else:
            end_line = len(lines)
        
        return '\n'.join(lines[start_line:end_line])
    
    def create_documents(self) -> List[Document]:
        """문서를 생성합니다."""
        headers = self.extract_headers(self.raw_text)
        docs = []
        
        for i, header in enumerate(headers):
            content = self.get_content_between_headers(
                self.raw_text,
                header,
                headers[i + 1] if i + 1 < len(headers) else None
            )
            
            # 제목 계층 구조 생성
            title_hierarchy = self.build_title_hierarchy(headers, i)
            
            doc = Document(
                page_content=f"제목: {title_hierarchy}\n\n내용:\n{content}",
                metadata={
                    'title': header['title'],
                    'title_hierarchy': title_hierarchy,
                    'header_level': header['level'],
                    'source': self.markdown_file_path,
                    'line_number': header['line_number']
                }
            )
            docs.append(doc)
        
        return docs
    
    def build_title_hierarchy(self, headers: List[Dict], current_index: int) -> str:
        """현재 헤더까지의 제목 계층 구조를 생성합니다."""
        current_header = headers[current_index]
        hierarchy = [current_header['title']]
        
        for i in range(current_index - 1, -1, -1):
            if headers[i]['level'] < current_header['level']:
                hierarchy.insert(0, headers[i]['title'])
                current_header = headers[i]
        
        return ' / '.join(hierarchy)
    
    def setup_embedding_model(self) -> OllamaEmbeddings:
        """검색용 임베딩 모델을 설정합니다."""
        print("🔍 검색용 임베딩 모델 설정")
        
        # 검색에 최적화된 모델들 (우선순위 순)
        embedding_models = [
            "nomic-embed-text",    # 다국어 검색 특화
            "all-minilm",          # 경량 검색 모델
            "bge-small-en"         # 영어 검색 특화
        ]
        
        for model_name in embedding_models:
            try:
                print(f"임베딩 모델 시도: {model_name}")
                embeddings = OllamaEmbeddings(model=model_name)
                
                # 테스트 임베딩 생성
                test_text = "PNS purchaseState COMPLETED CANCELED"
                test_embedding = embeddings.embed_query(test_text)
                print(f"✅ {model_name} 모델 성공 - 임베딩 차원: {len(test_embedding)}")
                return embeddings
                
            except Exception as e:
                print(f"❌ {model_name} 모델 실패: {e}")
                continue
        
        # 모든 모델이 실패하면 기본 모델 사용
        print("기본 모델 사용: nomic-embed-text")
        return OllamaEmbeddings(model="nomic-embed-text")
    
    def setup_generation_model(self) -> Ollama:
        """생성용 모델을 설정합니다."""
        print("🤖 생성용 모델 설정")
        
        # 생성에 최적화된 모델들 (우선순위 순)
        generation_models = [
            "qwen2.5:7b",           # 다국어 생성 우수
            "llama3.2:3b",          # 경량 생성
            "mistral:7b",           # 기술 문서 생성
            "exaone3.5:latest"      # 일반 생성
        ]
        
        for model_name in generation_models:
            try:
                print(f"생성 모델 시도: {model_name}")
                llm = Ollama(model=model_name)
                
                # 테스트 생성
                test_prompt = "안녕하세요. 간단히 답변해주세요."
                response = llm.invoke(test_prompt)
                print(f"✅ {model_name} 모델 성공")
                return llm
                
            except Exception as e:
                print(f"❌ {model_name} 모델 실패: {e}")
                continue
        
        # 모든 모델이 실패하면 기본 모델 사용
        print("기본 모델 사용: exaone3.5:latest")
        return Ollama(model="exaone3.5:latest")
    
    def create_vectorstore(self, docs: List[Document], embeddings: OllamaEmbeddings) -> FAISS:
        """벡터 스토어를 생성합니다."""
        print(f"\n📚 벡터 스토어 생성 중")
        print(f"문서 수: {len(docs)}")
        
        vectorstore = FAISS.from_documents(docs, embeddings)
        print("✅ 벡터 스토어 생성 완료")
        return vectorstore
    
    def test_retrieval(self, vectorstore: FAISS, query: str) -> List[Document]:
        """검색 성능을 테스트합니다."""
        print(f"\n🔍 검색 테스트: '{query}'")
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(query)
        
        print(f"검색된 문서 수: {len(docs)}")
        for i, doc in enumerate(docs):
            title = doc.metadata.get('title', 'Unknown')
            content_preview = doc.page_content[:200] + "..."
            print(f"  {i+1}. {title}")
            print(f"     {content_preview}")
        
        return docs
    
    def test_generation(self, llm: Ollama, query: str, context_docs: List[Document]) -> str:
        """생성 성능을 테스트합니다."""
        print(f"\n🤖 생성 테스트: '{query}'")
        
        # 컨텍스트 구성
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # 프롬프트 구성
        prompt = f"""
다음 문서를 참고하여 질문에 답변해주세요.

문서:
{context}

질문: {query}

답변:"""
        
        try:
            response = llm.invoke(prompt)
            print(f"생성된 답변: {response}")
            return response
        except Exception as e:
            print(f"생성 실패: {e}")
            return "답변 생성에 실패했습니다."
    
    def run_proper_rag_experiment(self):
        """올바른 RAG 실험을 실행합니다."""
        print("🚀 올바른 RAG 실험 시작")
        
        # 1. 문서 생성
        docs = self.create_documents()
        print(f"생성된 문서 수: {len(docs)}")
        
        # 2. 검색용 모델 설정
        embeddings = self.setup_embedding_model()
        
        # 3. 생성용 모델 설정
        llm = self.setup_generation_model()
        
        # 4. 벡터 스토어 생성
        vectorstore = self.create_vectorstore(docs, embeddings)
        
        # 5. 테스트 쿼리들
        test_queries = [
            "PNS의 purchaseState에는 어떤 값들이 있나요?",
            "원스토어 인앱결제의 결제 상태는 무엇인가요?",
            "COMPLETED와 CANCELED의 차이점은 무엇인가요?"
        ]
        
        results = {}
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"테스트 쿼리: {query}")
            print(f"{'='*60}")
            
            # 검색 단계
            retrieved_docs = self.test_retrieval(vectorstore, query)
            
            # 생성 단계
            generated_answer = self.test_generation(llm, query, retrieved_docs)
            
            results[query] = {
                'retrieved_docs': len(retrieved_docs),
                'generated_answer': generated_answer,
                'doc_titles': [doc.metadata.get('title', 'Unknown') for doc in retrieved_docs]
            }
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"proper_rag_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 결과가 {results_file}에 저장되었습니다.")
        return results

def main():
    """메인 실행 함수"""
    experiment = ProperRAGImplementation("data/dev_center_guide_allmd_touched.md")
    results = experiment.run_proper_rag_experiment()
    
    print("\n✅ 올바른 RAG 실험 완료!")

if __name__ == "__main__":
    main() 