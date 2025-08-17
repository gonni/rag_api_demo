#!/usr/bin/env python3
"""
검색 문제 진단 및 디버깅 도구
"""

import re
from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

class SearchDebugger:
    def __init__(self, markdown_file_path: str):
        self.markdown_file_path = markdown_file_path
        self.raw_text = self.load_markdown_file(markdown_file_path)
        
    def load_markdown_file(self, file_path: str) -> str:
        """마크다운 파일을 로드합니다."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def find_pns_documents(self) -> List[Dict[str, Any]]:
        """PNS 관련 문서들을 찾습니다."""
        lines = self.raw_text.split('\n')
        pns_docs = []
        
        for i, line in enumerate(lines):
            if 'PNS' in line or 'purchaseState' in line:
                # 주변 컨텍스트 포함
                context_start = max(0, i - 3)
                context_end = min(len(lines), i + 4)
                context = lines[context_start:context_end]
                
                pns_docs.append({
                    'line_number': i,
                    'line_content': line,
                    'context': context,
                    'context_lines': list(range(context_start, context_end))
                })
        
        return pns_docs
    
    def create_simple_documents(self) -> List[Document]:
        """간단한 문서 분할을 생성합니다."""
        lines = self.raw_text.split('\n')
        docs = []
        
        # 100줄씩 청크로 분할
        chunk_size = 100
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            content = '\n'.join(chunk_lines)
            
            # PNS 관련 키워드 검색
            pns_count = content.lower().count('pns')
            purchase_state_count = content.lower().count('purchasestate')
            
            doc = Document(
                page_content=content,
                metadata={
                    'chunk_id': i // chunk_size,
                    'start_line': i,
                    'end_line': min(i + chunk_size, len(lines)),
                    'pns_count': pns_count,
                    'purchase_state_count': purchase_state_count,
                    'source': self.markdown_file_path
                }
            )
            docs.append(doc)
        
        return docs
    
    def test_embedding_similarity(self, query: str, docs: List[Document]) -> Dict[str, Any] | None :
        """임베딩 유사도를 테스트합니다."""
        print(f"\n🔍 임베딩 유사도 테스트: '{query}'")
        
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            
            # 쿼리 임베딩
            query_embedding = embeddings.embed_query(query)
            print(f"쿼리 임베딩 차원: {len(query_embedding)}")
            
            # 문서 임베딩 및 유사도 계산
            similarities = []
            for i, doc in enumerate(docs):
                try:
                    doc_embedding = embeddings.embed_query(doc.page_content[:1000])  # 처음 1000자만
                    
                    # 코사인 유사도 계산
                    import numpy as np
                    similarity = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
                    
                    similarities.append({
                        'doc_id': i,
                        'similarity': similarity,
                        'pns_count': doc.metadata.get('pns_count', 0),
                        'purchase_state_count': doc.metadata.get('purchase_state_count', 0),
                        'content_preview': doc.page_content[:200]
                    })
                    
                except Exception as e:
                    print(f"문서 {i} 임베딩 실패: {e}")
            
            # 유사도 순으로 정렬
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            print(f"\n📊 유사도 결과 (상위 10개):")
            for i, sim in enumerate(similarities[:10]):
                print(f"  {i+1}. 문서 {sim['doc_id']}: 유사도 {sim['similarity']:.4f}, PNS {sim['pns_count']}, purchaseState {sim['purchase_state_count']}")
                print(f"     내용: {sim['content_preview'][:100]}...")
            
            return {
                'query': query,
                'total_docs': len(docs),
                'similarities': similarities[:10]
            }
            
        except Exception as e:
            print(f"임베딩 테스트 실패: {e}")
            return None
    
    def test_vectorstore_search(self, docs: List[Document], query: str) -> Dict[str, Any] | None:
        """벡터 스토어 검색을 테스트합니다."""
        print(f"\n🔍 벡터 스토어 검색 테스트: '{query}'")
        
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = FAISS.from_documents(docs, embeddings)
            
            # 다양한 검색 방법 시도
            search_methods = [
                {"k": 5, "search_type": "similarity"},
                {"k": 10, "search_type": "similarity"},
                {"k": 5, "search_type": "mmr"},
                {"k": 10, "search_type": "mmr"}
            ]
            
            results = {}
            for method in search_methods:
                try:
                    retriever = vectorstore.as_retriever(search_kwargs=method)
                    retrieved_docs = retriever.get_relevant_documents(query)
                    
                    # 관련성 분석
                    pns_found = 0
                    purchase_state_found = 0
                    
                    for doc in retrieved_docs:
                        content = doc.page_content.lower()
                        if 'pns' in content:
                            pns_found += 1
                        if 'purchasestate' in content:
                            purchase_state_found += 1
                    
                    method_name = f"{method['search_type']}_k{method['k']}"
                    results[method_name] = {
                        'docs_retrieved': len(retrieved_docs),
                        'pns_found': pns_found,
                        'purchase_state_found': purchase_state_found,
                        'doc_ids': [doc.metadata.get('chunk_id', 'unknown') for doc in retrieved_docs]
                    }
                    
                    print(f"  {method_name}: 검색된 문서 {len(retrieved_docs)}개, PNS {pns_found}개, purchaseState {purchase_state_found}개")
                    
                except Exception as e:
                    print(f"  {method}: 오류 - {e}")
            
            return results
            
        except Exception as e:
            print(f"벡터 스토어 검색 실패: {e}")
            return None
    
    def run_debug_analysis(self):
        """전체 디버깅 분석을 실행합니다."""
        print("🔍 검색 문제 진단 시작")
        
        # 1. PNS 문서 찾기
        pns_docs = self.find_pns_documents()
        print(f"\n📄 PNS 관련 문서 발견: {len(pns_docs)}개")
        
        for i, doc in enumerate(pns_docs[:5]):
            print(f"  {i+1}. 라인 {doc['line_number']}: {doc['line_content'][:100]}...")
        
        # 2. 간단한 문서 분할
        docs = self.create_simple_documents()
        print(f"\n📚 문서 분할 완료: {len(docs)}개 청크")
        
        # PNS 포함 문서 수 계산
        pns_chunks = sum(1 for doc in docs if doc.metadata['pns_count'] > 0)
        purchase_state_chunks = sum(1 for doc in docs if doc.metadata['purchase_state_count'] > 0)
        
        print(f"  PNS 포함 청크: {pns_chunks}개")
        print(f"  purchaseState 포함 청크: {purchase_state_chunks}개")
        
        # 3. 임베딩 유사도 테스트
        test_queries = ["PNS", "purchaseState", "PNS purchaseState"]
        
        for query in test_queries:
            similarity_result = self.test_embedding_similarity(query, docs)
            if similarity_result:
                print(f"✅ '{query}' 임베딩 테스트 완료")
        
        # 4. 벡터 스토어 검색 테스트
        for query in test_queries:
            search_result = self.test_vectorstore_search(docs, query)
            if search_result:
                print(f"✅ '{query}' 벡터 스토어 검색 완료")
        
        print("\n✅ 디버깅 분석 완료!")

def main():
    """메인 실행 함수"""
    debugger = SearchDebugger("data/dev_center_guide_allmd_touched.md")
    debugger.run_debug_analysis()

if __name__ == "__main__":
    main() 