#!/usr/bin/env python3
"""
한국어 최적화된 RAG 실험
"""

import os
import re
import torch
from typing import List, Dict, Any
from pathlib import Path
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import json
from datetime import datetime

class KoreanOptimizedRAG:
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
    
    def strategy_korean_optimized(self) -> List[Document]:
        """한국어 최적화된 분할 전략"""
        headers = self.extract_headers(self.raw_text)
        docs = []
        
        # 한국어 전문 키워드
        korean_keywords = [
            'PNS', 'purchaseState', 'COMPLETED', 'CANCELED',
            '결제', '취소', '원스토어', '인앱결제', 'Payment Notification Service',
            'API', 'SDK', '개발자', '가이드', '문서'
        ]
        
        for i, header in enumerate(headers):
            content = self.get_content_between_headers(
                self.raw_text,
                header,
                headers[i + 1] if i + 1 < len(headers) else None
            )
            
            # 한국어 키워드 검색
            found_keywords = []
            for keyword in korean_keywords:
                if keyword.lower() in content.lower():
                    found_keywords.append(keyword)
            
            # 제목 계층 구조 생성
            title_hierarchy = self.build_title_hierarchy(headers, i)
            
            # 한국어 최적화된 내용 구성
            enhanced_content = f"""
[한국어_최적화_섹션]
[키워드]: {', '.join(found_keywords) if found_keywords else '없음'}
[전체_제목]: {title_hierarchy}
[짧은_제목]: {header['title']}
[헤더_레벨]: {header['level']}
[내용]:
{content}
[섹션_끝]
"""
            
            doc = Document(
                page_content=enhanced_content,
                metadata={
                    'type': 'korean_optimized',
                    'title': header['title'],
                    'title_hierarchy': title_hierarchy,
                    'keywords': found_keywords,
                    'keyword_count': len(found_keywords),
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
    
    def create_vectorstore_with_korean_models(self, docs: List[Document]) -> FAISS:
        """한국어 최적화된 벡터 스토어를 생성합니다."""
        print(f"\n=== 한국어 최적화 벡터 스토어 생성 중 ===")
        print(f"문서 수: {len(docs)}")
        
        # 한국어 최적화 모델들
        korean_models = [
            "qwen2.5:7b",           # 다국어 지원 우수
            "koalpaca-polyglot-5.8b", # 한국어 특화
            "llama3.2:3b",          # 경량 다국어
            "mistral:7b",           # 기술 문서 특화
            "nomic-embed-text"      # 기본 모델
        ]
        
        for model_name in korean_models:
            try:
                print(f"한국어 모델 시도: {model_name}")
                embeddings = OllamaEmbeddings(model=model_name)
                
                # 한국어 테스트 임베딩 생성
                test_text = "PNS purchaseState COMPLETED CANCELED 결제 취소"
                test_embedding = embeddings.embed_query(test_text)
                print(f"임베딩 차원: {len(test_embedding)}")
                
                vectorstore = FAISS.from_documents(docs, embeddings)
                print(f"✅ {model_name} 모델로 벡터 스토어 생성 성공")
                return vectorstore
                
            except Exception as e:
                print(f"❌ {model_name} 모델 실패: {e}")
                continue
        
        # 모든 모델이 실패하면 기본 모델 사용
        print("기본 모델 사용")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    
    def test_korean_search(self, vectorstore: FAISS, query: str) -> Dict[str, Any] | None:
        """한국어 검색 성능을 테스트합니다."""
        print(f"\n🔍 한국어 검색 테스트: '{query}'")
        
        # 한국어 검색 방법들
        search_methods = [
            {"k": 5, "search_type": "similarity"},
            {"k": 10, "search_type": "similarity"},
            {"k": 5, "search_type": "mmr"},
            {"k": 10, "search_type": "mmr"}
        ]
        
        best_results = None
        best_score = 0
        
        for method in search_methods:
            try:
                retriever = vectorstore.as_retriever(search_kwargs=method)
                docs = retriever.get_relevant_documents(query)
                
                # 한국어 관련성 점수 계산
                relevance_score = 0
                pns_found = 0
                purchase_state_found = 0
                korean_found = 0
                
                for doc in docs:
                    content = doc.page_content.lower()
                    if 'pns' in content:
                        relevance_score += 10
                        pns_found += 1
                    if 'purchasestate' in content or 'purchase_state' in content:
                        relevance_score += 8
                        purchase_state_found += 1
                    if '결제' in content or '취소' in content:
                        relevance_score += 6
                        korean_found += 1
                    if 'completed' in content or 'canceled' in content:
                        relevance_score += 5
                    if '원스토어' in content or '인앱결제' in content:
                        relevance_score += 4
                
                avg_score = relevance_score / len(docs) if docs else 0
                
                print(f"  {method}: 점수 {avg_score:.2f}, PNS {pns_found}개, purchaseState {purchase_state_found}개, 한국어 {korean_found}개")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_results = {
                        'method': method,
                        'docs': docs,
                        'score': avg_score,
                        'pns_found': pns_found,
                        'purchase_state_found': purchase_state_found,
                        'korean_found': korean_found
                    }
                    
            except Exception as e:
                print(f"  {method}: 오류 - {e}")
        
        return best_results
    
    def run_korean_experiment(self):
        """한국어 최적화 실험을 실행합니다."""
        print("🚀 한국어 최적화 RAG 실험 시작")
        
        # 문서 분할
        docs = self.strategy_korean_optimized()
        print(f"생성된 문서 수: {len(docs)}")
        
        # 벡터 스토어 생성
        vectorstore = self.create_vectorstore_with_korean_models(docs)
        
        # 한국어 검색 테스트
        korean_queries = [
            "PNS",
            "purchaseState",
            "결제 상태",
            "원스토어 인앱결제",
            "COMPLETED CANCELED",
            "Payment Notification Service"
        ]
        
        results = {}
        for query in korean_queries:
            result = self.test_korean_search(vectorstore, query)
            if result:
                results[query] = result
                print(f"✅ '{query}' 한국어 검색 완료 - 점수: {result['score']:.2f}")
            else:
                print(f"❌ '{query}' 한국어 검색 실패")
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"korean_experiment_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'korean_search_results': results,
                'timestamp': timestamp
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 결과가 {results_file}에 저장되었습니다.")
        return results

def main():
    """메인 실행 함수"""
    experiment = KoreanOptimizedRAG("data/dev_center_guide_allmd_touched.md")
    results = experiment.run_korean_experiment()
    
    print("\n✅ 한국어 최적화 실험 완료!")

if __name__ == "__main__":
    main() 