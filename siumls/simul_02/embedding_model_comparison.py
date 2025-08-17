#!/usr/bin/env python3
"""
RAG 임베딩 모델 비교 시뮬레이션
다양한 임베딩 모델의 검색 성능을 비교하여 최적의 모델을 찾습니다.
"""

import os
import json
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
import ollama

@dataclass
class SearchResult:
    """검색 결과를 저장하는 데이터 클래스"""
    model_name: str
    query: str
    top_results: List[Dict[str, Any]]
    search_time: float
    relevance_score: float

class EmbeddingModelTester:
    """다양한 임베딩 모델의 검색 성능을 테스트하는 클래스"""
    
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # 테스트할 임베딩 모델들 (Ollama 기반)
        self.embedding_models = {
            "codellama": {
                "model_name": "codellama",
                "description": "Code Llama 기반 임베딩"
            },
            "mistral": {
                "model_name": "mistral",
                "description": "Mistral 기반 임베딩"
            },
            "llama3.2": {
                "model_name": "llama3.2",
                "description": "Llama 3.2 기반 임베딩"
            },
            "mixtral": {
                "model_name": "mixtral",
                "description": "Mixtral 기반 임베딩"
            },
            "eeve-ko": {
                "model_name": "eeve-ko",
                "description": "한국어 특화 EEVE 모델 기반 임베딩"
            }
        }
        
        self.test_queries = [
            "PNS의 purchaseState의 값은 무엇이 있나요",
            "purchaseState COMPLETED CANCELED",
            "결제 상태 값",
            "purchaseState 필드 설명",
            "COMPLETED CANCELED 결제"
        ]
        
    def load_document(self) -> str:
        """문서를 로드합니다."""
        with open(self.document_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def split_document(self, text: str) -> List[Document]:
        """문서를 청크로 분할합니다."""
        chunks = self.text_splitter.split_text(text)
        return [Document(page_content=chunk, metadata={"source": self.document_path}) for chunk in chunks]
    
    def create_vectorstore(self, embeddings, documents: List[Document]) -> FAISS:
        """벡터 스토어를 생성합니다."""
        return FAISS.from_documents(documents, embeddings)
    
    def search_with_model(self, model_name: str, query: str, vectorstore: FAISS) -> Tuple[List[Dict], float]:
        """특정 모델로 검색을 수행합니다."""
        start_time = time.time()
        results = vectorstore.similarity_search_with_score(query, k=5)
        search_time = time.time() - start_time
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "score": float(score),
                "full_content": doc.page_content
            })
        
        return formatted_results, search_time
    
    def calculate_relevance_score(self, results: List[Dict], target_keywords: List[str]) -> float:
        """검색 결과의 관련성을 계산합니다."""
        if not results:
            return 0.0
        
        # 타겟 키워드들이 검색 결과에 포함되어 있는지 확인
        content_text = " ".join([result["content"].lower() for result in results])
        
        keyword_matches = 0
        for keyword in target_keywords:
            if keyword.lower() in content_text:
                keyword_matches += 1
        
        # 스코어 계산 (키워드 매칭 + 유사도 스코어)
        keyword_score = keyword_matches / len(target_keywords)
        similarity_score = 1.0 - float(np.mean([result["score"] for result in results]))
        
        return float((keyword_score * 0.7) + (similarity_score * 0.3))
    
    def test_model(self, model_config: Dict[str, str]) -> Dict[str, Any]:
        """단일 모델을 테스트합니다."""
        model_name = model_config["model_name"]
        print(f"\n테스트 중인 모델: {model_name}")
        
        try:
            # Ollama 임베딩 모델 로드
            embeddings = OllamaEmbeddings(model=model_name)
            
            # 문서 로드 및 분할
            text = self.load_document()
            documents = self.split_document(text)
            print(f"문서 분할 완료: {len(documents)}개 청크")
            
            # 벡터 스토어 생성
            vectorstore = self.create_vectorstore(embeddings, documents)
            
            # 테스트 쿼리로 검색 수행
            all_results = []
            target_keywords = ["purchaseState", "COMPLETED", "CANCELED", "결제완료", "취소"]
            
            for query in self.test_queries:
                results, search_time = self.search_with_model(model_name, query, vectorstore)
                relevance_score = self.calculate_relevance_score(results, target_keywords)
                
                all_results.append({
                    "query": query,
                    "results": results,
                    "search_time": search_time,
                    "relevance_score": relevance_score
                })
            
            # 평균 성능 계산
            avg_search_time = np.mean([r["search_time"] for r in all_results])
            avg_relevance_score = np.mean([r["relevance_score"] for r in all_results])
            
            return {
                "model_name": model_name,
                "description": model_config["description"],
                "avg_search_time": avg_search_time,
                "avg_relevance_score": avg_relevance_score,
                "detailed_results": all_results,
                "status": "success"
            }
            
        except Exception as e:
            print(f"모델 {model_name} 테스트 실패: {str(e)}")
            return {
                "model_name": model_name,
                "description": model_config["description"],
                "avg_search_time": float('inf'),
                "avg_relevance_score": 0.0,
                "detailed_results": [],
                "status": "failed",
                "error": str(e)
            }
    
    def run_comparison(self) -> Dict[str, Any]:
        """모든 모델을 비교 테스트합니다."""
        print("RAG 임베딩 모델 비교 테스트 시작")
        print("=" * 50)
        
        results = {}
        
        for model_key, model_config in self.embedding_models.items():
            result = self.test_model(model_config)
            results[model_key] = result
        
        # 결과 정렬 (관련성 점수 기준)
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]["avg_relevance_score"],
            reverse=True
        )
        
        # 요약 리포트 생성
        summary = {
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "document_path": self.document_path,
            "total_models_tested": len(results),
            "successful_models": len([r for r in results.values() if r["status"] == "success"]),
            "best_model": sorted_results[0][0] if sorted_results else None,
            "detailed_results": results,
            "ranking": sorted_results
        }
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """결과를 파일로 저장합니다."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"결과가 {output_path}에 저장되었습니다.")
    
    def print_summary(self, results: Dict[str, Any]):
        """결과 요약을 출력합니다."""
        print("\n" + "=" * 50)
        print("테스트 결과 요약")
        print("=" * 50)
        
        print(f"테스트 날짜: {results['test_date']}")
        print(f"테스트된 모델 수: {results['total_models_tested']}")
        print(f"성공한 모델 수: {results['successful_models']}")
        print(f"최고 성능 모델: {results['best_model']}")
        
        print("\n모델별 성능 순위:")
        print("-" * 50)
        for i, (model_key, result) in enumerate(results['ranking'], 1):
            if result['status'] == 'success':
                print(f"{i}. {model_key}")
                print(f"   관련성 점수: {result['avg_relevance_score']:.3f}")
                print(f"   평균 검색 시간: {result['avg_search_time']:.3f}초")
                print(f"   설명: {result['description']}")
                print()

def main():
    """메인 실행 함수"""
    # 설정
    document_path = "../data/dev_center_guide_allmd_touched.md"
    output_path = "embedding_comparison_results.json"
    
    # 테스터 초기화
    tester = EmbeddingModelTester(document_path)
    
    # 비교 테스트 실행
    results = tester.run_comparison()
    
    # 결과 저장
    tester.save_results(results, output_path)
    
    # 요약 출력
    tester.print_summary(results)

if __name__ == "__main__":
    main() 