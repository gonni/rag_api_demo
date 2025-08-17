#!/usr/bin/env python3
"""
RAG 체인 테스트 스크립트
최적의 임베딩 모델과 LLM 체인을 결합하여 실제 RAG 시스템을 테스트합니다.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
import ollama

@dataclass
class RAGTestResult:
    """RAG 테스트 결과를 저장하는 데이터 클래스"""
    query: str
    retrieved_docs: List[Dict[str, Any]]
    generated_answer: str
    search_time: float
    generation_time: float
    total_time: float
    relevance_score: float

class RAGChainTester:
    """RAG 체인을 테스트하는 클래스"""
    
    def __init__(self, document_path: str, embedding_model: str = "codellama"):
        self.document_path = document_path
        self.embedding_model = embedding_model
        
        # 텍스트 분할기 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # LLM 모델 설정 (Ollama)
        self.llm_models = {
            "codellama": {
                "model_name": "codellama",
                "description": "코드 생성에 특화된 Llama 모델"
            },
            "mistral": {
                "model_name": "mistral",
                "description": "Mistral AI의 7B 모델"
            },
            "llama3.2": {
                "model_name": "llama3.2",
                "description": "Meta의 Llama 3.2 모델"
            },
            "mixtral": {
                "model_name": "mixtral",
                "description": "Mixtral AI의 혼합 전문가 모델"
            },
            "eeve-ko": {
                "model_name": "eeve-ko",
                "description": "한국어 특화 EEVE 모델"
            }
        }
        
        # 테스트 쿼리들
        self.test_queries = [
            "PNS의 purchaseState의 값은 무엇이 있나요?",
            "purchaseState 필드에 대해 자세히 설명해주세요",
            "COMPLETED와 CANCELED 상태의 차이점은 무엇인가요?",
            "결제 상태를 확인하는 방법은?",
            "purchaseState가 COMPLETED일 때와 CANCELED일 때의 의미는?"
        ]
        
        # RAG 프롬프트 템플릿
        self.qa_prompt_template = """다음 컨텍스트를 사용하여 질문에 답변하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
        
        self.qa_prompt = PromptTemplate(
            template=self.qa_prompt_template,
            input_variables=["context", "question"]
        )
    
    def load_document(self) -> str:
        """문서를 로드합니다."""
        with open(self.document_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def split_document(self, text: str) -> List[Document]:
        """문서를 청크로 분할합니다."""
        chunks = self.text_splitter.split_text(text)
        return [Document(page_content=chunk, metadata={"source": self.document_path}) for chunk in chunks]
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """벡터 스토어를 생성합니다."""
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        return FAISS.from_documents(documents, embeddings)
    
    def create_rag_chain(self, vectorstore: FAISS, llm_model: str) -> RetrievalQA:
        """RAG 체인을 생성합니다."""
        llm = Ollama(model=llm_model)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True
        )
        
        return qa_chain
    
    def test_rag_chain(self, qa_chain: RetrievalQA, query: str, llm_model: str) -> RAGTestResult:
        """RAG 체인을 테스트합니다."""
        start_time = time.time()
        
        # 검색 및 답변 생성
        result = qa_chain({"query": query})
        
        total_time = time.time() - start_time
        
        # 검색된 문서들 추출
        retrieved_docs = []
        if "source_documents" in result:
            for i, doc in enumerate(result["source_documents"]):
                retrieved_docs.append({
                    "rank": i + 1,
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "full_content": doc.page_content
                })
        
        # 관련성 점수 계산
        target_keywords = ["purchaseState", "COMPLETED", "CANCELED", "결제완료", "취소"]
        relevance_score = self.calculate_relevance_score(
            result.get("result", ""), 
            retrieved_docs, 
            target_keywords
        )
        
        return RAGTestResult(
            query=query,
            retrieved_docs=retrieved_docs,
            generated_answer=result.get("result", ""),
            search_time=total_time * 0.3,  # 추정치
            generation_time=total_time * 0.7,  # 추정치
            total_time=total_time,
            relevance_score=relevance_score
        )
    
    def calculate_relevance_score(self, answer: str, retrieved_docs: List[Dict], target_keywords: List[str]) -> float:
        """답변의 관련성을 계산합니다."""
        if not answer:
            return 0.0
        
        # 답변과 검색된 문서에서 키워드 매칭 확인
        all_text = answer.lower() + " " + " ".join([doc["content"].lower() for doc in retrieved_docs])
        
        keyword_matches = 0
        for keyword in target_keywords:
            if keyword.lower() in all_text:
                keyword_matches += 1
        
        # 답변 품질 점수 (키워드 포함 여부)
        keyword_score = keyword_matches / len(target_keywords)
        
        # 답변 길이 점수 (적절한 길이)
        length_score = min(len(answer) / 100, 1.0)  # 100자 이상이면 만점
        
        # 종합 점수
        return (keyword_score * 0.7) + (length_score * 0.3)
    
    def test_llm_model(self, llm_config: Dict[str, str], vectorstore: FAISS) -> Dict[str, Any]:
        """단일 LLM 모델을 테스트합니다."""
        model_name = llm_config["model_name"]
        print(f"\n테스트 중인 LLM 모델: {model_name}")
        
        try:
            # RAG 체인 생성
            qa_chain = self.create_rag_chain(vectorstore, model_name)
            
            # 테스트 쿼리로 테스트 수행
            all_results = []
            
            for query in self.test_queries:
                print(f"  쿼리 테스트: {query[:50]}...")
                result = self.test_rag_chain(qa_chain, query, model_name)
                all_results.append(result)
            
            # 평균 성능 계산
            avg_total_time = np.mean([r.total_time for r in all_results])
            avg_relevance_score = np.mean([r.relevance_score for r in all_results])
            
            return {
                "model_name": model_name,
                "description": llm_config["description"],
                "avg_total_time": avg_total_time,
                "avg_relevance_score": avg_relevance_score,
                "detailed_results": all_results,
                "status": "success"
            }
            
        except Exception as e:
            print(f"LLM 모델 {model_name} 테스트 실패: {str(e)}")
            return {
                "model_name": model_name,
                "description": llm_config["description"],
                "avg_total_time": float('inf'),
                "avg_relevance_score": 0.0,
                "detailed_results": [],
                "status": "failed",
                "error": str(e)
            }
    
    def run_comparison(self) -> Dict[str, Any]:
        """모든 LLM 모델을 비교 테스트합니다."""
        print("RAG 체인 LLM 모델 비교 테스트 시작")
        print("=" * 50)
        
        # 문서 로드 및 벡터 스토어 생성
        print("문서 로드 중...")
        text = self.load_document()
        documents = self.split_document(text)
        print(f"문서 분할 완료: {len(documents)}개 청크")
        
        print("벡터 스토어 생성 중...")
        vectorstore = self.create_vectorstore(documents)
        print("벡터 스토어 생성 완료")
        
        results = {}
        
        for model_key, model_config in self.llm_models.items():
            result = self.test_llm_model(model_config, vectorstore)
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
            "embedding_model": self.embedding_model,
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
        print("RAG 체인 테스트 결과 요약")
        print("=" * 50)
        
        print(f"테스트 날짜: {results['test_date']}")
        print(f"임베딩 모델: {results['embedding_model']}")
        print(f"테스트된 LLM 모델 수: {results['total_models_tested']}")
        print(f"성공한 모델 수: {results['successful_models']}")
        print(f"최고 성능 모델: {results['best_model']}")
        
        print("\nLLM 모델별 성능 순위:")
        print("-" * 50)
        for i, (model_key, result) in enumerate(results['ranking'], 1):
            if result['status'] == 'success':
                print(f"{i}. {model_key}")
                print(f"   관련성 점수: {result['avg_relevance_score']:.3f}")
                print(f"   평균 응답 시간: {result['avg_total_time']:.3f}초")
                print(f"   설명: {result['description']}")
                print()
    
    def print_detailed_results(self, results: Dict[str, Any]):
        """상세한 결과를 출력합니다."""
        print("\n" + "=" * 50)
        print("상세 테스트 결과")
        print("=" * 50)
        
        for model_key, result in results['detailed_results'].items():
            if result['status'] == 'success':
                print(f"\n모델: {model_key}")
                print(f"설명: {result['description']}")
                print(f"평균 관련성 점수: {result['avg_relevance_score']:.3f}")
                print(f"평균 응답 시간: {result['avg_total_time']:.3f}초")
                
                print("\n상세 결과:")
                for i, detail in enumerate(result['detailed_results']):
                    print(f"  {i+1}. 쿼리: {detail.query}")
                    print(f"     답변: {detail.generated_answer[:100]}...")
                    print(f"     관련성 점수: {detail.relevance_score:.3f}")
                    print(f"     총 시간: {detail.total_time:.3f}초")
                    print()

def main():
    """메인 실행 함수"""
    # 설정
    document_path = "../data/dev_center_guide_allmd_touched.md"
    output_path = "rag_chain_results.json"
    
    # 최적의 임베딩 모델 (이전 테스트 결과에서 선택)
    best_embedding_model = "codellama"
    
    # 테스터 초기화
    tester = RAGChainTester(document_path, best_embedding_model)
    
    # 비교 테스트 실행
    results = tester.run_comparison()
    
    # 결과 저장
    tester.save_results(results, output_path)
    
    # 요약 출력
    tester.print_summary(results)
    
    # 상세 결과 출력 (선택사항)
    # tester.print_detailed_results(results)

if __name__ == "__main__":
    main() 