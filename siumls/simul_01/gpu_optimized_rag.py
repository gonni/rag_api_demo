#!/usr/bin/env python3
"""
GPU 최적화된 RAG 실험
"""

import os
import re
import torch
from typing import List, Dict, Any
from pathlib import Path
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter
import json
from datetime import datetime

# GPU 메모리 최적화 설정
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # GPU 메모리 할당 최적화
    torch.backends.cudnn.benchmark = True
    print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

class GPUOptimizedDocumentSplitterExperiment:
    def __init__(self, markdown_file_path: str):
        self.markdown_file_path = markdown_file_path
        self.raw_text = self.load_markdown_file(markdown_file_path)
        self.results = {}
        
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
    
    def strategy_1_hierarchical_with_context(self) -> List[Document]:
        """전략 1: 계층적 분할 + 전체 맥락 포함"""
        headers = self.extract_headers(self.raw_text)
        docs = []
        
        # 대제목(##) 기준으로 메인 섹션 생성
        main_sections = []
        for i, header in enumerate(headers):
            if header['level'] == 2:
                section_content = self.get_content_between_headers(
                    self.raw_text, 
                    header, 
                    headers[i + 1] if i + 1 < len(headers) else None
                )
                main_sections.append({
                    'title': header['title'],
                    'content': section_content,
                    'start_line': header['line_number']
                })
        
        # 각 메인 섹션에 대해 세부 분할
        for section in main_sections:
            # 메인 섹션 전체를 하나의 문서로 생성
            main_doc = Document(
                page_content=f"[MAIN_SECTION]: {section['title']}\n\n{section['content']}",
                metadata={
                    'type': 'main_section',
                    'title': section['title'],
                    'source': self.markdown_file_path,
                    'section_level': 2
                }
            )
            docs.append(main_doc)
            
            # 해당 섹션 내의 소제목들로 세부 분할
            section_headers = self.extract_headers(section['content'])
            for sub_header in section_headers:
                if sub_header['level'] >= 3:
                    sub_content = self.get_content_between_headers(
                        section['content'],
                        sub_header,
                        section_headers[section_headers.index(sub_header) + 1] if section_headers.index(sub_header) + 1 < len(section_headers) else None
                    )
                    
                    title_hierarchy = f"{section['title']} / {sub_header['title']}"
                    
                    sub_doc = Document(
                        page_content=f"[SUBSECTION]: {title_hierarchy}\n\n{sub_content}",
                        metadata={
                            'type': 'subsection',
                            'title': sub_header['title'],
                            'parent_title': section['title'],
                            'title_hierarchy': title_hierarchy,
                            'source': self.markdown_file_path,
                            'section_level': sub_header['level']
                        }
                    )
                    docs.append(sub_doc)
        
        return docs
    
    def create_vectorstore(self, docs: List[Document], strategy_name: str) -> FAISS:
        """GPU 최적화된 벡터 스토어를 생성합니다."""
        print(f"\n=== {strategy_name} 벡터 스토어 생성 중 ===")
        print(f"문서 수: {len(docs)}")
        
        # GPU 메모리 상태 확인
        if torch.cuda.is_available():
            print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            print(f"GPU 메모리 캐시: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
        
        # 배치 크기 조정 (GPU 메모리에 따라)
        batch_size = 32 if torch.cuda.is_available() else 16
        
        embeddings = OllamaEmbeddings(model="exaone3.5:latest")
        
        # 배치 처리로 벡터 스토어 생성
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU 메모리 정리 후: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        
        return vectorstore
    
    def test_query(self, vectorstore: FAISS, query: str, strategy_name: str) -> Dict[str, Any]:
        """쿼리를 테스트하고 결과를 반환합니다."""
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(query)
        
        # 결과 분석
        relevant_docs = []
        for i, doc in enumerate(docs):
            relevance_score = 0
            if 'purchaseState' in doc.page_content:
                relevance_score += 10
            if 'COMPLETED' in doc.page_content or 'CANCELED' in doc.page_content:
                relevance_score += 5
            if 'PNS' in doc.page_content:
                relevance_score += 3
            
            relevant_docs.append({
                'rank': i + 1,
                'title': doc.metadata.get('title', 'Unknown'),
                'content_preview': doc.page_content[:200] + '...',
                'relevance_score': relevance_score,
                'metadata': doc.metadata
            })
        
        return {
            'strategy': strategy_name,
            'query': query,
            'total_docs': len(docs),
            'relevant_docs': relevant_docs,
            'avg_relevance_score': sum(d['relevance_score'] for d in relevant_docs) / len(relevant_docs) if relevant_docs else 0
        }
    
    def run_experiment(self, test_queries: List[str] | None = None):
        """GPU 최적화된 실험을 실행합니다."""
        if test_queries is None:
            test_queries = [
                "PNS의 purchaseState에는 어떤 값들이 있나요?",
                "purchaseState COMPLETED CANCELED",
                "원스토어 결제 상태 값",
                "PNS payment notification service"
            ]
        
        strategies = {
            'strategy_1': self.strategy_1_hierarchical_with_context,
        }
        
        results = {}
        
        for strategy_name, strategy_func in strategies.items():
            print(f"\n=== 실행 중: {strategy_name} ===")
            
            # 문서 분할
            docs = strategy_func()
            print(f"생성된 문서 수: {len(docs)}")
            
            # GPU 최적화된 벡터 스토어 생성
            vectorstore = self.create_vectorstore(docs, strategy_name)
            
            # 각 쿼리 테스트
            strategy_results = []
            for query in test_queries:
                result = self.test_query(vectorstore, query, strategy_name)
                strategy_results.append(result)
                print(f"쿼리: {query}")
                print(f"평균 관련성 점수: {result['avg_relevance_score']:.2f}")
            
            results[strategy_name] = {
                'doc_count': len(docs),
                'query_results': strategy_results
            }
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"gpu_experiment_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n실험 결과가 {results_file}에 저장되었습니다.")
        
        return results

def main():
    """메인 실행 함수"""
    print("🚀 GPU 최적화된 RAG 실험 시작")
    
    # GPU 상태 확인
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("CPU 모드로 실행됩니다.")
    
    # 실험 실행
    experiment = GPUOptimizedDocumentSplitterExperiment("data/dev_center_guide_allmd_touched.md")
    
    # 테스트 쿼리 정의
    test_queries = [
        "PNS의 purchaseState에는 어떤 값들이 있나요?",
        "purchaseState COMPLETED CANCELED 값",
        "원스토어 결제 상태 값들",
        "PNS payment notification service purchaseState",
        "COMPLETED CANCELED 결제 상태"
    ]
    
    # 실험 실행
    results = experiment.run_experiment(test_queries)
    
    print("\n✅ GPU 최적화 실험 완료!")

if __name__ == "__main__":
    main() 