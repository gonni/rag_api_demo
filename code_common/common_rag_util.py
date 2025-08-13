import os
import re
import pickle
from typing import List
from pathlib import Path
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from typing import List, Dict, Any, Tuple, Optional

class CommonRAGUtil:
    def __init__(self):
        pass

    def save_documents(self, docs: List[Document], output_path: str):
        """
        List[Document]를 pickle 파일로 저장합니다.
        
        Args:
            docs (List[Document]): 저장할 문서 리스트
            output_path (str): 저장할 디렉토리 경로
        """
        os.makedirs(output_path, exist_ok=True)
        docs_file_path = os.path.join(output_path, "documents.pkl")
        
        with open(docs_file_path, "wb") as f:
            pickle.dump(docs, f)
        
        print(f"✅ 문서 저장 완료: {docs_file_path}")
        print(f"📄 저장된 문서 수: {len(docs)}")


    def load_documents(self, input_path: str) -> List[Document]:
        """
        저장된 pickle 파일에서 List[Document]를 로드합니다.
        
        Args:
            input_path (str): 문서가 저장된 디렉토리 경로 또는 파일 경로
            
        Returns:
            List[Document]: 로드된 문서 리스트
        """
        # 디렉토리 경로인 경우 documents.pkl 파일을 찾음
        if os.path.isdir(input_path):
            docs_file_path = os.path.join(input_path, "documents.pkl")
        else:
            docs_file_path = input_path
        
        if not os.path.exists(docs_file_path):
            raise FileNotFoundError(f"문서 파일을 찾을 수 없습니다: {docs_file_path}")
        
        with open(docs_file_path, "rb") as f:
            docs = pickle.load(f)
        
        print(f"✅ 문서 로드 완료: {docs_file_path}")
        print(f"📄 로드된 문서 수: {len(docs)}")
        
        return docs


    def embed_and_save_with_docs(self, docs: List[Document], output_path: str, model_name: str = "bge-m3:latest"):
        """
        문서를 임베딩하고 FAISS 데이터베이스로 저장하며, 동시에 원본 문서도 저장합니다.
        
        Args:
            docs (List[Document]): 처리할 문서 리스트
            output_path (str): 저장할 디렉토리 경로
            model_name (str): 임베딩 모델명
        """
        # 임베딩 모델 초기화
        embedding_model = OllamaEmbeddings(model=model_name)
        
        # FAISS 데이터베이스 생성 및 저장
        db = FAISS.from_documents(docs, embedding_model)
        db.save_local(output_path)
        print(f"✅ 임베딩 저장 완료: {output_path}")
        
        # 원본 문서도 함께 저장
        self.save_documents(docs, output_path)
        

    def load_both_faiss_and_docs(self, folder_path: str, model_name: str = "bge-m3:latest") -> tuple[FAISS, List[Document]]:
        """
        FAISS 벡터 데이터베이스와 원본 문서를 모두 로드합니다.
        
        Args:
            folder_path (str): 데이터가 저장된 디렉토리 경로
            model_name (str): 임베딩 모델명
            
        Returns:
            tuple: (FAISS 데이터베이스, List[Document])
        """
        # 임베딩 모델 초기화
        embedding_model = OllamaEmbeddings(model=model_name)
        
        # FAISS 데이터베이스 로드
        loaded_db = FAISS.load_local(
            folder_path=folder_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )
        print(f"✅ FAISS 데이터베이스 로드 완료: {folder_path}")
        
        # 원본 문서 로드
        docs = self.load_documents(folder_path)
        
        return loaded_db, docs
    
class SmartRetriever:
    """스마트 검색기 - 키워드 우선순위 기반"""
    
    def __init__(self, documents: List[Document], embedding_model_name: str = "bge-m3:latest"):
        self.documents = documents
        self.embedding_model_name = embedding_model_name
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
    def build_retrievers(self):
        """검색기 구축"""
        print(f"🔧 검색기 구축 중... (문서 수: {len(self.documents)})")
        
        # Vector store 구축
        embeddings = OllamaEmbeddings(model=self.embedding_model_name)
        self.vector_store = FAISS.from_documents(self.documents, embeddings)
        
        # BM25 검색기 구축
        self.bm25_retriever = BM25Retriever.from_documents(
            self.documents,
            bm25_params={"k1": 1.5, "b": 0.75}
        )
        self.bm25_retriever.k = 20
        
        # Vector 검색기
        vector_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "fetch_k": 50, "lambda_mult": 0.7}
        )
        
        # 앙상블 검색기
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[0.6, 0.4]  # BM25에 더 높은 가중치
        )
        
        print("✅ 검색기 구축 완료")
    
    def get_retriever(self):
        return self.ensemble_retriever
    
    def smart_search(self, query: str, max_results: int = 10) -> List[Document]:
        """스마트 검색 - 키워드 우선순위 적용"""
        if not self.ensemble_retriever:
            raise ValueError("검색기가 구축되지 않았습니다. build_retrievers()를 먼저 호출하세요.")
        
        # 1. 앙상블 검색으로 더 많은 후보 검색
        raw_results = self.ensemble_retriever.invoke(query)
        
        # 2. 키워드 기반 필터링 및 점수 계산
        scored_results = self._score_documents(query, raw_results)
        
        # 3. 점수순 정렬
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # 4. 상위 결과 반환
        return [doc for score, doc in scored_results[:max_results]]
    
    def _score_documents(self, query: str, documents: List[Document]) -> List[Tuple[float, Document]]:
        """문서 점수 계산"""
        scored_docs = []
        query_keywords = self._extract_query_keywords(query)
        
        for doc in documents:
            score = 0.0
            content_lower = doc.page_content.lower()
            
            # 1. 키워드 매칭 점수 (가장 중요)
            keyword_matches = 0
            for keyword in query_keywords:
                if keyword.lower() in content_lower:
                    keyword_matches += 1
                    # 정확한 매칭에 높은 점수
                    if keyword.lower() == keyword.lower():  # 완전 일치
                        score += 10
                    else:
                        score += 5
            
            # 2. 키워드 밀도 점수
            density = doc.metadata.get('keyword_density', 0)
            score += density * 20
            
            # 3. 전략별 보너스 점수
            strategy = doc.metadata.get('source_strategy', '')
            if 'keyword' in strategy:
                score += 5
            
            # 4. 위치 점수 (문서 앞부분에 키워드가 있으면 가점)
            first_half = content_lower[:len(content_lower)//2]
            if any(kw.lower() in first_half for kw in query_keywords):
                score += 8
            
            # 5. 길이 적정성 점수
            doc_length = len(doc.page_content.split())
            if 50 <= doc_length <= 300:  # 적정 길이
                score += 3
            
            doc.metadata['search_score'] = score
            doc.metadata['keyword_matches'] = keyword_matches
            scored_docs.append((score, doc))
        
        return scored_docs
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """쿼리에서 중요 키워드 추출"""
        # 기술 용어 패턴
        tech_patterns = [
            r'\b[A-Z]{2,}\b',  # PNS, API 등
            r'\b[a-z]+[A-Z][a-zA-Z]*\b',  # purchaseState 등
        ]
        
        keywords = []
        for pattern in tech_patterns:
            keywords.extend(re.findall(pattern, query))
        
        # 한글 키워드 추가
        korean_keywords = ['메시지', '규격', '값', '구성', '상태', '결제', '서버']
        for keyword in korean_keywords:
            if keyword in query:
                keywords.append(keyword)
        
        return list(set(keywords))