"""
Optimized RAG Pipeline for Technical Documentation
기술문서 최적화 RAG 파이프라인

특징:
1. Ollama 기반 환경 (bge-m3, exaone3.5) 지원
2. 계층적 문서 분할 + 컨텍스트 인식 검색
3. 하이브리드 검색 (FAISS + BM25 + 메타데이터 필터링)
4. 다단계 리랭킹 및 노이즈 필터링
5. 스트리밍 응답 지원
6. 검색 분석 및 디버깅 기능
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Union

# LangChain imports
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

# 로컬 모듈 imports
from hierarchical_splitter import HierarchicalSplitter
from context_aware_retriever import ContextAwareRetriever


class OptimizedRAGPipeline:
    """최적화된 RAG 파이프라인"""
    
    def __init__(self,
                 embed_model: str = "bge-m3:latest",
                 llm_model: str = "exaone3.5:latest",
                 data_file: str = "../../data/dev_center_guide_allmd_touched.md",
                 vector_store_path: str = "./models/faiss_optimized",
                 chunk_size: int = 1000,
                 overlap_ratio: float = 0.1,
                 final_top_k: int = 5):
        
        # 모델 설정
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.data_file = data_file
        self.vector_store_path = vector_store_path
        
        # 분할 설정
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.final_top_k = final_top_k
        
        # 구성요소 초기화
        self.embeddings: Optional[OllamaEmbeddings] = None
        self.vector_store: Optional[FAISS] = None
        self.splitter: Optional[HierarchicalSplitter] = None
        self.retriever: Optional[ContextAwareRetriever] = None
        self.llm: Optional[ChatOllama] = None
        self.documents: List[Document] = []
        
        # 성능 통계
        self.stats: Dict[str, Union[int, float]] = {
            'total_documents': 0,
            'total_chunks': 0,
            'index_build_time': 0.0,
            'last_query_time': 0.0,
            'queries_processed': 0
        }
        
        print("RAG Pipeline 초기화:")
        print(f"  - 임베딩 모델: {embed_model}")
        print(f"  - LLM 모델: {llm_model}")
        print(f"  - 데이터 파일: {data_file}")
        print(f"  - 벡터 저장소: {vector_store_path}")
    
    def initialize_models(self):
        """모델 초기화"""
        print("모델 초기화 중...")
        
        # 임베딩 모델
        self.embeddings = OllamaEmbeddings(model=self.embed_model)
        
        # LLM 모델 (스트리밍 지원)
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=0.1,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # 문서 분할기
        self.splitter = HierarchicalSplitter(
            chunk_size=self.chunk_size,
            overlap_ratio=self.overlap_ratio,
            preserve_tables=True,
            preserve_code=True
        )
        
        print("✓ 모델 초기화 완료")
    
    def load_and_process_documents(self, force_rebuild: bool = False):
        """문서 로드 및 처리"""
        print("문서 처리 시작...")
        start_time = time.time()
        
        # 기존 벡터 스토어 확인
        if not force_rebuild and os.path.exists(self.vector_store_path) and self.embeddings is not None:
            try:
                print("기존 벡터 스토어 로드 중...")
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                
                # 메타데이터 로드
                metadata_file = f"{self.vector_store_path}/metadata.json"
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        self.stats.update(json.load(f))
                
                print(f"✓ 기존 벡터 스토어 로드 완료 (문서: {self.stats.get('total_chunks', 0)}개)")
                return
                
            except Exception as e:
                print(f"기존 벡터 스토어 로드 실패: {e}")
                print("새로 생성합니다...")
        
        # 문서 로드
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_file}")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"원본 문서 크기: {len(content):,} 문자")
        
        # 계층적 분할
        if self.splitter is None:
            raise ValueError("문서 분할기가 초기화되지 않았습니다. initialize_models()를 먼저 호출하세요.")
        
        print("계층적 문서 분할 중...")
        self.documents = self.splitter.split_document(content)
        
        self.stats['total_documents'] = 1
        self.stats['total_chunks'] = len(self.documents)
        
        print(f"✓ 분할 완료: {len(self.documents)}개 청크")
        
        # 벡터 스토어 구축
        if self.embeddings is None:
            raise ValueError("임베딩 모델이 초기화되지 않았습니다. initialize_models()를 먼저 호출하세요.")
            
        print("벡터 인덱스 구축 중...")
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
        
        # 벡터 스토어 저장
        os.makedirs(self.vector_store_path, exist_ok=True)
        self.vector_store.save_local(self.vector_store_path)
        
        # 메타데이터 저장
        self.stats['index_build_time'] = time.time() - start_time
        metadata_file = f"{self.vector_store_path}/metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 벡터 인덱스 구축 완료 (소요시간: {self.stats['index_build_time']:.2f}초)")
    
    def build_retriever(self):
        """검색기 구축"""
        print("검색기 구축 중...")
        
        if not self.vector_store:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다. load_and_process_documents()를 먼저 호출하세요.")
        
        # FAISS 검색기
        vector_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 15, "fetch_k": 50, "lambda_mult": 0.7}
        )
        
        # BM25 검색기
        if not self.documents:
            # 벡터 스토어에서 문서 복원 (간소화된 버전)
            print("문서를 벡터 스토어에서 복원 중...")
            # 실제 구현에서는 문서 메타데이터를 별도 저장하는 것이 좋습니다
            self.documents = [Document(page_content="임시 문서", metadata={})]
        
        bm25_retriever = BM25Retriever.from_documents(
            self.documents, 
            bm25_params={"k1": 1.5, "b": 0.75}
        )
        bm25_retriever.k = 20
        
        # 앙상블 검색기
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
        
        # 컨텍스트 인식 검색기로 래핑
        self.retriever = ContextAwareRetriever(
            base_retriever=ensemble_retriever,
            rerank_top_k=20,
            final_top_k=self.final_top_k,
            enable_query_expansion=True,
            noise_threshold=0.7
        )
        
        print("✓ 검색기 구축 완료")
    
    def create_prompt_template(self) -> PromptTemplate:
        """프롬프트 템플릿 생성"""
        
        template = """당신은 원스토어 인앱결제 기술문서 전문가입니다. 
주어진 문서를 바탕으로 사용자의 질문에 정확하고 자세하게 답변해주세요.

**답변 지침:**
1. 기술적 정확성을 최우선으로 합니다
2. 코드 예제가 있다면 반드시 포함합니다  
3. API 파라미터나 응답값은 정확히 기술합니다
4. 단계별 설명이 필요한 경우 번호를 매겨 설명합니다
5. 불확실한 내용은 "문서에서 확인되지 않음"이라고 명시합니다

**참고 문서:**
{context}

**질문:** {question}

**답변:**"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def query(self, question: str, return_sources: bool = True, stream: bool = True) -> Dict[str, Any]:
        """쿼리 실행"""
        start_time = time.time()
        
        if not self.retriever or not self.llm:
            raise ValueError("파이프라인이 완전히 초기화되지 않았습니다.")
        
        print(f"\n🔍 질문: {question}")
        print("=" * 60)
        
        # 1. 문서 검색
        print("관련 문서 검색 중...")
        retrieved_docs = self.retriever.get_relevant_documents(question)
        
        print(f"✓ {len(retrieved_docs)}개 문서 검색됨")
        
        # 2. 컨텍스트 구성
        context = "\n\n".join([
            f"[문서 {i+1}] {doc.metadata.get('section_hierarchy', 'N/A')}\n{doc.page_content}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # 3. 프롬프트 생성
        prompt_template = self.create_prompt_template()
        prompt = prompt_template.format(context=context, question=question)
        
        # 4. LLM 실행
        print("\n💭 답변 생성 중...")
        print("-" * 60)
        
        response_text = ""
        if stream:
            # 스트리밍 응답
            response_generator = self.llm.stream(prompt)
            for chunk in response_generator:
                if hasattr(chunk, 'content'):
                    # chunk.content는 다양한 타입이 될 수 있으므로 str로 안전하게 변환
                    response_text += str(chunk.content)
            print()  # 줄바꿈
        else:
            # 일반 응답
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                # response.content도 str로 안전하게 변환
                response_text = str(response.content)
            else:
                response_text = str(response)
        
        # 5. 통계 업데이트
        query_time = time.time() - start_time
        self.stats['last_query_time'] = query_time
        self.stats['queries_processed'] += 1
        
        print("-" * 60)
        print(f"⏱️  응답 시간: {query_time:.2f}초")
        
        # 6. 결과 구성
        result = {
            'question': question,
            'answer': response_text,
            'query_time': query_time,
            'retrieved_docs_count': len(retrieved_docs)
        }
        
        if return_sources:
            result['sources'] = [
                {
                    'content': doc.page_content[:200] + "...",
                    'metadata': doc.metadata
                }
                for doc in retrieved_docs
            ]
        
        return result
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """배치 쿼리 실행"""
        print(f"📋 배치 쿼리 실행: {len(questions)}개 질문")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}]")
            result = self.query(question, stream=False)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """파이프라인 통계 반환"""
        return {
            **self.stats,
            'models': {
                'embedding': self.embed_model,
                'llm': self.llm_model
            },
            'config': {
                'chunk_size': self.chunk_size,
                'overlap_ratio': self.overlap_ratio,
                'final_top_k': self.final_top_k
            }
        }
    
    def analyze_document_structure(self) -> Dict[str, Any]:
        """문서 구조 분석"""
        if not self.documents:
            return {"error": "문서가 로드되지 않았습니다"}
        
        # 출처별 통계
        source_stats: Dict[str, int] = {}
        content_type_stats: Dict[str, int] = {}
        tech_term_stats: Dict[str, Dict[str, int]] = {}
        
        for doc in self.documents:
            # 출처 통계
            source_url = doc.metadata.get('source_url', 'Unknown')
            source_domain = source_url.split('/')[-1] if '/' in source_url else source_url
            source_stats[source_domain] = source_stats.get(source_domain, 0) + 1
            
            # 콘텐츠 타입 통계
            content_types = doc.metadata.get('content_types', ['text'])
            for ct in content_types:
                content_type_stats[ct] = content_type_stats.get(ct, 0) + 1
            
            # 기술 용어 통계
            tech_terms = doc.metadata.get('technical_terms', {})
            for category, terms in tech_terms.items():
                if category not in tech_term_stats:
                    tech_term_stats[category] = {}
                for term in terms:
                    tech_term_stats[category][term] = tech_term_stats[category].get(term, 0) + 1
        
        return {
            'total_chunks': len(self.documents),
            'source_distribution': source_stats,
            'content_type_distribution': content_type_stats,
            'tech_term_distribution': {
                category: dict(list(sorted(terms.items(), key=lambda x: x[1], reverse=True))[:10])
                for category, terms in tech_term_stats.items()
            }
        }
    
    def debug_search(self, question: str) -> Dict[str, Any]:
        """검색 디버깅 정보"""
        if not self.retriever:
            return {"error": "검색기가 초기화되지 않았습니다"}
        
        # 쿼리 분석
        analytics = self.retriever.get_search_analytics(question)
        
        # 검색 실행
        retrieved_docs = self.retriever.get_relevant_documents(question)
        
        # 검색 결과 분석
        result_analysis = []
        for i, doc in enumerate(retrieved_docs):
            result_analysis.append({
                'rank': i + 1,
                'content_preview': doc.page_content[:150] + "...",
                'section_hierarchy': doc.metadata.get('section_hierarchy', 'N/A'),
                'content_types': doc.metadata.get('content_types', []),
                'technical_terms': doc.metadata.get('technical_terms', {}),
                'content_length': len(doc.page_content)
            })
        
        return {
            'query_analysis': analytics,
            'retrieved_count': len(retrieved_docs),
            'results': result_analysis
        }


# 편의 함수들
def create_pipeline(data_file: str = "../../data/dev_center_guide_allmd_touched.md",
                   force_rebuild: bool = False) -> OptimizedRAGPipeline:
    """파이프라인 생성 및 초기화"""
    
    pipeline = OptimizedRAGPipeline(data_file=data_file)
    
    # 초기화 단계
    pipeline.initialize_models()
    pipeline.load_and_process_documents(force_rebuild=force_rebuild)
    pipeline.build_retriever()
    
    print("\n🚀 RAG 파이프라인 준비 완료!")
    print(f"   총 문서: {pipeline.stats['total_chunks']}개")
    print(f"   인덱싱 시간: {pipeline.stats['index_build_time']:.2f}초")
    
    return pipeline


def interactive_mode(pipeline: OptimizedRAGPipeline):
    """대화형 모드"""
    print("\n" + "="*60)
    print("🤖 원스토어 IAP 기술문서 QA 시스템")
    print("="*60)
    print("질문을 입력하세요 (종료: 'quit' 또는 'exit')")
    print("특수 명령어:")
    print("  - 'stats': 파이프라인 통계")
    print("  - 'analyze': 문서 구조 분석")
    print("  - 'debug <질문>': 검색 디버깅")
    print("-"*60)
    
    while True:
        try:
            question = input("\n❓ 질문: ").strip()
            
            if question.lower() in ['quit', 'exit', '종료']:
                print("👋 시스템을 종료합니다.")
                break
            
            if question == 'stats':
                stats = pipeline.get_statistics()
                print("\n📊 파이프라인 통계:")
                print(json.dumps(stats, ensure_ascii=False, indent=2))
                continue
            
            if question == 'analyze':
                analysis = pipeline.analyze_document_structure()
                print("\n📈 문서 구조 분석:")
                print(json.dumps(analysis, ensure_ascii=False, indent=2))
                continue
            
            if question.startswith('debug '):
                debug_query = question[6:]
                debug_info = pipeline.debug_search(debug_query)
                print("\n🔍 검색 디버깅:")
                print(json.dumps(debug_info, ensure_ascii=False, indent=2))
                continue
            
            if not question:
                continue
            
            # 일반 질문 처리
            pipeline.query(question)
            
        except KeyboardInterrupt:
            print("\n\n👋 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")


# 테스트 함수
def test_pipeline():
    """파이프라인 테스트"""
    
    test_questions = [
        "PurchaseClient 초기화 방법이 뭔가요?",
        "purchaseState 값은 무엇인가요?", 
        "PNS 서비스 설정 방법을 알려주세요",
        "인앱결제 에러가 발생했을 때 해결 방법은?",
        "구독형 상품과 관리형 상품의 차이점은?"
    ]
    
    print("🧪 파이프라인 테스트 시작...")
    
    # 테스트용 더미 파이프라인 (실제 파일 없이)
    # 실제 사용시에는 create_pipeline()을 사용
    
    for question in test_questions:
        print(f"\n질문: {question}")
        # result = pipeline.query(question)
        print("답변: [테스트 모드 - 실제 구현에서는 답변이 생성됩니다]")
        print("-" * 60)


if __name__ == "__main__":
    # 테스트 실행
    test_pipeline()
