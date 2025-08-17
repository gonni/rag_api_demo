"""
Contextual Retrieval Class Implementation

이 모듈은 문서의 컨텍스트 정보를 생성하여 검색 품질을 향상시키는 
ContextualRetriever 클래스를 제공합니다.
"""

import os
from pathlib import Path
from typing import Optional, List
from langchain.docstore.document import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ContextualRetriever:
    """Contextual Retrieval을 위한 클래스"""
    
    def __init__(self, whole_document: str, model_name: str = "exaone3.5:latest", temperature: float = 0.1):
        """
        ContextualRetriever 초기화
        
        Args:
            whole_document (str): 전체 문서 내용
            model_name (str): 사용할 LLM 모델명 (기본값: "exaone3.5:latest")
            temperature (float): LLM temperature 설정 (기본값: 0.1)
        """
        self.whole_document = whole_document
        self.model_name = model_name
        self.temperature = temperature
        
        # LLM 초기화
        self._initialize_llm()
        
        # 프롬프트 템플릿 설정
        self._setup_prompt()
        
        # 체인 구성
        self._setup_chain()
        
        print(f"🚀 ContextualRetriever 초기화 완료")
        print(f"📄 전체 문서 길이: {len(self.whole_document):,} characters")
        print(f"🤖 모델: {self.model_name}")
        print(f"🌡️ Temperature: {self.temperature}")
        
    def _initialize_llm(self):
        """LLM 초기화"""
        print(f"🤖 LLM 초기화 중... ({self.model_name})")
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature
        )
        
    def _setup_prompt(self):
        """프롬프트 템플릿 설정"""
        print("📝 프롬프트 템플릿 설정 중...")
        self.contextual_prompt = PromptTemplate.from_template(
            """<document> 
{WHOLE_DOCUMENT} 
</document> 
다음은 chunk 처리된 Document입니다. chunk의 내용은 전체 문서에서 일부분을 의미합니다.
<chunk> 
{CHUNK_CONTENT}
</chunk> 
- document의 맥락에서 chunk를 간단 명료하게 요약해주세요.
- 요약문의 최대 토큰은 150 이하여야 합니다. 
- 요약문의 목적은 Document 내에 재삽입하여 retriever를 통합 검색 품질을 높이기 위함입니다.
- 요약문은 한글로 작성해 주세요. 단, chunk내에 코드명, 영문 이니셜 혹은 영어 표현이 문서를 요약하는데 반드시 필요한 내용이라면 영어 그대로 포함될 수 있습니다.
- 요약문을 대표할 수 있는 용어(예, 영문 이니셜, 기능명 등)는 요약문 내에 포함해 주세요.
- 주요 코드값은 한국어로 변환하지 마세요.
- 요약문의 끝부분에 문서를 대표할 수 있는 키워드를 추가해주세요.
"""
        )
        
    def _setup_chain(self):
        """체인 구성"""
        print("⛓️ 체인 구성 중...")
        self.chain = self.contextual_prompt | self.llm | StrOutputParser()
        
    def get_contextual_text(self, chunk_content: str, verbose: bool = True) -> Optional[Document]:
        """
        주어진 청크에 대한 컨텍스트 정보를 생성하고 향상된 Document를 반환
        
        Args:
            chunk_content (str): 처리할 청크 내용
            verbose (bool): 상세 로그 출력 여부 (기본값: True)
            
        Returns:
            Document: 향상된 Document 객체 (성공시) 또는 None (실패시)
        """
        if verbose:
            print(f"\n🔍 컨텍스트 정보 생성 시작...")
            print(f"📊 청크 길이: {len(chunk_content):,} characters")
        
        try:
            # 컨텍스트 정보 생성
            context = self.chain.invoke({
                "WHOLE_DOCUMENT": self.whole_document,
                "CHUNK_CONTENT": chunk_content
            })
            
            if verbose:
                print("=" * 50)
                print(f"✅ 생성된 컨텍스트:\n{context}")
                print("=" * 50)
            
            # 향상된 Document 생성
            enhanced_content = f"[Abstract]: {context}\n\n[Origin]:{chunk_content}"
            
            enhanced_doc = Document(
                page_content=enhanced_content,
                metadata={
                    "original_content": chunk_content,
                    "contextual_info": context,
                    "source": "contextual_retrieval",
                    "model": self.model_name,
                    "original_length": len(chunk_content),
                    "enhanced_length": len(enhanced_content)
                }
            )
            
            if verbose:
                print(f"✅ 향상된 Document 생성 완료!")
                print(f"📏 원본 길이: {len(chunk_content):,} characters")
                print(f"📏 향상된 길이: {len(enhanced_doc.page_content):,} characters")
                print(f"📈 증가율: {(len(enhanced_doc.page_content) / len(chunk_content) - 1) * 100:.1f}%")
            
            return enhanced_doc
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            return None
    
    def process_multiple_chunks(self, chunks: list, verbose: bool = False) -> list:
        """
        여러 청크를 일괄 처리
        
        Args:
            chunks (list): 처리할 청크들의 리스트
            verbose (bool): 상세 로그 출력 여부
            
        Returns:
            list: 향상된 Document 객체들의 리스트
        """
        print(f"🔄 {len(chunks)}개 청크 일괄 처리 시작...")
        
        enhanced_docs = []
        for i, chunk in enumerate(chunks, 1):
            print(f"\n[{i}/{len(chunks)}] 청크 처리 중...")
            enhanced_doc = self.get_contextual_text(chunk, verbose=verbose)
            if enhanced_doc:
                enhanced_docs.append(enhanced_doc)
            else:
                print(f"⚠️ 청크 {i} 처리 실패")
        
        print(f"\n✅ 일괄 처리 완료: {len(enhanced_docs)}/{len(chunks)} 성공")
        return enhanced_docs
    
    def preview_enhancement(self, chunk_content: str, max_chars: int = 500):
        """
        향상된 Document의 미리보기를 출력
        
        Args:
            chunk_content (str): 처리할 청크 내용
            max_chars (int): 미리보기 최대 문자 수
        """
        enhanced_doc = self.get_contextual_text(chunk_content, verbose=False)
        if enhanced_doc:
            print("\n📄 향상된 Document 미리보기:")
            print("-" * 60)
            preview_text = enhanced_doc.page_content[:max_chars]
            if len(enhanced_doc.page_content) > max_chars:
                preview_text += "..."
            print(preview_text)
            print("-" * 60)
        else:
            print("❌ Document 향상 실패")
    
    def get_stats(self) -> dict:
        """
        클래스 인스턴스의 통계 정보를 반환
        
        Returns:
            dict: 통계 정보 딕셔너리
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "document_length": len(self.whole_document),
            "document_size_mb": len(self.whole_document) / (1024 * 1024),
        }
    
    def __repr__(self) -> str:
        """클래스의 문자열 표현"""
        return f"ContextualRetriever(model='{self.model_name}', doc_size={len(self.whole_document):,} chars)"


def create_contextual_retriever(file_path: str, **kwargs) -> ContextualRetriever:
    """
    파일에서 문서를 로드하여 ContextualRetriever 인스턴스를 생성하는 헬퍼 함수
    
    Args:
        file_path (str): 로드할 파일 경로
        **kwargs: ContextualRetriever 생성자에 전달할 추가 인수
        
    Returns:
        ContextualRetriever: 초기화된 ContextualRetriever 인스턴스
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            whole_document = file.read()
        
        print(f"📁 파일 로드 완료: {file_path}")
        print(f"📄 문서 크기: {len(whole_document):,} characters")
        
        return ContextualRetriever(whole_document, **kwargs)
        
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        raise
    except Exception as e:
        print(f"❌ 파일 로드 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    # 모듈이 직접 실행될 때의 테스트 코드
    print("ContextualRetriever 모듈 테스트")
    
    # 간단한 테스트 문서
    test_document = """
    원스토어 인앱결제 가이드
    
    1. 개요
    원스토어는 모바일 앱 개발자들을 위한 결제 시스템을 제공합니다.
    
    2. PNS (Payment Notification Service)
    PNS는 결제 알림 서비스입니다.
    """
    
    # 테스트 청크
    test_chunk = "PNS는 Payment Notification Service의 약자로 결제 알림을 제공합니다."
    
    try:
        # ContextualRetriever 인스턴스 생성
        retriever = ContextualRetriever(test_document)
        
        # 테스트 실행
        result = retriever.get_contextual_text(test_chunk, verbose=False)
        
        if result:
            print("✅ 테스트 성공!")
            print(f"📊 통계: {retriever.get_stats()}")
        else:
            print("❌ 테스트 실패!")
            
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
