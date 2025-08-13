"""
컨텍스트 인식 RAG 시스템

이 모듈은 문서의 컨텍스트를 고려하여 더 정확한 검색과 답변을 제공하는
Context-Aware RAG 시스템을 구현합니다.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate


class ContextAwareRAG:
    """컨텍스트 인식 RAG 시스템"""
    
    def __init__(self, documents: List[Document], embedding_model: str = "bge-m3:latest"):
        self.documents = documents
        self.embedding_model = embedding_model
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.llm = None
        
    def setup(self):
        """시스템 초기화"""
        print("🚀 컨텍스트 인식 RAG 시스템 초기화...")
        
        # 검색기 구축
        self._build_retrievers()
        
        # LLM 초기화
        self._initialize_llm()
        
        # 컨텍스트 분석
        self._analyze_contexts()
        
        print("✅ 컨텍스트 인식 RAG 시스템 초기화 완료")
    
    def _build_retrievers(self):
        """검색기 구축"""
        print("🔧 검색기 구축 중...")
        
        # Vector store 구축
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.vector_store = FAISS.from_documents(self.documents, embeddings)
        
        # BM25 검색기 구축
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 30
        
        # 앙상블 검색기
        vector_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "fetch_k": 50, "lambda_mult": 0.7}
        )
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[0.6, 0.4]
        )
        
        print("✅ 검색기 구축 완료")
    
    def _initialize_llm(self):
        """LLM 초기화"""
        self.llm = ChatOllama(model="exaone3.5:latest", temperature=0.3)
    
    def _analyze_contexts(self):
        """문서 컨텍스트 분석"""
        print("📊 문서 컨텍스트 분석 중...")
        
        # PNS 관련 문서 식별
        pns_docs = []
        for doc in self.documents:
            if self._is_pns_related(doc):
                pns_docs.append(doc)
        
        print(f"  PNS 관련 문서: {len(pns_docs)}개")
        
        # 컨텍스트 그룹핑
        self.context_groups = self._group_by_context(pns_docs)
        
        for group_name, docs in self.context_groups.items():
            print(f"  {group_name}: {len(docs)}개 문서")
    
    def _is_pns_related(self, doc: Document) -> bool:
        """PNS 관련 문서 여부 확인"""
        content_lower = doc.page_content.lower()
        metadata = doc.metadata
        
        # 메타데이터에서 확인
        if metadata.get('contains_pns', False):
            return True
        
        # 내용에서 확인
        pns_keywords = ['pns', 'payment notification', '결제알림', '메시지 규격']
        return any(keyword in content_lower for keyword in pns_keywords)
    
    def _group_by_context(self, docs: List[Document]) -> Dict[str, List[Document]]:
        """컨텍스트별 그룹핑"""
        groups = {
            'message_specifications': [],
            'purchase_state_info': [],
            'signature_verification': [],
            'general_pns': []
        }
        
        for doc in docs:
            content_lower = doc.page_content.lower()
            
            if 'purchasestate' in content_lower:
                groups['purchase_state_info'].append(doc)
            elif 'signature' in content_lower:
                groups['signature_verification'].append(doc)
            elif 'element name' in content_lower or 'parameter name' in content_lower:
                groups['message_specifications'].append(doc)
            else:
                groups['general_pns'].append(doc)
        
        return groups
    
    def query(self, question: str, max_context_docs: int = 8) -> Dict[str, Any]:
        """컨텍스트 인식 질의 처리"""
        print(f"🔍 질의 처리: {question}")
        
        # 1. 컨텍스트 분석
        context_type = self._analyze_query_context(question)
        print(f"  컨텍스트 타입: {context_type}")
        
        # 2. 컨텍스트별 검색
        relevant_docs = self._context_aware_search(question, context_type, max_context_docs)
        
        # 3. 컨텍스트 강화
        enhanced_context = self._enhance_context(relevant_docs, context_type)
        
        # 4. 답변 생성
        answer = self._generate_answer(question, enhanced_context)
        
        return {
            'question': question,
            'context_type': context_type,
            'relevant_docs': relevant_docs,
            'answer': answer,
            'context_info': {
                'total_docs': len(relevant_docs),
                'context_groups': [doc.metadata.get('content_type', 'unknown') for doc in relevant_docs]
            }
        }
    
    def _analyze_query_context(self, question: str) -> str:
        """질의 컨텍스트 분석"""
        question_lower = question.lower()
        
        if 'purchasestate' in question_lower or 'purchase state' in question_lower:
            return 'purchase_state'
        elif 'signature' in question_lower:
            return 'signature_verification'
        elif '메시지' in question or 'message' in question_lower:
            return 'message_specification'
        elif 'pns' in question_lower or 'payment notification' in question_lower:
            return 'general_pns'
        else:
            return 'general'
    
    def _context_aware_search(self, question: str, context_type: str, max_docs: int) -> List[Document]:
        """컨텍스트 인식 검색"""
        # 1. 기본 앙상블 검색
        base_results = self.ensemble_retriever.invoke(question)
        
        # 2. 컨텍스트별 우선순위 적용
        if context_type in self.context_groups:
            context_docs = self.context_groups[context_type]
            
            # 컨텍스트 문서 우선 선택
            prioritized_docs = []
            for doc in context_docs:
                if doc in base_results:
                    prioritized_docs.append(doc)
            
            # 나머지 문서 추가
            remaining_docs = [doc for doc in base_results if doc not in prioritized_docs]
            prioritized_docs.extend(remaining_docs)
            
            return prioritized_docs[:max_docs]
        
        return base_results[:max_docs]
    
    def _enhance_context(self, docs: List[Document], context_type: str) -> str:
        """컨텍스트 강화"""
        enhanced_parts = []
        
        # 컨텍스트 타입별 설명 추가
        context_descriptions = {
            'purchase_state': "이 질의는 PNS 메시지의 purchaseState 필드와 관련된 내용입니다.",
            'signature_verification': "이 질의는 PNS 메시지의 서명 검증과 관련된 내용입니다.",
            'message_specification': "이 질의는 PNS 메시지 규격과 관련된 내용입니다.",
            'general_pns': "이 질의는 PNS(Payment Notification Service) 일반 정보와 관련된 내용입니다."
        }
        
        if context_type in context_descriptions:
            enhanced_parts.append(context_descriptions[context_type])
        
        # 문서 내용 추가
        for i, doc in enumerate(docs):
            doc_content = doc.page_content.strip()
            if doc_content:
                enhanced_parts.append(f"[문서 {i+1}]: {doc_content}")
        
        return "\n\n".join(enhanced_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """답변 생성"""
        prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""
당신은 PNS(Payment Notification Service) 전문가입니다. 
주어진 컨텍스트를 바탕으로 질문에 정확하고 상세한 답변을 제공해주세요.

질문: {question}

컨텍스트:
{context}

답변을 한국어로 작성하고, 가능한 한 구체적이고 정확한 정보를 제공해주세요.
"""
        )
        
        chain = prompt_template | self.llm
        
        try:
            response = chain.invoke({"question": question, "context": context})
            return response.content
        except Exception as e:
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"


class PNSQueryAnalyzer:
    """PNS 질의 분석기"""
    
    def __init__(self):
        self.query_patterns = {
            'purchase_state': [
                r'purchasestate.*?값',
                r'purchase.*?state.*?무엇',
                r'결제.*?상태.*?값',
                r'purchasestate.*?포함'
            ],
            'message_specification': [
                r'메시지.*?규격',
                r'message.*?specification',
                r'pns.*?메시지.*?구성',
                r'요청.*?body.*?구성'
            ],
            'signature_verification': [
                r'signature.*?검증',
                r'서명.*?검증',
                r'signature.*?verification'
            ],
            'general_pns': [
                r'pns.*?무엇',
                r'payment.*?notification.*?service',
                r'결제알림.*?서비스'
            ]
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """질의 분석"""
        query_lower = query.lower()
        
        analysis = {
            'query_type': 'unknown',
            'confidence': 0.0,
            'keywords': [],
            'suggestions': []
        }
        
        # 패턴 매칭
        max_matches = 0
        best_type = 'unknown'
        
        for query_type, patterns in self.query_patterns.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    matches += 1
            
            if matches > max_matches:
                max_matches = matches
                best_type = query_type
        
        analysis['query_type'] = best_type
        analysis['confidence'] = max_matches / len(self.query_patterns[best_type]) if best_type != 'unknown' else 0.0
        
        # 키워드 추출
        analysis['keywords'] = self._extract_keywords(query)
        
        # 제안사항
        analysis['suggestions'] = self._generate_suggestions(best_type, query)
        
        return analysis
    
    def _extract_keywords(self, query: str) -> List[str]:
        """키워드 추출"""
        keywords = []
        
        # 기술 용어
        tech_patterns = [
            r'\b[A-Z]{2,}\b',  # PNS, API 등
            r'\b[a-z]+[A-Z][a-zA-Z]*\b',  # purchaseState 등
        ]
        
        for pattern in tech_patterns:
            keywords.extend(re.findall(pattern, query))
        
        # 한글 키워드
        korean_keywords = ['메시지', '규격', '값', '구성', '상태', '결제', '서버', '알림']
        for keyword in korean_keywords:
            if keyword in query:
                keywords.append(keyword)
        
        return list(set(keywords))
    
    def _generate_suggestions(self, query_type: str, query: str) -> List[str]:
        """제안사항 생성"""
        suggestions = []
        
        if query_type == 'purchase_state':
            suggestions.extend([
                "purchaseState 필드의 가능한 값들을 확인해보세요",
                "COMPLETED, CANCELED 등의 상태값을 찾아보세요"
            ])
        elif query_type == 'message_specification':
            suggestions.extend([
                "메시지 규격 테이블을 참조하세요",
                "Element Name, Data Type, Description 필드를 확인하세요"
            ])
        elif query_type == 'signature_verification':
            suggestions.extend([
                "서명 검증 방법과 코드 예제를 확인하세요",
                "PublicKey와 signature 필드를 참조하세요"
            ])
        
        return suggestions
