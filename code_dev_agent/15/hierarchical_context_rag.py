"""
계층적 맥락 보존 RAG 시스템

이 모듈은 문서의 계층적 구조를 보존하여 검색 시 상위 맥락 정보를 
포함한 정확한 검색 결과를 제공합니다.

문제 해결:
- 단순 제목만으로는 맥락 파악이 어려운 문제
- 상위 섹션 정보가 손실되어 잘못된 정보 제공 위험
- purchaseState 같은 범용 용어의 모듈별 차이점 구분
"""

import re
import uuid
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
import json

from langchain.docstore.document import Document
from langchain.storage import InMemoryStore

if TYPE_CHECKING:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever

# Optional imports for better error handling


try:
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    print("Warning: OllamaEmbeddings not available. Please install langchain-community.")

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
except ImportError:
    print("Warning: FAISS not available. Please install faiss-cpu or faiss-gpu.")

try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    print("Warning: BM25Retriever not available. Please install rank-bm25.")

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    print("Warning: EnsembleRetriever not available.")


@dataclass
class HierarchicalSection:
    """계층적 섹션 정보를 담는 클래스"""
    id: str
    level: int  # 헤더 레벨 (1, 2, 3, ...)
    title: str
    full_path: str  # 전체 경로 (예: "SDK > API Specification")
    content: str
    start_line: int = 0  # 섹션 시작 라인 번호
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)  # 하위 섹션 ID들
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextualDocument:
    """맥락 정보가 포함된 문서"""
    id: str
    content: str
    section_path: str  # 전체 섹션 경로
    section_hierarchy: List[str]  # 계층 구조 (["SDK", "API Specification"])
    parent_context: str  # 상위 맥락 정보
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalContextRAG:
    """계층적 맥락을 보존하는 RAG 시스템"""
    
    def __init__(self, embedding_model: str = "bge-m3:latest"):
        self.embedding_model = embedding_model
        self.sections: Dict[str, HierarchicalSection] = {}
        self.documents: List[ContextualDocument] = []
        self.vector_store: Optional[Any] = None
        self.bm25_retriever: Optional[Any] = None
        self.ensemble_retriever: Optional[Any] = None
        
    def parse_markdown_hierarchy(self, md_text: str, doc_id: str) -> List[HierarchicalSection]:
        """마크다운을 계층적 구조로 파싱"""
        lines = md_text.splitlines()
        sections: List[HierarchicalSection] = []
        current_hierarchy: List[str] = []  # 현재까지의 계층 구조
        current_section = None
        
        for i, line in enumerate(lines):
            # 헤더 매칭
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # 계층 구조 업데이트
                if level == 1:
                    current_hierarchy = [title]
                elif level <= len(current_hierarchy):
                    current_hierarchy = current_hierarchy[:level-1] + [title]
                else:
                    current_hierarchy.append(title)
                
                # 이전 섹션 완료
                if current_section:
                    current_section.content = '\n'.join(lines[current_section.start_line:i])
                    sections.append(current_section)
                
                # 새 섹션 생성
                section_id = str(uuid.uuid4())
                current_section = HierarchicalSection(
                    id=section_id,
                    level=level,
                    title=title,
                    full_path=' > '.join(current_hierarchy),
                    content='',
                    start_line=i+1
                )
                
                # 부모-자식 관계 설정
                if level > 1 and len(current_hierarchy) > 1:
                    parent_path = ' > '.join(current_hierarchy[:-1])
                    for section in sections:
                        if section.full_path == parent_path:
                            current_section.parent_id = section.id
                            section.children.append(section_id)
                            break
        
        # 마지막 섹션 처리
        if current_section:
            current_section.content = '\n'.join(lines[current_section.start_line:])
            sections.append(current_section)
        
        return sections
    
    def create_contextual_documents(self, sections: List[HierarchicalSection]) -> List[ContextualDocument]:
        """계층적 섹션을 맥락 정보가 포함된 문서로 변환"""
        contextual_docs = []
        
        for section in sections:
            # 상위 맥락 정보 생성
            parent_context = self._build_parent_context(section, sections)
            
            # 계층 구조 생성
            hierarchy = section.full_path.split(' > ')
            
            # 문서 생성
            doc = ContextualDocument(
                id=section.id,
                content=section.content,
                section_path=section.full_path,
                section_hierarchy=hierarchy,
                parent_context=parent_context,
                metadata={
                    'section_title': section.title,
                    'section_level': section.level,
                    'parent_id': section.parent_id,
                    'children': section.children
                }
            )
            contextual_docs.append(doc)
        
        return contextual_docs
    
    def _build_parent_context(self, section: HierarchicalSection, all_sections: List[HierarchicalSection]) -> str:
        """상위 맥락 정보 구축"""
        if not section.parent_id:
            return section.title
        
        parent_contexts: List[str] = []
        current_section = section
        
        # 상위 섹션들을 거슬러 올라가며 맥락 구축
        while current_section.parent_id:
            for parent in all_sections:
                if parent.id == current_section.parent_id:
                    parent_contexts.insert(0, f"{parent.title}: {parent.content[:200]}...")
                    current_section = parent
                    break
            else:
                break
        
        return " | ".join(parent_contexts)
    
    def build_search_documents(self, contextual_docs: List[ContextualDocument]) -> List[Document]:
        """검색용 Document 객체 생성"""
        search_docs = []
        
        for doc in contextual_docs:
            # 검색 텍스트에 맥락 정보 포함
            search_content = f"""
제목: {doc.section_path}
상위 맥락: {doc.parent_context}

내용:
{doc.content}
""".strip()
            
            # 메타데이터에 계층 정보 포함
            metadata = {
                **doc.metadata,
                'section_path': doc.section_path,
                'section_hierarchy': doc.section_hierarchy,
                'parent_context': doc.parent_context,
                'context_level': len(doc.section_hierarchy)
            }
            
            search_docs.append(Document(
                page_content=search_content,
                metadata=metadata
            ))
        
        return search_docs
    
    def build_retrievers(self, search_docs: List[Document]):
        """검색기 구축"""
        # 의존성 확인
        if OllamaEmbeddings is None:
            raise ImportError("OllamaEmbeddings가 필요합니다. 'pip install langchain-community'를 실행하세요.")
        if FAISS is None:
            raise ImportError("FAISS가 필요합니다. 'pip install faiss-cpu'를 실행하세요.")
        if BM25Retriever is None:
            raise ImportError("BM25Retriever가 필요합니다. 'pip install rank-bm25'를 실행하세요.")
        if EnsembleRetriever is None:
            raise ImportError("EnsembleRetriever가 필요합니다.")
        
        # 벡터 검색
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.vector_store = FAISS.from_documents(
            search_docs, 
            embeddings, 
            distance_strategy=DistanceStrategy.COSINE
        )
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        
        # BM25 검색 (맥락 정보 강화)
        bm25_docs = []
        for doc in search_docs:
            # BM25용으로 맥락 정보를 더 강화
            enhanced_content = f"""
{doc.page_content}

추가 맥락:
- 섹션 경로: {doc.metadata.get('section_path', '')}
- 계층 구조: {' > '.join(doc.metadata.get('section_hierarchy', []))}
- 상위 맥락: {doc.metadata.get('parent_context', '')}
""".strip()
            
            bm25_docs.append(Document(
                page_content=enhanced_content,
                metadata=doc.metadata
            ))
        
        self.bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        self.bm25_retriever.k = 10
        
        # 앙상블 검색기
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[0.6, 0.4]  # BM25에 더 높은 가중치 (맥락 정보 활용)
        )
    
    def search_with_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """맥락을 고려한 검색"""
        if not self.ensemble_retriever:
            raise ValueError("검색기가 구축되지 않았습니다. build_retrievers()를 먼저 호출하세요.")
        
        # 검색 실행
        docs = self.ensemble_retriever.get_relevant_documents(query)
        
        # 결과에 맥락 정보 추가
        results = []
        for doc in docs:
            result = {
                'content': doc.page_content,
                'section_path': doc.metadata.get('section_path', ''),
                'section_hierarchy': doc.metadata.get('section_hierarchy', []),
                'parent_context': doc.metadata.get('parent_context', ''),
                'context_level': doc.metadata.get('context_level', 0),
                'metadata': doc.metadata
            }
            results.append(result)
        
        return results[:k]
    
    def search_by_module(self, query: str, target_module: str, k: int = 5) -> List[Dict[str, Any]]:
        """특정 모듈 내에서만 검색"""
        all_results = self.search_with_context(query, k * 3)  # 더 많은 결과 가져오기
        
        # 대상 모듈로 필터링
        filtered_results = []
        for result in all_results:
            hierarchy = result['section_hierarchy']
            if hierarchy and hierarchy[0].lower() == target_module.lower():
                filtered_results.append(result)
                if len(filtered_results) >= k:
                    break
        
        return filtered_results
    
    def compare_across_modules(self, query: str, modules: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """여러 모듈에서 동일한 쿼리로 검색하여 비교"""
        comparison = {}
        
        for module in modules:
            results = self.search_by_module(query, module, k=3)
            comparison[module] = results
        
        return comparison


def create_sample_document() -> str:
    """테스트용 샘플 문서 생성"""
    return """
# SDK
## 개요
SDK는 Software Development Kit의 약자로, 개발자가 쉽게 애플리케이션을 개발할 수 있도록 도와주는 도구 모음입니다.

## API Specification
### purchaseState
purchaseState는 결제 상태를 나타내는 열거형입니다.

**Enum 값:**
- SUCCEED: 결제 성공
- FAILED: 결제 실패

### 사용 예시
```java
if (purchaseState == PurchaseState.SUCCEED) {
    // 결제 성공 처리
}
```

## 주의사항
- purchaseState는 결제 완료 후에만 정확한 값을 반환합니다.
- 네트워크 오류 시 FAILED가 반환될 수 있습니다.

# Server to Server API
## 개요
Server to Server API는 서버 간 통신을 위한 RESTful API입니다.

## API Specification
### purchaseState
purchaseState는 결제 상태를 나타내는 JSON 필드입니다.

**JSON 값:**
- COMPLETED: 결제 완료
- FAILED: 결제 실패  
- PROCESSING: 결제 처리 중

### API 응답 예시
```json
{
  "purchaseState": "COMPLETED",
  "transactionId": "12345",
  "amount": 1000
}
```

## 주의사항
- PROCESSING 상태는 일시적이며, 최종적으로 COMPLETED 또는 FAILED로 변경됩니다.
- 웹훅을 통해 상태 변경을 실시간으로 받을 수 있습니다.
"""


def demo_hierarchical_context_rag():
    """계층적 맥락 RAG 시스템 데모"""
    print("🚀 계층적 맥락 RAG 시스템 데모 시작")
    
    try:
        # 1. 시스템 초기화
        rag = HierarchicalContextRAG()
        
        # 2. 샘플 문서 생성 및 파싱
        sample_doc = create_sample_document()
        print("\n📄 샘플 문서 생성 완료")
        
        sections = rag.parse_markdown_hierarchy(sample_doc, "sample_doc")
        print(f"📋 계층적 섹션 파싱 완료: {len(sections)}개 섹션")
        
        # 3. 맥락 정보가 포함된 문서 생성
        contextual_docs = rag.create_contextual_documents(sections)
        print(f"📝 맥락 문서 생성 완료: {len(contextual_docs)}개 문서")
        
        # 4. 검색 문서 생성
        search_docs = rag.build_search_documents(contextual_docs)
        print(f"🔍 검색 문서 생성 완료: {len(search_docs)}개 문서")
        
        # 5. 검색기 구축
        try:
            rag.build_retrievers(search_docs)
            print("🔧 검색기 구축 완료")
        except ImportError as e:
            print(f"⚠️  검색기 구축 실패: {e}")
            print("📝 파싱 및 문서 생성은 완료되었습니다.")
            return
        
        # 6. 테스트 검색
        print("\n" + "="*60)
        print("🧪 검색 테스트")
        print("="*60)
        
        # 테스트 1: 일반 검색
        print("\n1️⃣ 일반 검색: 'purchaseState는 어떤 값들이 있나요?'")
        results = rag.search_with_context("purchaseState는 어떤 값들이 있나요?", k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- 결과 {i} ---")
            print(f"📂 섹션: {result['section_path']}")
            print(f"📄 내용: {result['content'][:200]}...")
        
        # 테스트 2: 모듈별 검색
        print("\n2️⃣ SDK 모듈에서만 검색: 'purchaseState 값'")
        sdk_results = rag.search_by_module("purchaseState 값", "SDK", k=2)
        
        for i, result in enumerate(sdk_results, 1):
            print(f"\n--- SDK 결과 {i} ---")
            print(f"📂 섹션: {result['section_path']}")
            print(f"📄 내용: {result['content'][:200]}...")
        
        # 테스트 3: Server to Server API 모듈에서만 검색
        print("\n3️⃣ Server to Server API 모듈에서만 검색: 'purchaseState 값'")
        server_results = rag.search_by_module("purchaseState 값", "Server to Server API", k=2)
        
        for i, result in enumerate(server_results, 1):
            print(f"\n--- Server to Server API 결과 {i} ---")
            print(f"📂 섹션: {result['section_path']}")
            print(f"📄 내용: {result['content'][:200]}...")
        
        # 테스트 4: 모듈 간 비교
        print("\n4️⃣ 모듈 간 비교: 'purchaseState'")
        comparison = rag.compare_across_modules("purchaseState", ["SDK", "Server to Server API"])
        
        for module, results in comparison.items():
            print(f"\n--- {module} 모듈 결과 ---")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['section_path']}")
                print(f"     {result['content'][:100]}...")
        
        print("\n✅ 데모 완료!")
        
    except Exception as e:
        print(f"❌ 데모 실행 중 오류 발생: {e}")
        print("🔧 의존성 설치가 필요할 수 있습니다:")
        print("   pip install langchain-community faiss-cpu rank-bm25")


if __name__ == "__main__":
    demo_hierarchical_context_rag()
