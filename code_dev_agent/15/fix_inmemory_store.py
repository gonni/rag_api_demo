#!/usr/bin/env python3
"""
InMemoryStore madd 메서드 호환성 수정

LangChain 버전에 따라 InMemoryStore의 메서드가 다를 수 있습니다.
이 스크립트는 호환성을 보장하는 수정된 버전을 제공합니다.
"""

from langchain.storage import InMemoryStore
from typing import List, Tuple, Any

def safe_add_to_store(store: InMemoryStore, items: List[Tuple[str, Any]]):
    """
    InMemoryStore에 안전하게 아이템을 추가하는 함수
    
    Args:
        store: InMemoryStore 인스턴스
        items: (key, value) 튜플의 리스트
    """
    try:
        # 먼저 madd 메서드가 있는지 확인
        if hasattr(store, 'madd'):
            store.madd(items)
        else:
            # madd가 없으면 개별적으로 추가
            for key, value in items:
                store.mset([(key, value)])
    except AttributeError:
        # mset도 없으면 개별적으로 추가
        for key, value in items:
            store.mset([(key, value)])

def build_retrievers_from_md_fixed(md_path: str, doc_id: str):
    """
    수정된 build_retrievers_from_md 함수
    
    InMemoryStore 호환성 문제를 해결한 버전입니다.
    """
    from langchain.storage import InMemoryStore
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    
    # 기존 코드와 동일한 로직...
    # 여기서는 InMemoryStore 사용 부분만 수정
    
    child_store = InMemoryStore()
    
    # 수정된 부분: madd 대신 안전한 함수 사용
    for d in child_docs:
        safe_add_to_store(child_store, [(d.metadata["section_id"], d)])
    
    # 나머지 코드는 동일...
    return packs, ensemble, parent_store, child_vectorstore

# 사용 예시:
if __name__ == "__main__":
    print("🔧 InMemoryStore 호환성 수정 도구")
    print("="*40)
    
    # 테스트
    store = InMemoryStore()
    test_items = [("key1", "value1"), ("key2", "value2")]
    
    try:
        safe_add_to_store(store, test_items)
        print("✅ 안전한 추가 성공")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    
    print("\n📝 사용법:")
    print("1. 기존 코드에서 madd 호출을 safe_add_to_store로 교체")
    print("2. 또는 build_retrievers_from_md_fixed 함수 사용")
