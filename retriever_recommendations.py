"""
EnsembleRetriever 검색 품질 향상을 위한 추가 검색 방식들

BM25 외에 고려할 수 있는 다양한 검색 방식을 제안합니다.
"""

# 1. MultiQueryRetriever - 다양한 쿼리 변형으로 검색
def create_multi_query_retriever(loaded_db, llm):
    """
    하나의 질문을 여러 변형으로 만들어 검색 품질을 향상시킵니다.
    """
    from langchain.retrievers import MultiQueryRetriever
    from langchain_core.prompts import PromptTemplate
    
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""주어진 질문을 기반으로 3개의 다른 방식으로 질문을 변형해주세요.
        각 변형은 원래 질문의 의미를 유지하면서 다른 표현을 사용해야 합니다.
        
        원래 질문: {question}
        
        변형된 질문들:"""
    )
    
    return MultiQueryRetriever.from_llm(
        retriever=loaded_db.as_retriever(search_kwargs={"k": 10}),
        llm=llm,
        prompt=query_prompt,
        parser_key="text"
    )

# 2. ContextualCompressionRetriever - 컨텍스트 압축으로 관련성 향상
def create_contextual_compression_retriever(loaded_db, llm):
    """
    검색된 문서에서 질문과 관련된 부분만 추출하여 관련성을 높입니다.
    """
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain_core.prompts import PromptTemplate
    
    compressor_prompt = """다음 문서에서 질문과 관련된 정보만 추출해주세요.
    질문: {question}
    문서: {context}
    관련 정보:"""
    
    compressor = LLMChainExtractor.from_llm(
        llm=llm,
        prompt=PromptTemplate.from_template(compressor_prompt)
    )
    
    return ContextualCompressionRetriever(
        base_retriever=loaded_db.as_retriever(search_kwargs={"k": 20}),
        base_compressor=compressor
    )

# 3. SelfQueryRetriever - 메타데이터 기반 필터링
def create_self_query_retriever(loaded_db, llm):
    """
    메타데이터를 활용하여 더 정확한 필터링을 수행합니다.
    """
    from langchain.retrievers import SelfQueryRetriever
    
    metadata_field_info = [
        {"name": "title", "description": "문서의 제목", "type": "string"},
        {"name": "section", "description": "문서의 섹션", "type": "string"},
        {"name": "chunk_idx", "description": "청크 인덱스", "type": "integer"},
        {"name": "type", "description": "문서 타입", "type": "string"},
    ]
    
    return SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=loaded_db,
        document_contents="원스토어 인앱결제 관련 문서",
        metadata_field_info=metadata_field_info,
    )

# # 4. TimeWeightedVectorStoreRetriever - 시간 기반 가중치
# def create_time_weighted_retriever(loaded_db):
#     """
#     최신 문서에 더 높은 가중치를 부여합니다.
#     """
#     from langchain.retrievers import TimeWeightedVectorStoreRetriever
    
#     return TimeWeightedVectorStoreRetriever(
#         vectorstore=loaded_db,
#         decay_rate=0.01,  # 시간에 따른 감쇠율
#         k=20,
#         time_decay_function=lambda x: 1 / (1 + x * 0.01)  # 커스텀 감쇠 함수
#     )

# 5. EnsembleRetriever 개선 - 더 많은 검색 방식 조합
def create_enhanced_ensemble_retriever(loaded_db, docs_markdown, llm):
    """
    여러 검색 방식을 조합한 향상된 앙상블 검색기
    """
    from langchain.retrievers import EnsembleRetriever, BM25Retriever
    
    # 기존 검색기들
    bm25 = BM25Retriever.from_documents(
        docs_markdown,
        bm25_params={"k1": 1.5, "b": 0.75}
    )
    bm25.k = 20
    
    vector_retriever = loaded_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 30, "fetch_k": 70, "lambda_mult": 0.7}
    )
    
    # 새로운 검색기들
    multi_query_retriever = create_multi_query_retriever(loaded_db, llm)
    contextual_retriever = create_contextual_compression_retriever(loaded_db, llm)
    # time_weighted_retriever = create_time_weighted_retriever(loaded_db)
    
    # 앙상블 구성 (가중치 조정)
    enhanced_ensemble = EnsembleRetriever(
        retrievers=[
            bm25,                    # 키워드 기반 검색
            vector_retriever,        # 벡터 유사도 검색
            multi_query_retriever,   # 다중 쿼리 검색
            contextual_retriever,    # 컨텍스트 압축 검색
            # time_weighted_retriever  # 시간 가중치 검색
        ],
        weights=[0.3, 0.3, 0.2, 0.1, 0.1]  # 가중치 조정
    )
    
    return enhanced_ensemble

# 6. Reranker 기반 검색 - 검색 결과 재순위화
def create_reranker_retriever(loaded_db, llm):
    """
    검색 결과를 LLM을 사용하여 재순위화합니다.
    """
    def rerank_documents(query, documents):
        scores = []
        for doc in documents:
            prompt = f"""다음 질문과 문서의 관련성을 0-10 점수로 평가해주세요.
            
            질문: {query}
            문서: {doc.page_content}
            
            관련성 점수 (0-10):"""
            
            response = llm.invoke(prompt)
            try:
                score = float(response.content)
                scores.append(score)
            except:
                scores.append(0.0)
        
        # 점수에 따라 재정렬
        ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs]
    
    return rerank_documents

# 사용 예시
def get_recommended_retrievers(loaded_db, docs_markdown, llm):
    """
    추천 검색 방식들을 반환합니다.
    """
    return {
        "multi_query": create_multi_query_retriever(loaded_db, llm),
        "contextual_compression": create_contextual_compression_retriever(loaded_db, llm),
        "self_query": create_self_query_retriever(loaded_db, llm),
        # "time_weighted": create_time_weighted_retriever(loaded_db),
        "enhanced_ensemble": create_enhanced_ensemble_retriever(loaded_db, docs_markdown, llm),
        "reranker": create_reranker_retriever(loaded_db, llm)
    }

# 성능 비교 함수
def compare_retrievers(query, retrievers):
    """
    다양한 검색기의 성능을 비교합니다.
    """
    results = {}
    
    for name, retriever in retrievers.items():
        try:
            if name == "reranker":
                # Reranker는 특별한 처리 필요
                base_docs = retrievers["enhanced_ensemble"].invoke(query)
                results[name] = retriever(query, base_docs)
            else:
                results[name] = retriever.invoke(query)
        except Exception as e:
            print(f"Error with {name}: {e}")
            results[name] = []
    
    return results

if __name__ == "__main__":
    print("검색 품질 향상을 위한 추천 방식들:")
    print("1. MultiQueryRetriever - 다양한 쿼리 변형")
    print("2. ContextualCompressionRetriever - 컨텍스트 압축")
    print("3. SelfQueryRetriever - 메타데이터 필터링")
    print("4. TimeWeightedVectorStoreRetriever - 시간 가중치")
    print("5. Enhanced EnsembleRetriever - 다중 검색기 조합")
    print("6. Reranker - LLM 기반 재순위화") 