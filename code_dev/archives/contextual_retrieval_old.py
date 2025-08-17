#!/usr/bin/env python3
"""
Contextual Retrieval Implementation

This script implements contextual retrieval functionality that enhances document chunks
with contextual information using the specified prompt with exaone3.5 model.
"""

import os
import re
from typing import List
from pathlib import Path
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def hierarchical_markdown_split(md_text: str, path_prefix: str = "") -> list[Document]:
    """마크다운 문서를 계층적으로 분할합니다."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "title"),
        ("##", "section"),
        ("###", "subsection"),
        ("####", "subsubsection"),
        ("#####", "subsubsubsection")
    ])
    docs = splitter.split_text(md_text)

    result_docs = []
    current_title = None
    chunk_idx = 0
    for doc in docs:
        metadata = doc.metadata
        if "title" in metadata:
            current_title = metadata["title"]

        if current_title:
            chunk_idx += 1
            full_title = "" + current_title
            if "section" in metadata:
                full_title += f" / {metadata['section']}"
            if "subsection" in metadata:
                full_title += f" / {metadata['subsection']}"
            if "subsubsection" in metadata:
                full_title += f" / {metadata['subsubsection']}"
            if "subsubsubsection" in metadata:
                full_title += f" / {metadata['subsubsubsection']}"

            content = f"[section_path]: {full_title}\n\n{doc.page_content}"
            doc = Document(page_content=content, metadata={
                **doc.metadata,
                "type": "documentation",
                "source": "dev_center_guide_allmd.md",
                "chunk_idx": chunk_idx
            })

        result_docs.append(doc)

    return result_docs


def load_markdown_file(file_path: str) -> str:
    """마크다운 파일을 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


class ContextualRetrieval:
    def __init__(self, model_name: str = "exaone3.5:latest"):
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1
        )
        
        # Contextual prompt template
        self.contextual_prompt = PromptTemplate.from_template(
            """<document> 
{{WHOLE_DOCUMENT}} 
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{{CHUNK_CONTENT}} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
        )
        
        self.chain = self.contextual_prompt | self.llm | StrOutputParser()
    
    def generate_context(self, whole_document: str, chunk_content: str) -> str:
        """Generate contextual information for a chunk within the whole document."""
        try:
            context = self.chain.invoke({
                "WHOLE_DOCUMENT": whole_document,
                "CHUNK_CONTENT": chunk_content
            })
            return context.strip()
        except Exception as e:
            print(f"Error generating context: {e}")
            return ""
    
    def enhance_documents(self, documents: List[Document], whole_document: str) -> List[Document]:
        """Enhance documents with contextual information."""
        enhanced_docs = []
        
        for i, doc in enumerate(documents):
            print(f"Processing document {i+1}/{len(documents)}")
            
            # Generate contextual information
            context = self.generate_context(whole_document, doc.page_content)
            
            # Create enhanced content
            if context:
                enhanced_content = f"[Context]: {context}\n\n{doc.page_content}"
            else:
                enhanced_content = doc.page_content
            
            # Create new document with enhanced content
            enhanced_doc = Document(
                page_content=enhanced_content,
                metadata={
                    **doc.metadata,
                    "original_content": doc.page_content,
                    "contextual_info": context
                }
            )
            
            enhanced_docs.append(enhanced_doc)
        
        return enhanced_docs


def embed_and_save(docs: List[Document], output_path: str, model_name: str = "bge-m3:latest"):
    """문서를 임베딩하고 FAISS 데이터베이스로 저장합니다."""
    # 임베딩 모델 초기화
    embedding_model = OllamaEmbeddings(model=model_name)
    
    # FAISS 데이터베이스 생성 및 저장
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(output_path)
    print(f"✅ 임베딩 저장 완료: {output_path}")


def test_contextual_retrieval():
    """Test the contextual retrieval functionality."""
    
    # Load documents
    print("Loading markdown documents...")
    str_md_file = load_markdown_file("../data/dev_center_guide_allmd_touched.md") 
    docs_markdown = hierarchical_markdown_split(str_md_file)
    print(f"Loaded {len(docs_markdown)} documents")
    
    # Initialize contextual retrieval
    print("Initializing contextual retrieval...")
    contextual_retrieval = ContextualRetrieval(model_name="exaone3.5:latest")
    
    # Test with a small sample first
    sample_docs = docs_markdown[:10]  # First 10 documents
    print(f"Testing with {len(sample_docs)} sample documents...")
    
    # Enhance documents with contextual information
    enhanced_docs = contextual_retrieval.enhance_documents(sample_docs, str_md_file)
    
    print(f"Enhanced {len(enhanced_docs)} documents with contextual information")
    
    # Show some examples
    for i, (original, enhanced) in enumerate(zip(sample_docs[:3], enhanced_docs[:3])):
        print(f"\n=== Document {i+1} ===")
        print(f"Original length: {len(original.page_content)}")
        print(f"Enhanced length: {len(enhanced.page_content)}")
        
        if "contextual_info" in enhanced.metadata and enhanced.metadata["contextual_info"]:
            print(f"Contextual Info: {enhanced.metadata['contextual_info']}")
        
        print(f"Enhanced Content Preview:")
        print(enhanced.page_content[:300] + "..." if len(enhanced.page_content) > 300 else enhanced.page_content)
        print("-" * 80)
    
    # Create vector database with enhanced documents
    print("\nCreating vector database with enhanced documents...")
    output_dir = "../models/faiss_contextual_enhanced_exa"
    os.makedirs(output_dir, exist_ok=True)
    embed_and_save(enhanced_docs, output_dir, "bge-m3:latest")
    
    # Test retrieval
    print("\nTesting retrieval with enhanced documents...")
    embedding_model = OllamaEmbeddings(model="bge-m3:latest")
    enhanced_db = FAISS.load_local(
        folder_path=output_dir,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    
    enhanced_retriever = enhanced_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
    )
    
    # Test query
    query = "원스토어 인앱결제의 PNS의 개념을 설명해주세요"
    print(f"\nQuery: {query}")
    
    results = enhanced_retriever.invoke(query)
    print(f"Retrieved {len(results)} documents")
    
    for i, doc in enumerate(results[:3]):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)
    
    return enhanced_docs, enhanced_retriever


if __name__ == "__main__":
    print("Starting Contextual Retrieval Test...")
    enhanced_docs, retriever = test_contextual_retrieval()
    print("\nContextual retrieval test completed!") 