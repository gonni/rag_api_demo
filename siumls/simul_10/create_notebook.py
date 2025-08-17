#!/usr/bin/env python3
"""
Script to create the contextual retrieval test notebook
"""

import json

def create_notebook():
    """Create the complete Jupyter notebook content."""
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Contextual Retrieval Test\n",
                    "\n",
                    "This notebook implements and tests contextual retrieval functionality that enhances document chunks with contextual information using the specified prompt."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import re\n",
                    "from typing import List\n",
                    "from pathlib import Path\n",
                    "from langchain.docstore.document import Document\n",
                    "from langchain_community.document_loaders import TextLoader\n",
                    "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
                    "from langchain_ollama import OllamaEmbeddings, ChatOllama\n",
                    "from langchain_community.vectorstores import FAISS\n",
                    "from langchain_text_splitters import MarkdownHeaderTextSplitter\n",
                    "from langchain_core.prompts import PromptTemplate\n",
                    "from langchain_core.output_parsers import StrOutputParser\n",
                    "from langchain_core.runnables import RunnablePassthrough"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Document Loading and Splitting"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def hierarchical_markdown_split(md_text: str, path_prefix: str = \"\") -> list[Document]:\n",
                    "    \"\"\"마크다운 문서를 계층적으로 분할합니다.\"\"\"\n",
                    "    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[\n",
                    "        (\"#\", \"title\"),\n",
                    "        (\"##\", \"section\"),\n",
                    "        (\"###\", \"subsection\"),\n",
                    "        (\"####\", \"subsubsection\"),\n",
                    "        (\"#####\", \"subsubsubsection\")\n",
                    "    ])\n",
                    "    docs = splitter.split_text(md_text)\n",
                    "\n",
                    "    result_docs = []\n",
                    "    current_title = None\n",
                    "    chunk_idx = 0\n",
                    "    for doc in docs:\n",
                    "        metadata = doc.metadata\n",
                    "        if \"title\" in metadata:\n",
                    "            current_title = metadata[\"title\"]\n",
                    "\n",
                    "        if current_title:\n",
                    "            chunk_idx += 1\n",
                    "            full_title = \"\" + current_title\n",
                    "            if \"section\" in metadata:\n",
                    "                full_title += f\" / {metadata['section']}\"\n",
                    "            if \"subsection\" in metadata:\n",
                    "                full_title += f\" / {metadata['subsection']}\"\n",
                    "            if \"subsubsection\" in metadata:\n",
                    "                full_title += f\" / {metadata['subsubsection']}\"\n",
                    "            if \"subsubsubsection\" in metadata:\n",
                    "                full_title += f\" / {metadata['subsubsubsection']}\"\n",
                    "\n",
                    "            content = f\"[section_path]: {full_title}\\n\\n{doc.page_content}\"\n",
                    "            doc = Document(page_content=content, metadata={\n",
                    "                **doc.metadata,\n",
                    "                \"type\": \"documentation\",\n",
                    "                \"source\": \"dev_center_guide_allmd.md\",\n",
                    "                \"chunk_idx\": chunk_idx\n",
                    "            })\n",
                    "\n",
                    "        result_docs.append(doc)\n",
                    "\n",
                    "    return result_docs\n",
                    "\n",
                    "def load_markdown_file(file_path: str) -> str:\n",
                    "    \"\"\"마크다운 파일을 로드합니다.\"\"\"\n",
                    "    with open(file_path, 'r', encoding='utf-8') as file:\n",
                    "        return file.read()\n",
                    "\n",
                    "# 마크다운 파일 로드 및 분할\n",
                    "str_md_file = load_markdown_file(\"../data/dev_center_guide_allmd_touched.md\") \n",
                    "docs_markdown = hierarchical_markdown_split(str_md_file)\n",
                    "\n",
                    "print(f\"마크다운 문서 분할 완료: {len(docs_markdown)}개 청크\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Contextual Retrieval Implementation"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "class ContextualRetrieval:\n",
                    "    def __init__(self, model_name: str = \"exaone3.5:latest\"):\n",
                    "        self.model_name = model_name\n",
                    "        self.llm = ChatOllama(\n",
                    "            model=model_name,\n",
                    "            temperature=0.1\n",
                    "        )\n",
                    "        \n",
                    "        # Contextual prompt template\n",
                    "        self.contextual_prompt = PromptTemplate.from_template(\n",
                    "            \"\"\"<document> \n",
                    "{{WHOLE_DOCUMENT}} \n",
                    "</document> \n",
                    "Here is the chunk we want to situate within the whole document \n",
                    "<chunk> \n",
                    "{{CHUNK_CONTENT}} \n",
                    "</chunk> \n",
                    "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.\"\"\"\n",
                    "        )\n",
                    "        \n",
                    "        self.chain = self.contextual_prompt | self.llm | StrOutputParser()\n",
                    "    \n",
                    "    def generate_context(self, whole_document: str, chunk_content: str) -> str:\n",
                    "        \"\"\"Generate contextual information for a chunk within the whole document.\"\"\"\n",
                    "        try:\n",
                    "            context = self.chain.invoke({\n",
                    "                \"WHOLE_DOCUMENT\": whole_document,\n",
                    "                \"CHUNK_CONTENT\": chunk_content\n",
                    "            })\n",
                    "            return context.strip()\n",
                    "        except Exception as e:\n",
                    "            print(f\"Error generating context: {e}\")\n",
                    "            return \"\"\n",
                    "    \n",
                    "    def enhance_documents(self, documents: List[Document], whole_document: str) -> List[Document]:\n",
                    "        \"\"\"Enhance documents with contextual information.\"\"\"\n",
                    "        enhanced_docs = []\n",
                    "        \n",
                    "        for i, doc in enumerate(documents):\n",
                    "            print(f\"Processing document {i+1}/{len(documents)}\")\n",
                    "            \n",
                    "            # Generate contextual information\n",
                    "            context = self.generate_context(whole_document, doc.page_content)\n",
                    "            \n",
                    "            # Create enhanced content\n",
                    "            if context:\n",
                    "                enhanced_content = f\"[Context]: {context}\\n\\n{doc.page_content}\"\n",
                    "            else:\n",
                    "                enhanced_content = doc.page_content\n",
                    "            \n",
                    "            # Create new document with enhanced content\n",
                    "            enhanced_doc = Document(\n",
                    "                page_content=enhanced_content,\n",
                    "                metadata={\n",
                    "                    **doc.metadata,\n",
                    "                    \"original_content\": doc.page_content,\n",
                    "                    \"contextual_info\": context\n",
                    "                }\n",
                    "            )\n",
                    "            \n",
                    "            enhanced_docs.append(enhanced_doc)\n",
                    "        \n",
                    "        return enhanced_docs"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Test Contextual Retrieval on Sample Documents"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Initialize contextual retrieval\n",
                    "contextual_retrieval = ContextualRetrieval(model_name=\"exaone3.5:latest\")\n",
                    "\n",
                    "# Test with a small sample first\n",
                    "sample_docs = docs_markdown[:5]  # First 5 documents\n",
                    "print(f\"Testing with {len(sample_docs)} sample documents...\")\n",
                    "\n",
                    "# Enhance documents with contextual information\n",
                    "enhanced_docs = contextual_retrieval.enhance_documents(sample_docs, str_md_file)\n",
                    "\n",
                    "print(f\"\\nEnhanced {len(enhanced_docs)} documents with contextual information\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Compare Original vs Enhanced Documents"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Compare original and enhanced documents\n",
                    "for i, (original, enhanced) in enumerate(zip(sample_docs, enhanced_docs)):\n",
                    "    print(f\"\\n=== Document {i+1} ===\")\n",
                    "    print(f\"Original length: {len(original.page_content)}\")\n",
                    "    print(f\"Enhanced length: {len(enhanced.page_content)}\")\n",
                    "    \n",
                    "    # Show the contextual information if available\n",
                    "    if \"contextual_info\" in enhanced.metadata and enhanced.metadata[\"contextual_info\"]:\n",
                    "        print(f\"\\nContextual Info: {enhanced.metadata['contextual_info']}\")\n",
                    "    \n",
                    "    print(f\"\\nEnhanced Content Preview:\")\n",
                    "    print(enhanced.page_content[:500] + \"...\" if len(enhanced.page_content) > 500 else enhanced.page_content)\n",
                    "    print(\"-\" * 80)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Create Vector Database with Enhanced Documents"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def embed_and_save(docs: List[Document], output_path: str, model_name: str = \"bge-m3:latest\"):\n",
                    "    \"\"\"문서를 임베딩하고 FAISS 데이터베이스로 저장합니다.\"\"\"\n",
                    "    # 임베딩 모델 초기화\n",
                    "    embedding_model = OllamaEmbeddings(model=model_name)\n",
                    "    \n",
                    "    # FAISS 데이터베이스 생성 및 저장\n",
                    "    db = FAISS.from_documents(docs, embedding_model)\n",
                    "    db.save_local(output_path)\n",
                    "    print(f\"✅ 임베딩 저장 완료: {output_path}\")\n",
                    "\n",
                    "# Create enhanced documents for all documents (or a subset for testing)\n",
                    "print(\"Creating enhanced documents for all documents...\")\n",
                    "all_enhanced_docs = contextual_retrieval.enhance_documents(docs_markdown, str_md_file)\n",
                    "\n",
                    "# Save enhanced documents\n",
                    "output_dir = \"../models/faiss_contextual_enhanced_\" + contextual_retrieval.model_name[:3]\n",
                    "os.makedirs(output_dir, exist_ok=True)\n",
                    "embed_and_save(all_enhanced_docs, output_dir, \"bge-m3:latest\")\n",
                    "\n",
                    "print(f\"Total enhanced documents: {len(all_enhanced_docs)}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Test Retrieval with Enhanced Documents"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load the enhanced vector database\n",
                    "embedding_model = OllamaEmbeddings(model=\"bge-m3:latest\")\n",
                    "enhanced_db = FAISS.load_local(\n",
                    "    folder_path=output_dir,\n",
                    "    embeddings=embedding_model,\n",
                    "    allow_dangerous_deserialization=True,\n",
                    ")\n",
                    "\n",
                    "enhanced_retriever = enhanced_db.as_retriever(\n",
                    "    search_type=\"mmr\",\n",
                    "    search_kwargs={\"k\": 10, \"fetch_k\": 25, \"lambda_mult\": 0.7}\n",
                    ")\n",
                    "\n",
                    "# Test queries\n",
                    "test_queries = [\n",
                    "    \"원스토어 인앱결제의 PNS의 개념을 설명해주세요\",\n",
                    "    \"PNS 메시지 규격의 purchaseState는 어떤 값으로 구성되나요?\",\n",
                    "    \"원스토어 인앱결제 SDK 사용법\",\n",
                    "    \"결제 테스트 및 보안 관련 정보\"\n",
                    "]\n",
                    "\n",
                    "for query in test_queries:\n",
                    "    print(f\"\\n=== Query: {query} ===\")\n",
                    "    results = enhanced_retriever.invoke(query)\n",
                    "    print(f\"Retrieved {len(results)} documents\")\n",
                    "    \n",
                    "    for i, doc in enumerate(results[:3]):  # Show first 3 results\n",
                    "        print(f\"\\n--- Result {i+1} ---\")\n",
                    "        print(doc.page_content[:300] + \"...\" if len(doc.page_content) > 300 else doc.page_content)\n",
                    "    print(\"=\" * 80)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Comparison with Original Retrieval"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load original vector database for comparison\n",
                    "original_db = FAISS.load_local(\n",
                    "    folder_path=\"../models/faiss_vs_rag_iap_v10_1_bge\",\n",
                    "    embeddings=embedding_model,\n",
                    "    allow_dangerous_deserialization=True,\n",
                    ")\n",
                    "\n",
                    "original_retriever = original_db.as_retriever(\n",
                    "    search_type=\"mmr\",\n",
                    "    search_kwargs={\"k\": 10, \"fetch_k\": 25, \"lambda_mult\": 0.7}\n",
                    ")\n",
                    "\n",
                    "# Compare retrieval results\n",
                    "query = \"원스토어 인앱결제의 PNS의 개념을 설명해주세요\"\n",
                    "\n",
                    "print(\"=== Original Retrieval ===\")\n",
                    "original_results = original_retriever.invoke(query)\n",
                    "for i, doc in enumerate(original_results[:3]):\n",
                    "    print(f\"\\nOriginal Result {i+1}:\")\n",
                    "    print(doc.page_content[:200] + \"...\")\n",
                    "\n",
                    "print(\"\\n=== Enhanced Retrieval ===\")\n",
                    "enhanced_results = enhanced_retriever.invoke(query)\n",
                    "for i, doc in enumerate(enhanced_results[:3]):\n",
                    "    print(f\"\\nEnhanced Result {i+1}:\")\n",
                    "    print(doc.page_content[:200] + \"...\")\n",
                    "\n",
                    "print(\"\\n=== Comparison Summary ===\")\n",
                    "print(f\"Original results count: {len(original_results)}\")\n",
                    "print(f\"Enhanced results count: {len(enhanced_results)}\")\n",
                    "\n",
                    "# Check if contextual information is present in enhanced results\n",
                    "contextual_count = 0\n",
                    "for doc in enhanced_results:\n",
                    "    if \"[Context]:\" in doc.page_content:\n",
                    "        contextual_count += 1\n",
                    "\n",
                    "print(f\"Enhanced documents with contextual info: {contextual_count}/{len(enhanced_results)}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Performance Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Analyze the performance of contextual retrieval\n",
                    "import time\n",
                    "\n",
                    "# Test retrieval speed\n",
                    "test_query = \"원스토어 인앱결제의 PNS의 개념을 설명해주세요\"\n",
                    "\n",
                    "print(\"Testing retrieval performance...\")\n",
                    "\n",
                    "# Test original retrieval\n",
                    "start_time = time.time()\n",
                    "original_results = original_retriever.invoke(test_query)\n",
                    "original_time = time.time() - start_time\n",
                    "\n",
                    "# Test enhanced retrieval\n",
                    "start_time = time.time()\n",
                    "enhanced_results = enhanced_retriever.invoke(test_query)\n",
                    "enhanced_time = time.time() - start_time\n",
                    "\n",
                    "print(f\"\\nPerformance Results:\")\n",
                    "print(f\"Original retrieval time: {original_time:.4f} seconds\")\n",
                    "print(f\"Enhanced retrieval time: {enhanced_time:.4f} seconds\")\n",
                    "print(f\"Time difference: {enhanced_time - original_time:.4f} seconds\")\n",
                    "\n",
                    "# Analyze result quality\n",
                    "print(f\"\\nQuality Analysis:\")\n",
                    "print(f\"Original results: {len(original_results)} documents\")\n",
                    "print(f\"Enhanced results: {len(enhanced_results)} documents\")\n",
                    "\n",
                    "# Check for contextual information in enhanced results\n",
                    "contextual_docs = [doc for doc in enhanced_results if \"[Context]:\" in doc.page_content]\n",
                    "print(f\"Enhanced documents with context: {len(contextual_docs)}/{len(enhanced_results)}\")\n",
                    "\n",
                    "if contextual_docs:\n",
                    "    print(f\"\\nSample contextual information:\")\n",
                    "    sample_context = contextual_docs[0].page_content.split(\"[Context]:\")[1].split(\"\\n\\n\")[0]\n",
                    "    print(sample_context.strip())"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

if __name__ == "__main__":
    # Create the notebook
    notebook = create_notebook()
    
    # Write to file
    with open("contextual_retrieval_test.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("✅ Notebook created successfully: contextual_retrieval_test.ipynb") 