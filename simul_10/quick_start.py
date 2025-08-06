#!/usr/bin/env python3
"""
Quick Start Script for Contextual Retrieval

This script demonstrates the key functionality of the contextual retrieval implementation.
"""

import os
from pathlib import Path
from langchain.docstore.document import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def demonstrate_contextual_retrieval():
    """Demonstrate the contextual retrieval functionality with a simple example."""
    
    print("🚀 Contextual Retrieval Quick Start Demo")
    print("=" * 50)
    
    # 1. Initialize the LLM
    print("\n1. Initializing LLM (exaone3.5:latest)...")
    llm = ChatOllama(
        model="exaone3.5:latest",
        temperature=0.1
    )
    
    # 2. Set up the contextual prompt
    print("2. Setting up contextual prompt...")
    contextual_prompt = PromptTemplate.from_template(
        """<document> 
{{WHOLE_DOCUMENT}} 
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{{CHUNK_CONTENT}} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
    )
    
    chain = contextual_prompt | llm | StrOutputParser()
    
    # 3. Test with sample data
    print("3. Testing with sample data...")
    
    # Sample whole document (simplified version of the actual document)
    whole_document = """
    # 원스토어 인앱결제 API V7(SDK V21) 연동 안내
    
    ## 01. 원스토어 인앱결제 개요
    원스토어 인앱결제는 모바일 앱에서 상품을 구매할 수 있는 서비스입니다.
    
    ## 02. PNS(Payment Notification Service) 이용하기
    PNS는 Payment Notification Service의 약자입니다. PNS는 모바일의 네트워크 연결 불안정성을 보완하기 위해 개발사가 지정한 서버로 원스토어의 서버가 개별 사용자의 결제 상태(결제 완료, 결제 취소)를 메시지로 전송하여 결제 트랜젝션의 상태를 손실없이 알려주기 위한 용도의 기능입니다.
    
    ## 03. SDK 사용법
    SDK를 사용하여 인앱결제를 구현하는 방법을 설명합니다.
    
    ## 04. 결제 테스트
    결제 테스트 및 보안 관련 정보를 제공합니다.
    """
    
    # Sample chunk content
    chunk_content = """
    PNS는 Payment Notification Service의 약자입니다. PNS는 모바일의 네트워크 연결 불안정성을 보완하기 위해 개발사가 지정한 서버로 원스토어의 서버가 개별 사용자의 결제 상태(결제 완료, 결제 취소)를 메시지로 전송하여 결제 트랜젝션의 상태를 손실없이 알려주기 위한 용도의 기능입니다.
    """
    
    print(f"Whole document length: {len(whole_document)} characters")
    print(f"Chunk content length: {len(chunk_content)} characters")
    
    # 4. Generate contextual information
    print("\n4. Generating contextual information...")
    try:
        context = chain.invoke({
            "WHOLE_DOCUMENT": whole_document,
            "CHUNK_CONTENT": chunk_content
        })
        
        print(f"✅ Generated context: {context}")
        
        # 5. Create enhanced document
        print("\n5. Creating enhanced document...")
        enhanced_content = f"[Context]: {context}\n\n{chunk_content}"
        
        enhanced_doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": chunk_content,
                "contextual_info": context,
                "source": "demo"
            }
        )
        
        print("✅ Enhanced document created successfully!")
        print(f"Original content length: {len(chunk_content)}")
        print(f"Enhanced content length: {len(enhanced_doc.page_content)}")
        
        print("\n📄 Enhanced Document Preview:")
        print("-" * 40)
        print(enhanced_doc.page_content)
        print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_usage_examples():
    """Show usage examples for the contextual retrieval implementation."""
    
    print("\n📚 Usage Examples:")
    print("=" * 30)
    
    print("\n1. Basic Usage:")
    print("""
    from contextual_retrieval import ContextualRetrieval
    
    # Initialize
    cr = ContextualRetrieval(model_name="exaone3.5:latest")
    
    # Enhance documents
    enhanced_docs = cr.enhance_documents(documents, whole_document)
    """)
    
    print("\n2. Running the full test:")
    print("""
    python contextual_retrieval.py
    """)
    
    print("\n3. Using the Jupyter notebook:")
    print("""
    jupyter notebook contextual_retrieval_test.ipynb
    """)
    
    print("\n4. Running simple tests:")
    print("""
    python test_simple.py
    """)

if __name__ == "__main__":
    print("🎯 Contextual Retrieval Implementation Demo")
    print("This demo shows how the contextual retrieval feature works.")
    
    # Run the demonstration
    success = demonstrate_contextual_retrieval()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        show_usage_examples()
    else:
        print("\n⚠️ Demo encountered an error. Please check your setup.")
    
    print("\n📖 For more information, see README.md") 