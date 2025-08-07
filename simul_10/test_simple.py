#!/usr/bin/env python3
"""
Simple test script for contextual retrieval implementation
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from langchain.docstore.document import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def test_contextual_prompt():
    """Test the contextual prompt with a simple example."""
    
    # Initialize the LLM
    llm = ChatOllama(
        model="exaone3.5:latest",
        temperature=0.1
    )
    
    # Contextual prompt template
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
    
    # Test with a simple example
    whole_document = """
    # 원스토어 인앱결제 API V7(SDK V21) 연동 안내
    
    ## 01. 원스토어 인앱결제 개요
    원스토어 인앱결제는 모바일 앱에서 상품을 구매할 수 있는 서비스입니다.
    
    ## 02. PNS(Payment Notification Service) 이용하기
    PNS는 Payment Notification Service의 약자입니다. PNS는 모바일의 네트워크 연결 불안정성을 보완하기 위해 개발사가 지정한 서버로 원스토어의 서버가 개별 사용자의 결제 상태(결제 완료, 결제 취소)를 메시지로 전송하여 결제 트랜젝션의 상태를 손실없이 알려주기 위한 용도의 기능입니다.
    
    ## 03. SDK 사용법
    SDK를 사용하여 인앱결제를 구현하는 방법을 설명합니다.
    """
    
    chunk_content = """
    PNS는 Payment Notification Service의 약자입니다. PNS는 모바일의 네트워크 연결 불안정성을 보완하기 위해 개발사가 지정한 서버로 원스토어의 서버가 개별 사용자의 결제 상태(결제 완료, 결제 취소)를 메시지로 전송하여 결제 트랜젝션의 상태를 손실없이 알려주기 위한 용도의 기능입니다.
    """
    
    try:
        print("Testing contextual prompt generation...")
        context = chain.invoke({
            "WHOLE_DOCUMENT": whole_document,
            "CHUNK_CONTENT": chunk_content
        })
        
        print(f"Generated context: {context}")
        print("✅ Contextual prompt test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error in contextual prompt test: {e}")
        return False

def test_document_enhancement():
    """Test document enhancement functionality."""
    
    from code_dev.contextual_retrieval_old import ContextualRetrieval
    
    # Create a simple test document
    test_doc = Document(
        page_content="PNS는 Payment Notification Service의 약자입니다.",
        metadata={"title": "PNS 개요", "chunk_idx": 1}
    )
    
    whole_document = """
    # 원스토어 인앱결제 API V7(SDK V21) 연동 안내
    
    ## 01. 원스토어 인앱결제 개요
    원스토어 인앱결제는 모바일 앱에서 상품을 구매할 수 있는 서비스입니다.
    
    ## 02. PNS(Payment Notification Service) 이용하기
    PNS는 Payment Notification Service의 약자입니다. PNS는 모바일의 네트워크 연결 불안정성을 보완하기 위해 개발사가 지정한 서버로 원스토어의 서버가 개별 사용자의 결제 상태(결제 완료, 결제 취소)를 메시지로 전송하여 결제 트랜젝션의 상태를 손실없이 알려주기 위한 용도의 기능입니다.
    """
    
    try:
        print("\nTesting document enhancement...")
        contextual_retrieval = ContextualRetrieval(model_name="exaone3.5:latest")
        
        enhanced_docs = contextual_retrieval.enhance_documents([test_doc], whole_document)
        
        if enhanced_docs:
            enhanced_doc = enhanced_docs[0]
            print(f"Original content: {test_doc.page_content}")
            print(f"Enhanced content: {enhanced_doc.page_content[:200]}...")
            print("✅ Document enhancement test successful!")
            return True
        else:
            print("❌ No enhanced documents returned")
            return False
            
    except Exception as e:
        print(f"❌ Error in document enhancement test: {e}")
        return False

if __name__ == "__main__":
    print("Starting simple tests for contextual retrieval...")
    
    # Test 1: Contextual prompt
    test1_passed = test_contextual_prompt()
    
    # Test 2: Document enhancement
    test2_passed = test_document_enhancement()
    
    print(f"\nTest Results:")
    print(f"Contextual prompt test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Document enhancement test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The contextual retrieval implementation is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.") 