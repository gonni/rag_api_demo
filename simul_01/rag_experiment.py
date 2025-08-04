import os
import re
from typing import List, Dict, Any
from pathlib import Path
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.schema import BaseRetriever
from langchain_community.retrievers import FAISSRetriever
import json
from datetime import datetime

class DocumentSplitterExperiment:
    def __init__(self, markdown_file_path: str):
        self.markdown_file_path = markdown_file_path
        self.raw_text = self.load_markdown_file(markdown_file_path)
        self.results = {}
        
    def load_markdown_file(self, file_path: str) -> str:
        """마크다운 파일을 로드합니다."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def extract_headers(self, text: str) -> List[Dict[str, Any]]:
        """마크다운 텍스트에서 헤더 정보를 추출합니다."""
        headers = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # 헤더 패턴 매칭 (#, ##, ###, ####, #####, ######)
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                headers.append({
                    'level': level,
                    'title': title,
                    'line_number': i,
                    'content_start': i + 1
                })
        
        return headers
    
    def get_content_between_headers(self, text: str, start_header: Dict, end_header: Dict[str, Any] | None = None) -> str:
        """두 헤더 사이의 내용을 추출합니다."""
        lines = text.split('\n')
        start_line = start_header['content_start']
        
        if end_header:
            end_line = end_header['line_number']
        else:
            end_line = len(lines)
        
        return '\n'.join(lines[start_line:end_line])
    
    def strategy_1_hierarchical_with_context(self) -> List[Document]:
        """전략 1: 계층적 분할 + 전체 맥락 포함"""
        headers = self.extract_headers(self.raw_text)
        docs = []
        
        # 대제목(##) 기준으로 메인 섹션 생성
        main_sections = []
        for i, header in enumerate(headers):
            if header['level'] == 2:  # ## 레벨
                section_content = self.get_content_between_headers(
                    self.raw_text, 
                    header, 
                    headers[i + 1] if i + 1 < len(headers) else None
                )
                main_sections.append({
                    'title': header['title'],
                    'content': section_content,
                    'start_line': header['line_number']
                })
        
        # 각 메인 섹션에 대해 세부 분할
        for section in main_sections:
            # 메인 섹션 전체를 하나의 문서로 생성 (전체 맥락)
            main_doc = Document(
                page_content=f"[MAIN_SECTION]: {section['title']}\n\n{section['content']}",
                metadata={
                    'type': 'main_section',
                    'title': section['title'],
                    'source': self.markdown_file_path,
                    'section_level': 2
                }
            )
            docs.append(main_doc)
            
            # 해당 섹션 내의 소제목들로 세부 분할
            section_headers = self.extract_headers(section['content'])
            for sub_header in section_headers:
                if sub_header['level'] >= 3:  # ###, ####, ##### 레벨
                    sub_content = self.get_content_between_headers(
                        section['content'],
                        sub_header,
                        section_headers[section_headers.index(sub_header) + 1] if section_headers.index(sub_header) + 1 < len(section_headers) else None
                    )
                    
                    # 제목 계층 구조 생성
                    title_hierarchy = f"{section['title']} / {sub_header['title']}"
                    
                    sub_doc = Document(
                        page_content=f"[SUBSECTION]: {title_hierarchy}\n\n{sub_content}",
                        metadata={
                            'type': 'subsection',
                            'title': sub_header['title'],
                            'parent_title': section['title'],
                            'title_hierarchy': title_hierarchy,
                            'source': self.markdown_file_path,
                            'section_level': sub_header['level']
                        }
                    )
                    docs.append(sub_doc)
        
        return docs
    
    def strategy_2_enhanced_hierarchical(self) -> List[Document]:
        """전략 2: 향상된 계층적 분할 (제목 정보를 더 명확하게 포함)"""
        headers = self.extract_headers(self.raw_text)
        docs = []
        
        for i, header in enumerate(headers):
            content = self.get_content_between_headers(
                self.raw_text,
                header,
                headers[i + 1] if i + 1 < len(headers) else None
            )
            
            # 제목 계층 구조 생성
            title_hierarchy = self.build_title_hierarchy(headers, i)
            
            # 제목 정보를 다양한 방식으로 포함
            enhanced_content = f"""
[HEADER_LEVEL]: {header['level']}
[FULL_TITLE]: {title_hierarchy}
[SHORT_TITLE]: {header['title']}
[CONTENT]:
{content}
"""
            
            doc = Document(
                page_content=enhanced_content,
                metadata={
                    'type': 'hierarchical_section',
                    'title': header['title'],
                    'title_hierarchy': title_hierarchy,
                    'header_level': header['level'],
                    'source': self.markdown_file_path,
                    'line_number': header['line_number']
                }
            )
            docs.append(doc)
        
        return docs
    
    def build_title_hierarchy(self, headers: List[Dict], current_index: int) -> str:
        """현재 헤더까지의 제목 계층 구조를 생성합니다."""
        current_header = headers[current_index]
        hierarchy = [current_header['title']]
        
        # 상위 헤더들을 찾아서 계층 구조 생성
        for i in range(current_index - 1, -1, -1):
            if headers[i]['level'] < current_header['level']:
                hierarchy.insert(0, headers[i]['title'])
                current_header = headers[i]
        
        return ' / '.join(hierarchy)
    
    def strategy_3_table_aware_split(self) -> List[Document]:
        """전략 3: 테이블 인식 분할 (테이블을 하나의 단위로 유지)"""
        headers = self.extract_headers(self.raw_text)
        docs = []
        
        for i, header in enumerate(headers):
            content = self.get_content_between_headers(
                self.raw_text,
                header,
                headers[i + 1] if i + 1 < len(headers) else None
            )
            
            # 테이블 패턴 찾기
            table_pattern = r'\|.*\|.*\n\|[\s\-:|]+\|\n(\|.*\|\n)*'
            tables = re.findall(table_pattern, content)
            
            if tables:
                # 테이블이 있는 경우, 테이블을 포함한 전체 섹션을 하나의 문서로
                title_hierarchy = self.build_title_hierarchy(headers, i)
                enhanced_content = f"""
[TABLE_SECTION]: {title_hierarchy}
[CONTENT]:
{content}
"""
                doc = Document(
                    page_content=enhanced_content,
                    metadata={
                        'type': 'table_section',
                        'title': header['title'],
                        'title_hierarchy': title_hierarchy,
                        'has_tables': True,
                        'table_count': len(tables),
                        'source': self.markdown_file_path,
                        'header_level': header['level']
                    }
                )
                docs.append(doc)
            else:
                # 테이블이 없는 경우 일반 분할
                title_hierarchy = self.build_title_hierarchy(headers, i)
                enhanced_content = f"""
[REGULAR_SECTION]: {title_hierarchy}
[CONTENT]:
{content}
"""
                doc = Document(
                    page_content=enhanced_content,
                    metadata={
                        'type': 'regular_section',
                        'title': header['title'],
                        'title_hierarchy': title_hierarchy,
                        'has_tables': False,
                        'source': self.markdown_file_path,
                        'header_level': header['level']
                    }
                )
                docs.append(doc)
        
        return docs
    
    def strategy_4_keyword_enhanced_split(self) -> List[Document]:
        """전략 4: 키워드 강화 분할 (중요 키워드를 제목에 반영)"""
        headers = self.extract_headers(self.raw_text)
        docs = []
        
        # 중요 키워드 패턴
        important_keywords = [
            'PNS', 'purchaseState', 'COMPLETED', 'CANCELED', '결제', '취소',
            'paymentMethod', 'DCB', 'PHONEBILL', 'ONEPAY', 'CREDITCARD',
            'Signature', '검증', 'API', 'SDK'
        ]
        
        for i, header in enumerate(headers):
            content = self.get_content_between_headers(
                self.raw_text,
                header,
                headers[i + 1] if i + 1 < len(headers) else None
            )
            
            # 내용에서 중요 키워드 찾기
            found_keywords = []
            for keyword in important_keywords:
                if keyword.lower() in content.lower():
                    found_keywords.append(keyword)
            
            title_hierarchy = self.build_title_hierarchy(headers, i)
            
            # 키워드 정보를 포함한 강화된 내용
            keyword_info = f"[KEYWORDS]: {', '.join(found_keywords)}" if found_keywords else "[KEYWORDS]: None"
            enhanced_content = f"""
{keyword_info}
[FULL_TITLE]: {title_hierarchy}
[CONTENT]:
{content}
"""
            
            doc = Document(
                page_content=enhanced_content,
                metadata={
                    'type': 'keyword_enhanced_section',
                    'title': header['title'],
                    'title_hierarchy': title_hierarchy,
                    'keywords': found_keywords,
                    'keyword_count': len(found_keywords),
                    'source': self.markdown_file_path,
                    'header_level': header['level']
                }
            )
            docs.append(doc)
        
        return docs
    
    def create_vectorstore(self, docs: List[Document], strategy_name: str) -> FAISS:
        """벡터 스토어를 생성합니다."""
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    
    def test_query(self, vectorstore: FAISS, query: str, strategy_name: str) -> Dict[str, Any]:
        """쿼리를 테스트하고 결과를 반환합니다."""
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(query)
        
        # 결과 분석
        relevant_docs = []
        for i, doc in enumerate(docs):
            relevance_score = 0
            if 'purchaseState' in doc.page_content:
                relevance_score += 10
            if 'COMPLETED' in doc.page_content or 'CANCELED' in doc.page_content:
                relevance_score += 5
            if 'PNS' in doc.page_content:
                relevance_score += 3
            
            relevant_docs.append({
                'rank': i + 1,
                'title': doc.metadata.get('title', 'Unknown'),
                'content_preview': doc.page_content[:200] + '...',
                'relevance_score': relevance_score,
                'metadata': doc.metadata
            })
        
        return {
            'strategy': strategy_name,
            'query': query,
            'total_docs': len(docs),
            'relevant_docs': relevant_docs,
            'avg_relevance_score': sum(d['relevance_score'] for d in relevant_docs) / len(relevant_docs) if relevant_docs else 0
        }
    
    def run_experiment(self, test_queries: List[str] | None = None):
        """전체 실험을 실행합니다."""
        if test_queries is None:
            test_queries = [
                "PNS의 purchaseState에는 어떤 값들이 있나요?",
                "purchaseState COMPLETED CANCELED",
                "원스토어 결제 상태 값",
                "PNS payment notification service"
            ]
        
        strategies = {
            'strategy_1': self.strategy_1_hierarchical_with_context,
            'strategy_2': self.strategy_2_enhanced_hierarchical,
            'strategy_3': self.strategy_3_table_aware_split,
            'strategy_4': self.strategy_4_keyword_enhanced_split
        }
        
        results = {}
        
        for strategy_name, strategy_func in strategies.items():
            print(f"\n=== 실행 중: {strategy_name} ===")
            
            # 문서 분할
            docs = strategy_func()
            print(f"생성된 문서 수: {len(docs)}")
            
            # 벡터 스토어 생성
            vectorstore = self.create_vectorstore(docs, strategy_name)
            
            # 각 쿼리 테스트
            strategy_results = []
            for query in test_queries:
                result = self.test_query(vectorstore, query, strategy_name)
                strategy_results.append(result)
                print(f"쿼리: {query}")
                print(f"평균 관련성 점수: {result['avg_relevance_score']:.2f}")
            
            results[strategy_name] = {
                'doc_count': len(docs),
                'query_results': strategy_results
            }
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"simul_01/experiment_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n실험 결과가 {results_file}에 저장되었습니다.")
        
        # 결과 요약 출력
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """실험 결과 요약을 출력합니다."""
        print("\n" + "="*60)
        print("실험 결과 요약")
        print("="*60)
        
        for strategy_name, strategy_data in results.items():
            print(f"\n{strategy_name}:")
            print(f"  문서 수: {strategy_data['doc_count']}")
            
            avg_scores = []
            for query_result in strategy_data['query_results']:
                avg_scores.append(query_result['avg_relevance_score'])
                print(f"  쿼리 '{query_result['query']}': 평균 점수 {query_result['avg_relevance_score']:.2f}")
            
            if avg_scores:
                overall_avg = sum(avg_scores) / len(avg_scores)
                print(f"  전체 평균 점수: {overall_avg:.2f}")

def main():
    """메인 실행 함수"""
    # 실험 실행
    experiment = DocumentSplitterExperiment("data/dev_center_guide_allmd_touched.md")
    
    # 테스트 쿼리 정의
    test_queries = [
        "PNS의 purchaseState에는 어떤 값들이 있나요?",
        "purchaseState COMPLETED CANCELED 값",
        "원스토어 결제 상태 값들",
        "PNS payment notification service purchaseState",
        "COMPLETED CANCELED 결제 상태"
    ]
    
    # 실험 실행
    results = experiment.run_experiment(test_queries)
    
    print("\n실험 완료!")

if __name__ == "__main__":
    main() 