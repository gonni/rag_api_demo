import re
import json
from typing import List, Dict, Any
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DocumentAnalyzer:
    def __init__(self, markdown_file_path: str):
        self.markdown_file_path = markdown_file_path
        self.raw_text = self.load_markdown_file(markdown_file_path)
        
    def load_markdown_file(self, file_path: str) -> str:
        """마크다운 파일을 로드합니다."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def analyze_document_structure(self) -> Dict[str, Any]:
        """문서 구조를 분석합니다."""
        lines = self.raw_text.split('\n')
        
        # 헤더 분석
        headers = []
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        
        for i, line in enumerate(lines):
            match = header_pattern.match(line.strip())
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                headers.append({
                    'level': level,
                    'title': title,
                    'line_number': i
                })
        
        # 테이블 분석
        tables = []
        table_pattern = re.compile(r'\|.*\|.*\n\|[\s\-:|]+\|\n(\|.*\|\n)*')
        table_matches = table_pattern.finditer(self.raw_text)
        
        for match in table_matches:
            table_content = match.group(0)
            tables.append({
                'content': table_content,
                'start_pos': match.start(),
                'end_pos': match.end()
            })
        
        # 코드 블록 분석
        code_blocks = []
        code_pattern = re.compile(r'```[\w]*\n(.*?)\n```', re.DOTALL)
        code_matches = code_pattern.finditer(self.raw_text)
        
        for match in code_matches:
            code_content = match.group(1)
            code_blocks.append({
                'content': code_content,
                'start_pos': match.start(),
                'end_pos': match.end()
            })
        
        # 키워드 분석
        important_keywords = [
            'PNS', 'purchaseState', 'COMPLETED', 'CANCELED', '결제', '취소',
            'paymentMethod', 'DCB', 'PHONEBILL', 'ONEPAY', 'CREDITCARD',
            'Signature', '검증', 'API', 'SDK', '원스토어', '인앱결제'
        ]
        
        keyword_counts = {}
        for keyword in important_keywords:
            count = len(re.findall(keyword, self.raw_text, re.IGNORECASE))
            if count > 0:
                keyword_counts[keyword] = count
        
        return {
            'total_lines': len(lines),
            'total_characters': len(self.raw_text),
            'headers': headers,
            'header_count': len(headers),
            'header_levels': Counter([h['level'] for h in headers]),
            'tables': tables,
            'table_count': len(tables),
            'code_blocks': code_blocks,
            'code_block_count': len(code_blocks),
            'keyword_counts': keyword_counts,
            'keyword_total': sum(keyword_counts.values())
        }
    
    def find_purchase_state_info(self) -> List[Dict[str, Any]]:
        """purchaseState 관련 정보를 찾습니다."""
        lines = self.raw_text.split('\n')
        purchase_state_info = []
        
        for i, line in enumerate(lines):
            if 'purchaseState' in line.lower() or 'purchase_state' in line.lower():
                # 주변 컨텍스트 포함
                context_start = max(0, i - 5)
                context_end = min(len(lines), i + 6)
                context = lines[context_start:context_end]
                
                purchase_state_info.append({
                    'line_number': i,
                    'line_content': line,
                    'context': context,
                    'context_lines': list(range(context_start, context_end))
                })
        
        return purchase_state_info
    
    def analyze_table_content(self) -> List[Dict[str, Any]]:
        """테이블 내용을 분석합니다."""
        table_pattern = re.compile(r'\|.*\|.*\n\|[\s\-:|]+\|\n(\|.*\|\n)*')
        table_matches = table_pattern.finditer(self.raw_text)
        
        table_analyses = []
        for match in table_matches:
            table_content = match.group(0)
            lines = table_content.strip().split('\n')
            
            # 헤더 행 추출
            if len(lines) >= 2:
                header_row = lines[0]
                separator_row = lines[1]
                
                # 헤더 컬럼 추출
                headers = [col.strip() for col in header_row.split('|')[1:-1]]
                
                # 데이터 행들
                data_rows = []
                for line in lines[2:]:
                    if line.strip() and '|' in line:
                        data_cells = [cell.strip() for cell in line.split('|')[1:-1]]
                        data_rows.append(data_cells)
                
                table_analyses.append({
                    'headers': headers,
                    'data_rows': data_rows,
                    'row_count': len(data_rows),
                    'column_count': len(headers),
                    'content': table_content,
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
        
        return table_analyses
    
    def generate_visualization(self, analysis_results: Dict[str, Any], output_dir: str = "simul_01"):
        """분석 결과를 시각화합니다."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. 헤더 레벨 분포
        plt.figure(figsize=(10, 6))
        header_levels = analysis_results['header_levels']
        levels = list(header_levels.keys())
        counts = list(header_levels.values())
        
        plt.bar(levels, counts, color='skyblue')
        plt.title('헤더 레벨 분포')
        plt.xlabel('헤더 레벨')
        plt.ylabel('개수')
        plt.xticks(levels)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/header_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 키워드 빈도
        if analysis_results['keyword_counts']:
            plt.figure(figsize=(12, 8))
            keywords = list(analysis_results['keyword_counts'].keys())
            counts = list(analysis_results['keyword_counts'].values())
            
            # 빈도순 정렬
            sorted_data = sorted(zip(keywords, counts), key=lambda x: x[1], reverse=True)
            keywords, counts = zip(*sorted_data)
            
            plt.barh(keywords, counts, color='lightcoral')
            plt.title('중요 키워드 빈도')
            plt.xlabel('빈도')
            plt.ylabel('키워드')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/keyword_frequency.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 문서 구조 요약
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 총 라인 수
        ax1.pie([analysis_results['total_lines']], labels=['총 라인 수'], autopct='%1.0f', colors=['lightblue'])
        ax1.set_title('문서 크기')
        
        # 헤더 수
        ax2.pie([analysis_results['header_count']], labels=['헤더 수'], autopct='%1.0f', colors=['lightgreen'])
        ax2.set_title('헤더 수')
        
        # 테이블 수
        ax3.pie([analysis_results['table_count']], labels=['테이블 수'], autopct='%1.0f', colors=['lightcoral'])
        ax3.set_title('테이블 수')
        
        # 코드 블록 수
        ax4.pie([analysis_results['code_block_count']], labels=['코드 블록 수'], autopct='%1.0f', colors=['lightyellow'])
        ax4.set_title('코드 블록 수')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/document_structure_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_analysis_report(self, analysis_results: Dict[str, Any], output_file: str = "simul_01/document_analysis_report.json"):
        """분석 결과를 JSON 파일로 저장합니다."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        print(f"분석 결과가 {output_file}에 저장되었습니다.")
    
    def print_summary(self, analysis_results: Dict[str, Any]):
        """분석 결과 요약을 출력합니다."""
        print("="*60)
        print("문서 분석 결과 요약")
        print("="*60)
        
        print(f"총 라인 수: {analysis_results['total_lines']:,}")
        print(f"총 문자 수: {analysis_results['total_characters']:,}")
        print(f"헤더 수: {analysis_results['header_count']}")
        print(f"테이블 수: {analysis_results['table_count']}")
        print(f"코드 블록 수: {analysis_results['code_block_count']}")
        
        print("\n헤더 레벨 분포:")
        for level, count in sorted(analysis_results['header_levels'].items()):
            print(f"  레벨 {level}: {count}개")
        
        print("\n중요 키워드 빈도:")
        for keyword, count in sorted(analysis_results['keyword_counts'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {keyword}: {count}회")

def main():
    """메인 실행 함수"""
    analyzer = DocumentAnalyzer("data/dev_center_guide_allmd_touched.md")
    
    # 문서 구조 분석
    analysis_results = analyzer.analyze_document_structure()
    
    # purchaseState 정보 찾기
    purchase_state_info = analyzer.find_purchase_state_info()
    print(f"\npurchaseState 관련 정보 발견: {len(purchase_state_info)}개")
    
    for info in purchase_state_info[:3]:  # 처음 3개만 출력
        print(f"라인 {info['line_number']}: {info['line_content'][:100]}...")
    
    # 테이블 내용 분석
    table_analyses = analyzer.analyze_table_content()
    print(f"\n테이블 분석 완료: {len(table_analyses)}개 테이블")
    
    # 결과 출력
    analyzer.print_summary(analysis_results)
    
    # 시각화 생성
    analyzer.generate_visualization(analysis_results)
    
    # 결과 저장
    analyzer.export_analysis_report(analysis_results)
    
    print("\n문서 분석 완료!")

if __name__ == "__main__":
    main() 