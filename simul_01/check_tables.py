#!/usr/bin/env python3
"""
purchaseState가 포함된 테이블들을 확인하는 스크립트
"""

import json
import re
from typing import List, Dict, Any

def load_analysis_report(file_path: str = "document_analysis_report.json"):
    """분석 결과를 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_purchase_state_tables(raw_text: str) -> List[Dict[str, Any]]:
    """purchaseState가 포함된 테이블들을 찾습니다."""
    table_pattern = re.compile(r'\|.*\|.*\n\|[\s\-:|]+\|\n(\|.*\|\n)*')
    table_matches = table_pattern.finditer(raw_text)
    
    purchase_state_tables = []
    for match in table_matches:
        table_content = match.group(0)
        
        # purchaseState 관련 키워드 검색 (오타 포함)
        if any(keyword in table_content for keyword in [
            'purchaseState', 'purcahseState', 'purchasestate', 'purchase_state',
            'COMPLETED', 'CANCELED', '결제완료', '취소'
        ]):
            lines = table_content.strip().split('\n')
            
            # 헤더 행 추출
            if len(lines) >= 2:
                header_row = lines[0]
                headers = [col.strip() for col in header_row.split('|')[1:-1]]
                
                # 데이터 행들
                data_rows = []
                for line in lines[2:]:
                    if line.strip() and '|' in line:
                        data_cells = [cell.strip() for cell in line.split('|')[1:-1]]
                        data_rows.append(data_cells)
                
                purchase_state_tables.append({
                    'headers': headers,
                    'data_rows': data_rows,
                    'content': table_content,
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
    
    return purchase_state_tables

def main():
    """메인 실행 함수"""
    print("🔍 purchaseState 관련 테이블 확인")
    
    # 원본 문서 로드
    with open("../data/dev_center_guide_allmd_touched.md", 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # purchaseState 관련 테이블 찾기
    tables = find_purchase_state_tables(raw_text)
    
    print(f"\n발견된 purchaseState 관련 테이블: {len(tables)}개")
    
    for i, table in enumerate(tables, 1):
        print(f"\n{'='*80}")
        print(f"테이블 {i}")
        print(f"{'='*80}")
        
        # 헤더 출력
        if table['headers']:
            print("헤더:")
            print(" | ".join(table['headers']))
            print("-" * 80)
        
        # 데이터 행 출력
        print("데이터:")
        for row in table['data_rows']:
            print(" | ".join(row))
        
        # purchaseState 관련 키워드 강조
        print("\n관련 키워드:")
        keywords_found = []
        for keyword in ['purchaseState', 'purcahseState', 'COMPLETED', 'CANCELED', '결제완료', '취소']:
            if keyword in table['content']:
                keywords_found.append(keyword)
        
        print(", ".join(keywords_found))
        print()

if __name__ == "__main__":
    main() 