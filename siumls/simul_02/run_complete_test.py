#!/usr/bin/env python3
"""
RAG 시스템 완전 테스트 스크립트
임베딩 모델 비교 → 최적 모델 선택 → RAG 체인 테스트 순서로 실행
"""

import os
import json
import time
import subprocess
from typing import Dict, Any

def run_embedding_comparison():
    """임베딩 모델 비교 테스트를 실행합니다."""
    print("=" * 60)
    print("1단계: 임베딩 모델 비교 테스트 시작")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ["python", "embedding_model_comparison.py"],
            capture_output=True,
            text=True,
            cwd="simul_02"
        )
        
        if result.returncode == 0:
            print("임베딩 모델 비교 테스트 완료")
            print(result.stdout)
        else:
            print("임베딩 모델 비교 테스트 실패")
            print(result.stderr)
            return None
            
    except Exception as e:
        print(f"임베딩 모델 비교 테스트 실행 중 오류: {str(e)}")
        return None

def load_embedding_results():
    """임베딩 테스트 결과를 로드합니다."""
    try:
        with open("simul_02/embedding_comparison_results.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("임베딩 테스트 결과 파일을 찾을 수 없습니다.")
        return None

def get_best_embedding_model(results: Dict[str, Any]) -> str:
    """최적의 임베딩 모델을 선택합니다."""
    if not results or 'ranking' not in results:
        return "codellama"  # 기본값
    
    # 성공한 모델 중에서 최고 성능 모델 선택
    for model_key, result in results['ranking']:
        if result['status'] == 'success':
            print(f"선택된 최적 임베딩 모델: {model_key}")
            print(f"  관련성 점수: {result['avg_relevance_score']:.3f}")
            print(f"  평균 검색 시간: {result['avg_search_time']:.3f}초")
            return result['model_name']
    
    return "codellama"  # 기본값

def update_rag_chain_config(best_embedding_model: str):
    """RAG 체인 테스트 설정을 업데이트합니다."""
    print(f"\n최적 임베딩 모델로 RAG 체인 설정 업데이트: {best_embedding_model}")
    
    # rag_chain_test.py 파일에서 임베딩 모델 설정 업데이트
    rag_chain_file = "simul_02/rag_chain_test.py"
    
    try:
        with open(rag_chain_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 기본 임베딩 모델 설정을 찾아서 업데이트
        import re
        pattern = r'best_embedding_model = "([^"]+)"'
        replacement = f'best_embedding_model = "{best_embedding_model}"'
        
        updated_content = re.sub(pattern, replacement, content)
        
        with open(rag_chain_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("RAG 체인 설정 업데이트 완료")
        
    except Exception as e:
        print(f"RAG 체인 설정 업데이트 실패: {str(e)}")

def run_rag_chain_test():
    """RAG 체인 테스트를 실행합니다."""
    print("\n" + "=" * 60)
    print("2단계: RAG 체인 테스트 시작")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ["python", "rag_chain_test.py"],
            capture_output=True,
            text=True,
            cwd="simul_02"
        )
        
        if result.returncode == 0:
            print("RAG 체인 테스트 완료")
            print(result.stdout)
        else:
            print("RAG 체인 테스트 실패")
            print(result.stderr)
            
    except Exception as e:
        print(f"RAG 체인 테스트 실행 중 오류: {str(e)}")

def load_rag_results():
    """RAG 테스트 결과를 로드합니다."""
    try:
        with open("simul_02/rag_chain_results.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("RAG 테스트 결과 파일을 찾을 수 없습니다.")
        return None

def print_final_summary(embedding_results: Dict[str, Any], rag_results: Dict[str, Any]):
    """최종 요약을 출력합니다."""
    print("\n" + "=" * 60)
    print("최종 테스트 결과 요약")
    print("=" * 60)
    
    print(f"테스트 완료 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if embedding_results:
        print(f"\n임베딩 모델 테스트:")
        print(f"  테스트된 모델 수: {embedding_results.get('total_models_tested', 0)}")
        print(f"  성공한 모델 수: {embedding_results.get('successful_models', 0)}")
        print(f"  최고 성능 모델: {embedding_results.get('best_model', 'N/A')}")
    
    if rag_results:
        print(f"\nRAG 체인 테스트:")
        print(f"  사용된 임베딩 모델: {rag_results.get('embedding_model', 'N/A')}")
        print(f"  테스트된 LLM 모델 수: {rag_results.get('total_models_tested', 0)}")
        print(f"  성공한 모델 수: {rag_results.get('successful_models', 0)}")
        print(f"  최고 성능 LLM 모델: {rag_results.get('best_model', 'N/A')}")
    
    print("\n최적 조합:")
    if embedding_results and rag_results:
        best_embedding = embedding_results.get('best_model', 'N/A')
        best_llm = rag_results.get('best_model', 'N/A')
        print(f"  임베딩 모델: {best_embedding}")
        print(f"  LLM 모델: {best_llm}")
    
    print("\n결과 파일:")
    print("  - 임베딩 비교 결과: simul_02/embedding_comparison_results.json")
    print("  - RAG 체인 결과: simul_02/rag_chain_results.json")

def main():
    """메인 실행 함수"""
    print("RAG 시스템 완전 테스트 시작")
    print("=" * 60)
    
    # 1단계: 임베딩 모델 비교 테스트
    run_embedding_comparison()
    
    # 임베딩 결과 로드
    embedding_results = load_embedding_results()
    
    if embedding_results:
        # 최적 임베딩 모델 선택
        best_embedding_model = get_best_embedding_model(embedding_results)
        
        # RAG 체인 설정 업데이트
        update_rag_chain_config(best_embedding_model)
        
        # 2단계: RAG 체인 테스트
        run_rag_chain_test()
        
        # RAG 결과 로드
        rag_results = load_rag_results()
        
        # 최종 요약 출력
        print_final_summary(embedding_results, rag_results or {})
    else:
        print("임베딩 테스트 결과를 로드할 수 없어 RAG 체인 테스트를 건너뜁니다.")

if __name__ == "__main__":
    main() 