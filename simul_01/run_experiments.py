#!/usr/bin/env python3
"""
RAG 실험 실행 스크립트
다양한 문서 분할 전략을 테스트하고 결과를 분석합니다.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_command(command: str, description: str) -> bool:
    """명령어를 실행하고 결과를 출력합니다."""
    print(f"\n{'='*60}")
    print(f"실행 중: {description}")
    print(f"명령어: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True, encoding='utf-8')
        print("✅ 성공!")
        if result.stdout:
            print("출력:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ 실패!")
        print(f"오류 코드: {e.returncode}")
        if e.stdout:
            print("표준 출력:")
            print(e.stdout)
        if e.stderr:
            print("오류 출력:")
            print(e.stderr)
        return False

def check_dependencies():
    """필요한 의존성을 확인합니다."""
    print("의존성 확인 중...")
    
    required_packages = [
        'langchain',
        'langchain-community',
        'langchain-ollama',
        'langchain-text-splitters',
        'faiss-cpu',
        'matplotlib',
        'seaborn',
        'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 설치 필요")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n다음 패키지들을 설치해주세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_files():
    """필요한 파일들이 존재하는지 확인합니다."""
    print("\n파일 확인 중...")
    
    required_files = [
        "data/dev_center_guide_allmd_touched.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 파일이 없습니다")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n다음 파일들이 필요합니다:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    return True

def main():
    """메인 실행 함수"""
    print("🚀 RAG 실험 실행 스크립트")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 현재 디렉토리 확인
    current_dir = os.getcwd()
    print(f"현재 디렉토리: {current_dir}")
    
    # simul_01 디렉토리 생성
    simul_dir = Path("simul_01")
    simul_dir.mkdir(exist_ok=True)
    
    # 의존성 확인
    if not check_dependencies():
        print("\n❌ 의존성 확인 실패. 필요한 패키지를 설치한 후 다시 실행하세요.")
        return
    
    # 파일 확인
    if not check_files():
        print("\n❌ 파일 확인 실패. 필요한 파일을 준비한 후 다시 실행하세요.")
        return
    
    print("\n✅ 모든 사전 조건이 충족되었습니다!")
    
    # 실험 실행 순서
    experiments = [
        {
            'command': 'cd simul_01 && python document_analyzer.py',
            'description': '문서 구조 분석'
        },
        {
            'command': 'cd simul_01 && python rag_experiment.py',
            'description': 'RAG 실험 실행'
        },
        {
            'command': 'cd simul_01 && python result_analyzer.py',
            'description': '결과 분석 및 시각화'
        }
    ]
    
    # 실험 실행
    success_count = 0
    total_count = len(experiments)
    
    for i, experiment in enumerate(experiments, 1):
        print(f"\n📊 실험 {i}/{total_count}")
        
        if run_command(experiment['command'], experiment['description']):
            success_count += 1
        else:
            print(f"\n⚠️  실험 {i} 실패. 계속 진행하시겠습니까? (y/n)")
            response = input().lower()
            if response != 'y':
                print("실험을 중단합니다.")
                break
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("실험 완료 요약")
    print(f"{'='*60}")
    print(f"성공: {success_count}/{total_count}")
    print(f"실패: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 모든 실험이 성공적으로 완료되었습니다!")
        print("\n생성된 파일들:")
        
        # 생성된 파일 목록 출력
        simul_files = list(simul_dir.glob("*"))
        for file_path in sorted(simul_files):
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"  - {file_path.name} ({size:,} bytes)")
        
        print(f"\n📁 결과 파일들은 {simul_dir} 디렉토리에 저장되었습니다.")
        print("\n다음 파일들을 확인해보세요:")
        print("  - experiment_results_*.json: 실험 결과 데이터")
        print("  - document_analysis_report.json: 문서 분석 결과")
        print("  - detailed_analysis_report.md: 상세 분석 보고서")
        print("  - *.png: 시각화 차트들")
        
    else:
        print(f"\n⚠️  {total_count - success_count}개의 실험이 실패했습니다.")
        print("실패한 실험을 확인하고 다시 실행해보세요.")

if __name__ == "__main__":
    main() 