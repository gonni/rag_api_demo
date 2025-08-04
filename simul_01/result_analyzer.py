import json
import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from datetime import datetime

class ResultAnalyzer:
    def __init__(self, results_dir: str = "simul_01"):
        self.results_dir = results_dir
        self.results_files = self.find_results_files()
        
    def find_results_files(self) -> List[str]:
        """결과 파일들을 찾습니다."""
        results_files = []
        for file in os.listdir(self.results_dir):
            if file.startswith("experiment_results_") and file.endswith(".json"):
                results_files.append(os.path.join(self.results_dir, file))
        return sorted(results_files)
    
    def load_results(self, file_path: str) -> Dict[str, Any]:
        """결과 파일을 로드합니다."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_single_experiment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """단일 실험 결과를 분석합니다."""
        analysis = {
            'strategy_comparison': {},
            'query_performance': {},
            'overall_stats': {}
        }
        
        # 전략별 비교
        for strategy_name, strategy_data in results.items():
            doc_count = strategy_data['doc_count']
            query_results = strategy_data['query_results']
            
            # 평균 관련성 점수 계산
            avg_scores = []
            for query_result in query_results:
                avg_scores.append(query_result['avg_relevance_score'])
            
            overall_avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0
            
            analysis['strategy_comparison'][strategy_name] = {
                'doc_count': doc_count,
                'avg_relevance_score': overall_avg,
                'query_count': len(query_results),
                'max_score': max(avg_scores) if avg_scores else 0,
                'min_score': min(avg_scores) if avg_scores else 0
            }
        
        # 쿼리별 성능 분석
        query_performance = {}
        for strategy_name, strategy_data in results.items():
            for query_result in strategy_data['query_results']:
                query = query_result['query']
                if query not in query_performance:
                    query_performance[query] = {}
                
                query_performance[query][strategy_name] = {
                    'avg_relevance_score': query_result['avg_relevance_score'],
                    'total_docs': query_result['total_docs']
                }
        
        analysis['query_performance'] = query_performance
        
        # 전체 통계
        analysis['overall_stats'] = {
            'total_strategies': len(results),
            'total_queries': len(next(iter(results.values()))['query_results']) if results else 0,
            'best_strategy': max(analysis['strategy_comparison'].items(), 
                               key=lambda x: x[1]['avg_relevance_score'])[0] if analysis['strategy_comparison'] else None
        }
        
        return analysis
    
    def compare_experiments(self, experiment_files: List[str] | None = None) -> Dict[str, Any]:
        """여러 실험 결과를 비교합니다."""
        if experiment_files is None:
            experiment_files = self.results_files
        
        comparison = {
            'experiments': {},
            'best_performers': {},
            'trends': {}
        }
        
        for file_path in experiment_files:
            experiment_name = os.path.basename(file_path).replace('.json', '')
            results = self.load_results(file_path)
            analysis = self.analyze_single_experiment(results)
            
            comparison['experiments'][experiment_name] = analysis
            
            # 최고 성능 전략 찾기
            best_strategy = analysis['overall_stats']['best_strategy']
            if best_strategy:
                if best_strategy not in comparison['best_performers']:
                    comparison['best_performers'][best_strategy] = 0
                comparison['best_performers'][best_strategy] += 1
        
        return comparison
    
    def generate_comparison_visualization(self, comparison: Dict[str, Any], output_dir: str = "simul_01"):
        """비교 결과를 시각화합니다."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. 전략별 성능 비교 (최신 실험 기준)
        latest_experiment = list(comparison['experiments'].keys())[-1]
        latest_analysis = comparison['experiments'][latest_experiment]
        
        strategies = list(latest_analysis['strategy_comparison'].keys())
        avg_scores = [latest_analysis['strategy_comparison'][s]['avg_relevance_score'] for s in strategies]
        doc_counts = [latest_analysis['strategy_comparison'][s]['doc_count'] for s in strategies]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 평균 관련성 점수
        bars1 = ax1.bar(strategies, avg_scores, color='skyblue')
        ax1.set_title('전략별 평균 관련성 점수')
        ax1.set_ylabel('평균 점수')
        ax1.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, score in zip(bars1, avg_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{score:.2f}', ha='center', va='bottom')
        
        # 문서 수
        bars2 = ax2.bar(strategies, doc_counts, color='lightcoral')
        ax2.set_title('전략별 생성된 문서 수')
        ax2.set_ylabel('문서 수')
        ax2.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, count in zip(bars2, doc_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/strategy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 쿼리별 성능 비교
        query_performance = latest_analysis['query_performance']
        if query_performance:
            queries = list(query_performance.keys())
            query_data = []
            
            for query in queries:
                for strategy in strategies:
                    if strategy in query_performance[query]:
                        score = query_performance[query][strategy]['avg_relevance_score']
                        query_data.append({
                            'Query': query,
                            'Strategy': strategy,
                            'Score': score
                        })
            
            if query_data:
                df = pd.DataFrame(query_data)
                plt.figure(figsize=(12, 8))
                
                # 히트맵 생성
                pivot_df = df.pivot(index='Query', columns='Strategy', values='Score')
                sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.2f')
                plt.title('쿼리별 전략 성능 비교')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/query_performance_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. 최고 성능 전략 분포
        if comparison['best_performers']:
            plt.figure(figsize=(10, 6))
            best_strategies = list(comparison['best_performers'].keys())
            counts = list(comparison['best_performers'].values())
            
            plt.bar(best_strategies, counts, color='lightgreen')
            plt.title('실험별 최고 성능 전략 분포')
            plt.ylabel('최고 성능 횟수')
            plt.xlabel('전략')
            
            # 값 표시
            for i, count in enumerate(counts):
                plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/best_strategy_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_detailed_report(self, comparison: Dict[str, Any], output_file: str = "simul_01/detailed_analysis_report.md"):
        """상세한 분석 보고서를 생성합니다."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# RAG 실험 결과 상세 분석 보고서\n\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 최신 실험 결과
            latest_experiment = list(comparison['experiments'].keys())[-1]
            latest_analysis = comparison['experiments'][latest_experiment]
            
            f.write("## 최신 실험 결과 요약\n\n")
            f.write(f"**실험명**: {latest_experiment}\n\n")
            
            # 전략별 성능
            f.write("### 전략별 성능 비교\n\n")
            f.write("| 전략 | 문서 수 | 평균 점수 | 최고 점수 | 최저 점수 |\n")
            f.write("|------|---------|-----------|-----------|-----------|\n")
            
            for strategy, data in latest_analysis['strategy_comparison'].items():
                f.write(f"| {strategy} | {data['doc_count']} | {data['avg_relevance_score']:.2f} | "
                       f"{data['max_score']:.2f} | {data['min_score']:.2f} |\n")
            
            f.write("\n")
            
            # 최고 성능 전략
            best_strategy = latest_analysis['overall_stats']['best_strategy']
            if best_strategy:
                f.write(f"**최고 성능 전략**: {best_strategy}\n\n")
            
            # 쿼리별 성능
            f.write("### 쿼리별 성능 분석\n\n")
            for query, performances in latest_analysis['query_performance'].items():
                f.write(f"#### 쿼리: {query}\n\n")
                f.write("| 전략 | 평균 점수 |\n")
                f.write("|------|-----------|\n")
                
                for strategy, perf in performances.items():
                    f.write(f"| {strategy} | {perf['avg_relevance_score']:.2f} |\n")
                f.write("\n")
            
            # 전체 통계
            f.write("## 전체 실험 통계\n\n")
            f.write(f"**총 실험 수**: {len(comparison['experiments'])}\n")
            f.write(f"**최고 성능 전략 분포**:\n")
            
            for strategy, count in comparison['best_performers'].items():
                f.write(f"- {strategy}: {count}회\n")
        
        print(f"상세 보고서가 {output_file}에 저장되었습니다.")
    
    def print_summary(self, comparison: Dict[str, Any]):
        """비교 결과 요약을 출력합니다."""
        print("="*60)
        print("실험 결과 비교 요약")
        print("="*60)
        
        latest_experiment = list(comparison['experiments'].keys())[-1]
        latest_analysis = comparison['experiments'][latest_experiment]
        
        print(f"\n최신 실험: {latest_experiment}")
        print(f"총 전략 수: {latest_analysis['overall_stats']['total_strategies']}")
        print(f"총 쿼리 수: {latest_analysis['overall_stats']['total_queries']}")
        
        print("\n전략별 성능:")
        for strategy, data in latest_analysis['strategy_comparison'].items():
            print(f"  {strategy}:")
            print(f"    문서 수: {data['doc_count']}")
            print(f"    평균 점수: {data['avg_relevance_score']:.2f}")
            print(f"    최고 점수: {data['max_score']:.2f}")
            print(f"    최저 점수: {data['min_score']:.2f}")
        
        print(f"\n최고 성능 전략: {latest_analysis['overall_stats']['best_strategy']}")
        
        print("\n전체 실험에서 최고 성능 전략 분포:")
        for strategy, count in comparison['best_performers'].items():
            print(f"  {strategy}: {count}회")

def main():
    """메인 실행 함수"""
    analyzer = ResultAnalyzer()
    
    if not analyzer.results_files:
        print("실험 결과 파일을 찾을 수 없습니다.")
        print("먼저 rag_experiment.py를 실행하여 실험을 수행하세요.")
        return
    
    print(f"발견된 실험 결과 파일: {len(analyzer.results_files)}개")
    for file in analyzer.results_files:
        print(f"  - {os.path.basename(file)}")
    
    # 결과 비교
    comparison = analyzer.compare_experiments()
    
    # 결과 출력
    analyzer.print_summary(comparison)
    
    # 시각화 생성
    analyzer.generate_comparison_visualization(comparison)
    
    # 상세 보고서 생성
    analyzer.generate_detailed_report(comparison)
    
    print("\n결과 분석 완료!")

if __name__ == "__main__":
    main() 