# run_multi_agents.py
import os
import argparse
import warnings

from langgraph.graph import StateGraph, START, END

from agents.state import PatientState
from agents.health_analysis_agent import health_analysis_node
from agents.initial_goal_agent import initial_goal_node, initial_goal_reflection_node, initial_goal_router
from agents.health_monitoring_agent import health_monitoring_node_for_groupA, health_monitoring_node
from agents.feedback_agent import formalized_feedback_for_groupA, feedback_node, feedback_reflection_node, feedback_router
from agents.adaptive_goal_agent import adaptive_goal_node, adaptive_goal_reflection_node, adaptive_goal_router

from lib import utils as uts
from lib.display import display_goal, display_feedback, display_change, display_calendar
from lib.virtual_input import get_patient_input 

warnings.filterwarnings('ignore')

# 주차(Week)에 따른 라우팅 함수
def start_router(state: PatientState):
    if state.get('week', 1) == 1:
        return 'week_1'
    else:
        return 'week_n'

def main(args):
    print("="*50, "Multi-Agents AI System Simulation Start", "="*50)

    # 1. 강남검진센터 방문 후 검진 결과 수집
    current_state = get_patient_input(args.target_index, args.data_dir)

    # 중재군 A
    if args.group == 'A':

        # 오케스트레이터 설계
        orc = StateGraph(PatientState)

        # 노드 추가
        orc.add_node('health_analyzer', health_analysis_node)
        orc.add_node('initial_goal_generator', initial_goal_node)
        orc.add_node('initial_goal_reflector', initial_goal_reflection_node)
        
        # 2주차 이후를 담당할 노드 추가
        orc.add_node('health_monitor_for_groupA', health_monitoring_node_for_groupA) 
        orc.add_node('formalized_feedback', formalized_feedback_for_groupA)

        # 엣지 (workflow) 설정 변경
        # START에서 조건부 엣지로 분기합니다.
        orc.add_conditional_edges(
            START, 
            start_router,
            {
                'week_1': 'health_analyzer',            # 1주차는 기존대로 진단 및 목표 설정으로
                'week_n': 'health_monitor_for_groupA'   # 2주차 이상은 모니터링으로
            }
        )
        
        # [1주차 흐름]
        orc.add_edge('health_analyzer', 'initial_goal_generator')
        orc.add_edge('initial_goal_generator', 'initial_goal_reflector')
        orc.add_conditional_edges(
            'initial_goal_reflector', initial_goal_router,
            {
                'end': END,  
                'retry': 'initial_goal_generator' 
            }
        )       
        
        # [2주차 이상 흐름]
        orc.add_edge('health_monitor_for_groupA', 'formalized_feedback') # 피드백 노드로 연결 (선택)
        orc.add_edge('formalized_feedback', END)

        # 시뮬레이션 컴파일
        app = orc.compile()
        
        # 시뮬레이션 
        current_state['week'] = 1
        current_state['accumulated_weekly_data'] = []
        current_state['accumulated_achievements'] = []
        current_state['daily_logs'] = []
        current_state['goal_history'] = []

        try:
            while True:
                print(f"\n" + "="*20 + f" [ {current_state['week']}주차 시작 (중재군 A) ] " + "="*20)
                
                # 에이전트 실행 (LangGraph가 week를 보고 스스로 판단하여 실행)
                current_state = app.invoke(current_state)
                
                # 결과 출력
                display_goal(current_state)
                display_calendar(current_state)

                current_state['week'] += 1
                
                print("\n" + "-"*60)
                input(f"[System] {current_state['week']}주차로 넘어가려면 Enter를 누르세요. (종료하려면 Ctrl+C) ")

        except KeyboardInterrupt:
            print("\n\n[System] 시뮬레이션을 정상적으로 종료합니다.")
            print("최종 누적 주차:", current_state['week'] - 1)
            print("="*60 + "\n")
    
    # 중재군 B
    elif args.group == 'B':

        # 오케스트레이터 설계
        orc = StateGraph(PatientState)

        orc.add_node('health_analyzer', health_analysis_node)
        orc.add_node('initial_goal_generator', initial_goal_node)
        orc.add_node('initial_goal_reflector', initial_goal_reflection_node)
        
        # 2주차 이후를 담당할 노드 추가
        orc.add_node('health_monitor_node', health_monitoring_node) 
        orc.add_node('feedback_node', feedback_node)
        orc.add_node('feedback_reflector', feedback_reflection_node)
        orc.add_node('adaptive_goal_generator', adaptive_goal_node)
        orc.add_node('adaptive_goal_reflector', adaptive_goal_reflection_node)

        # 엣지 (workflow) 설정 변경
        # START에서 조건부 엣지로 분기합니다.
        orc.add_conditional_edges(
            START, 
            start_router,
            {
                'week_1': 'health_analyzer', # 1주차는 기존대로 진단 및 목표 설정으로
                'week_n': 'health_monitor_node'   # 2주차 이상은 모니터링으로
            }
        )
        
        # [1주차 흐름]
        orc.add_edge('health_analyzer', 'initial_goal_generator')
        orc.add_edge('initial_goal_generator', 'initial_goal_reflector')
        orc.add_conditional_edges(
            'initial_goal_reflector', initial_goal_router,
            {
                'end': END,  
                'retry': 'initial_goal_generator' 
            }
        )       
        
        # [2주차 이상 흐름]
        orc.add_edge('health_monitor_node', 'feedback_node') 
        orc.add_edge('feedback_node', 'feedback_reflector')
        orc.add_conditional_edges(
            'feedback_reflector', feedback_router,
            {
                'pass': 'adaptive_goal_generator',  
                'retry': 'feedback_node' 
            }
        )     
        orc.add_edge('adaptive_goal_generator', 'adaptive_goal_reflector')
        orc.add_conditional_edges(
            'adaptive_goal_reflector', adaptive_goal_router,
            {
                'end': END,
                'retry': 'adaptive_goal_generator'
            }
        )

        # 시뮬레이션 컴파일
        app = orc.compile()
        
        # 시뮬레이션 
        current_state['week'] = 1
        current_state['accumulated_weekly_data'] = []
        current_state['accumulated_achievements'] = []
        current_state['daily_logs'] = []
        current_state['accumulated_weekly_insights'] = []
        current_state['accumulated_weekly_goal'] = []
        current_state['accumulated_weekly_feedback_reports'] = []

        try:
            while True:
                print(f"\n" + "="*20 + f" [ {current_state['week']}주차 시작 (중재군 B) ] " + "="*20)
                
                # 에이전트 실행 (LangGraph가 week를 보고 스스로 판단하여 실행)
                current_state = app.invoke(current_state)
                
                # 결과 출력
                display_goal(current_state)
                display_feedback(current_state)
                display_change(current_state)
                display_calendar(current_state)

                current_state['week'] += 1
                
                print("\n" + "-"*60)
                input(f"[System] {current_state['week']}주차로 넘어가려면 Enter를 누르세요. (종료하려면 Ctrl+C) ")

        except KeyboardInterrupt:
            print("\n\n[System] 시뮬레이션을 정상적으로 종료합니다.")
            print("최종 누적 주차:", current_state['week'] - 1)
            print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',              type=str, default='../data/processed_data')
    parser.add_argument('--json_dir',              type=str, default='../data/json')
    parser.add_argument('--clustering_result_dir', type=str, default='../results/clustering_result')
    parser.add_argument('--seed',                  type=int, default=42)
    parser.add_argument('--target_index',          type=int, default=0,   help='환자 Index')
    parser.add_argument('--group',                 type=str, default='B', help='중재군 [A, B]')
    args = parser.parse_args()

    uts.set_seed(s=args.seed)
    main(args=args)
