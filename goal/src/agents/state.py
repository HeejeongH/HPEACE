from typing import TypedDict, List, Dict, Any, Optional
import pandas as pd

# ==========================================
# 1. 상태(State) 정의
# 에이전트들이 서로 공유할 데이터 바구니(Shared State)입니다.
# ==========================================
class PatientState(TypedDict):
    # 입력 데이터
    r_id: Optional[str]
    survey_data: Dict[str, Any]  # 전체 데이터
    diet_data: Dict[str, int]    # 식습관 데이터
    life_data: Dict[str, int]    # 생활습관 
    conf_data: Dict[str, Any]    # 나이, 성별
    disease_data: Dict[str, int] # 질병 유무
    disease_value_data: Dict[str, float] # 질병 수치 결과

    # --- [Health Analyzer 결과] ---
    health_analysis_results: Optional[str]   

    # --- [Initial Goal Agent 결과] ---
    goals: Optional[str]         # 목표 영역 
    initial_goal: Optional[str]  # 초기 목표 3개 (GoalList 스키마)
    initial_goal_feedback: Optional[str]

    # 주간 데이터
    week: Optional[int] # 몇주차?

    # --- [Health Monitoring Agent 결과] ---
    weekly_data: Optional[pd.DataFrame]                         # 이번 주 식습관 및 생활습관 데이터
    accumulated_weekly_data: Optional[List[pd.DataFrame]]       # 누적된 Raw Data List (Python 연산용)
    accumulated_achievements: Optional[List[Dict[str, Any]]]    # 주차별 목표 달성률 누적 기록 ({week, id, rate})
    daily_logs: Optional[List[Dict[str, float]]]                # 일별 평균 달성 점수 누적 ({week, day, average_score})
    weekly_insights: Optional[List[Dict[str, Any]]]             # 현재 주차의 통계 인사이트 
    accumulated_weekly_insights: Optional[List[Dict[str, Any]]] # 매주 인사이트 누적 리스트
    streak_data: Optional[Dict[str, int]]                       # 연속 달성 기록 ({current_streak, max_streak})
    statistical_summary: Optional[str]                          # 통계 분석 결과 텍스트 (LLM 입력용)

    # --- [Feedback Agent 결과] ---
    feedback_report: Optional[str]                           # 피드백 리포트 (FeedbackReport 스키마)
    feedback_reflection_feedback: Optional[str]
    accumulated_weekly_feedback_reports: Optional[List[str]] # 누적된 주간 피드백 정보
    
    # --- [Adaptive Goal Agent 결과] ---
    adaptive_goal: Optional[str]                            # 적응형 목표 3개 (GoalList 스키마)
    adaptive_goal_feedback: Optional[str]    
    accumulated_weekly_goal: Optional[List[Dict[str, Any]]] # 매주 목표 누적 리스트 (week 포함)
    goal_adjustment_banner: Optional[Dict[str, Any]]        # 동적 목표 조정 알림 배너용 데이터