# schema.py
from pydantic import BaseModel, Field
from typing import List, Optional

class GoalItem(BaseModel):
    goal_title: str = Field(description="목표명 (예: 채소반찬 하루 2끼 이상 먹기)")
    action_plans: List[str] = Field(description="구체적 실천 방안 3개. 합쇼체(~합니다/~하세요) 혼용. 3개 중 최소 1개는 '~합니다', 최소 1개는 '~하세요'로 종결. ❌금지: '~하지 마세요'(부정명령), '~하십시오'(격식체)")

    related_id: str = Field(description="대괄호를 제외한 관련 문항 ID (예: D5, L1, D16)")
    goal_measure: str = Field(description="측정 대상 한글명 (예: 곡류, 밥양, 단백질 반찬, 신체활동, 튀김·부침개)")
    
    direction: str = Field(description="목표 방향. 반드시 'Increase' 또는 'Decrease' 중 하나. Increase: 부족한 것을 늘리는 목표, Decrease: 과다한 것을 줄이는 목표")
    is_more_than: bool = Field(description="Increase 목표이면 True, Decrease 목표이면 False")
    
    strategy: str = Field(description="적용된 목표 적응 전략. 반드시 다음 중 하나: '초기설정', '유지', '강화', '하향', '교체'")
    substituted_from: Optional[str] = Field(description="'교체' 전략인 경우에만, 직전에 수행했던 기존 목표의 문항 ID (예: D16). 그 외 전략이면 null")
    
    final_goal: str = Field(description="최종 목표 텍스트. Increase: '[행동] (주 X회)' 형태 (예: 과일 먹기 (주 7회)). Decrease: '[대상] 줄이기 (주 X회 이하)' 형태 (예: 튀김·부침개 줄이기 (주 2회 이하))")
    weekly_target_num: int = Field(description="주간 목표 횟수 (반드시 정수만 입력, 예: 7, 2, 0)")
    daily_goal: str = Field(description="일일 목표 텍스트. Increase: ceil(weekly_target_num/7)+'회' (예: 1회, 2회). Decrease: 항상 '0회'")
    
    current_status: str = Field(description="현재 상태 요약. 반드시 공감형 서술체(~에요/~이에요)로 종결. 예: '현재 물을 거의 안 마시고 있어요.' ❌금지: '~있습니다'(합쇼체)")
    reason: str = Field(description="이 목표가 이 사용자에게 중요한 이유. 설명형 존댓말(~에요/~미칩니다 혼용)로 2~4문장 작성. 건강검진 수치와 연결 필수. ❌금지: '~해야 합니다'(당위형)")
    
class GoalList(BaseModel):
    goals: List[GoalItem]

class COMBAnalysis(BaseModel):
    capability: str = Field(description="역량(Capability) 원인 분석 (예: 지식, 양 조절 능력, 신체적 체력 등)")
    opportunity: str = Field(description="기회 및 환경(Opportunity) 원인 분석 (예: 평일/주말 환경, 외식 등)")
    motivation: str = Field(description="동기(Motivation) 원인 분석 (예: 감정적 보상, 스트레스 해소, 습관성 등)")

class FeedbackReport(BaseModel):
    behavior_summary: str = Field(description="통계 팩트를 바탕으로 한 이번 주 주요 행동 패턴 요약 (3-4줄)")
    com_b_analysis: COMBAnalysis = Field(description="COM-B 모델에 따른 논리적이고 구체적인 행동 원인 분석")