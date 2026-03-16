# feedback_agent.py
import google.generativeai as genai

from lib.config import API_MODEL, GOOGLE_API_KEY, MAX_RETRY

from agents.state import PatientState
from agents.schema import FeedbackReport

def formalized_feedback_for_groupA(state: PatientState):
    week = state.get('week', 1)
    achievements = state.get('accumulated_achievements', [])
    
    # 이번 주차(week)에 해당하는 달성률 데이터만 필터링
    current_week_achievements = [ach for ach in achievements if ach.get('week') == week]
    
    if not current_week_achievements:
        return {'feedback_report': f"[{week}주차] 데이터가 부족하여 목표 달성률을 계산하지 못했습니다."}
    
    # 가독성 좋은 리포트 문자열 생성
    report_lines = [f"\n📈 [ {week}주차 목표 달성 현황 요약 ]"]
    for ach in current_week_achievements:
        report_lines.append(f" - 문항 {ach['id']}: {ach['rate']}% 달성")

    feedback_str = "\n".join(report_lines)
    
    return {'feedback_report': feedback_str}

def create_contextual_prompt(statistical_summary: str, goal_text: str, reflection_feedback: str = None) -> str:
    prompt = f"""
    ### 역할
    당신은 통계 데이터와 행동 과학 이론을 기반으로 환자의 건강 관리 실패/성공 요인을 날카롭게 분석하는 '행동 변화 분석가(Behavioral Analyst)'입니다.
    
    ### 입력 데이터
    1. [사용자 목표]
    {goal_text}

    2. [파이썬 도구로 추출된 통계적 팩트 (Statistical Summary)]
    아래는 알고리즘이 직접 계산한 엄밀한 수치적 팩트입니다. 당신은 이 데이터 안에서만 추론해야 하며 수치를 지어내면 안 됩니다.
    {statistical_summary}

    ### 분석 지시사항 (COM-B 프레임워크 적용)
    위 통계적 팩트를 바탕으로 사용자가 목표를 수행하면서 보인 행동(Behavior)의 원인을 'COM-B 모델'에 따라 논리적으로 진단하고 리포트를 작성하세요.
    - **C (Capability, 역량)**: 올바른 식습관에 대한 지식, 양 조절 능력, 신체적 체력 등 (관련 지표: 밥양, 물 섭취, 운동 등)
    - **O (Opportunity, 기회)**: 평일/주말의 환경적 요인, 시간 부족, 회식/외식 등 외부 환경 (관련 지표: 평일 vs 주말 편차, 외식 빈도 등)
    - **M (Motivation, 동기)**: 스트레스 해소, 감정적 보상, 습관성 섭취 등 (관련 지표: 단 음식, 커피, 간식 섭취 등)
    - **주의 사항 1**: 각각의 목표에 대한 COM 분석이 이루어져야 합니다. 
    - **주의 사항 2 (금지)**: 최종 리포트는 사용자에게 보여지므로 글을 작성할 때 'D1', 'L3'와 같은 영문+숫자 형태의 문항 ID는 절대 포함하지 마세요. 반드시 '밥 양', '음주', '신체 활동' 등 자연스러운 행동 명칭으로 풀어서 작성하세요.

    ### 출력 형식 (엄격 준수)
    반드시 제공된 JSON Schema(FeedbackReport) 구조에 맞추어 유효한 JSON 포맷으로만 응답하세요. 
    마크다운 백틱(```json)이나 추가적인 설명 텍스트 없이 순수한 JSON 문자열만 출력해야 합니다.
    """
    
    if reflection_feedback:
        prompt += f"""
        ----------------------------------------------------------------
        🚨 [중요: 피드백 수정 요청] 🚨
        이전 생성된 리포트가 검수(Reflection) 과정에서 다음과 같은 지적을 받았습니다:
        {reflection_feedback}
        
        위 지적 사항을 반드시 해결하여 리포트를 수정/보완해 주세요!
        ----------------------------------------------------------------
        """
    return prompt

def feedback_node(state: PatientState):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(API_MODEL)

    current_goal = state.get('adaptive_goal') if state.get('adaptive_goal') else state.get('initial_goal', '목표 없음')
    statistical_summary = state.get('statistical_summary')    
    reflection_msg = state.get('feedback_reflection_feedback')

    report_prompt = create_contextual_prompt(statistical_summary, current_goal, reflection_msg)

    try:
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=FeedbackReport, 
        )
        
        response = model.generate_content(
            report_prompt,
            generation_config=generation_config
        )

        return {'feedback_report': response.text}
    
    except Exception as e:
        return {'feedback_report': f'{{"error": "분석 중 오류 발생: {str(e)}"}}'}

def feedback_reflection_node(state: PatientState):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(API_MODEL)

    report = state.get('feedback_report', '')
    stats = state.get('statistical_summary', '')

    prompt = f"""
    당신은 AI가 작성한 행동 분석 리포트의 논리적 결함과 형식 준수 여부를 검사하는 '최고 감사관(Chief Auditor)'입니다.

    [검사 대상 리포트]
    {report}

    [근거가 되는 통계 데이터 (Ground Truth)]
    {stats}

    [검사 기준]
    1. **COM-B 프레임워크 누락 여부**: 🧠 역량(Capability), 🌍 기회(Opportunity), 🔥 동기(Motivation) 3가지 항목이 모두 명확히 분리되어 작성되었는가?
    2. **통계적 근거 (Grounding)**: 리포트의 주장이 [근거가 되는 통계 데이터]의 수치(달성률, 추세, T-test, 상관관계)를 적절히 인용하여 작성되었는가? (통계 데이터에 없는 허위 사실을 지어내지 않았는가?)
    3. **논리적 타당성**: 통계 결과를 바탕으로 도출한 행동의 '원인 진단'이 상식적이고 논리적인가?

    위 기준을 완벽하게 통과했다면 오직 "PASS" 라고만 출력하세요.
    기준 중 하나라도 위반했거나 보완이 필요하다면, 어떤 부분을 어떻게 수정해야 하는지 구체적인 "수정 지시사항(Feedback)"을 작성하세요.
    """

    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        if "PASS" in result.upper() and len(result) < 10:
            # print("[Auditor] 🟢 리포트 검증 통과!")
            
            # 검증 통과 시 히스토리에 저장
            accumulated = state.get('accumulated_weekly_feedback_reports', [])
            new_accumulated = accumulated.copy() if accumulated else []
            new_accumulated.append(report)
            
            return {
                'feedback_reflection_feedback': 'PASS',
                'accumulated_weekly_feedback_reports': new_accumulated # State 업데이트
            }
        else:
            # print("[Auditor] 🔴 리포트 보완 요청 발생!")
            return {'feedback_reflection_feedback': result}
            
    except Exception as e:
        print(f"❌ Reflection API 호출 중 오류 발생: {e}")
        return {'feedback_reflection_feedback': f"REFLECTION_ERROR: {e}"}


def feedback_router(state: PatientState):
    retry_count = state.get('retry_count', 0)
    
    # 최대 재시도 횟수 초과 시 강제 종료
    if retry_count >= MAX_RETRY:
        print(f"\n[Feedback Router] 최대 재시도 횟수({MAX_RETRY}) 초과. 현재 결과로 종료합니다.")
        return "pass"
    
    feedback = state.get('feedback_reflection_feedback', '')
    
    if 'PASS' in feedback.upper(): 
        return "pass"
    else:
        print(f"\n[Feedback Router] Retry ({retry_count + 1}/{MAX_RETRY}): {feedback[:200]}...")
        return "retry"