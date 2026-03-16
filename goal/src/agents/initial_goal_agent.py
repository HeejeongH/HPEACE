import os
import json

import google.generativeai as genai

from sklearn.linear_model import LogisticRegression

from lib.config import target_goal_cols, checkpoint_path, artifact_path, json_data_path, \
                  diet_cols, life_cols, disease_cols, conf_cols
from lib import utils as uts
from lib.config import json_data_path
from lib.config import API_MODEL, GOOGLE_API_KEY, MAX_RETRY

from lib.model import AutoEncoder as AE
from lib.utils import numpy_converter

from agents.state import PatientState
from agents.schema import GoalList 


def get_goal_region(filtered_keys):
    """SHAP 기반 위험요인 순위에 따라 목표 영역 결정"""
    goals = [filtered_keys[0], filtered_keys[1], filtered_keys[2]]

    try:
        # Cluster의 순위(인덱스) 찾기 (0부터 시작하므로 0이 1등)
        cluster_rank_index = filtered_keys.index('Cluster')
        
        # Cluster가 1등 (index 0) -> 목표 1,2,3 모두 '식습관'
        if cluster_rank_index == 0:
            goals[0] = '식습관'
            goals[1] = '식습관'
            goals[2] = '식습관'
            
        # Cluster가 2등 (index 1) -> 목표 1은 유지, 목표 2,3은 '식습관'
        elif cluster_rank_index == 1:
            goals[1] = '식습관'
            goals[2] = '식습관'
            
        # Cluster가 3등 (index 2) -> 목표 1,2는 유지, 목표 3은 '식습관'
        elif cluster_rank_index == 2:
            goals[2] = '식습관'
            
        # Cluster가 4등 이하 (index 3 이상) -> 원래 1,2,3등 유지 (변화 없음)
        else:
            pass

    except ValueError:
        # 만약 데이터에 'Cluster'라는 키가 아예 없다면 그냥 상위 3개를 씁니다.
        pass
    
    # 영문 키워드를 한글로 변환
    keyword_mapping = {
        'Physical activity': '신체활동',
        'Alcohol Consumption': '음주',
        'Current smoking': '흡연',
        '식습관': '식습관'
    }
    mapped_goals = [keyword_mapping.get(g, g) for g in goals]

    final_goals = {
        "목표 1": mapped_goals[0],
        "목표 2": mapped_goals[1],
        "목표 3": mapped_goals[2]
    }

    goals_json = json.dumps(final_goals, default=numpy_converter, ensure_ascii=False, indent=4)
    return goals_json
        
def generate_action_plan(model, goals, json_data_str, diet_var_def, life_var_def, feedback=None, previous_goal=None):
    """
    JSON 데이터를 Gemini에게 보내서 목표 1, 2, 3에 대한 구체적인 가이드를 받습니다.
    """
    
    # 3. 프롬프트 엔지니어링 (페르소나 및 지시사항 부여)
    prompt = f"""
<SYSTEM>
당신은 대사증후군 예방을 위한 개인맞춤형 건강 목표를 설정하는 전문 영양 컨설턴트입니다.
아래 데이터를 분석하고, 정해진 규칙에 따라 목표 3개를 JSON으로 생성하세요.

## 필드별 어투 규칙 (반드시 준수)

모든 텍스트 필드는 아래 지정된 화법으로만 작성하세요. 다른 어미를 사용하면 위반입니다.

| 필드 | 화법 | 종결 어미 | 예시 |
|------|------|----------|------|
| current_status | 공감형 서술체 | ~에요/~이에요 | "현재 물을 거의 안 마시고 있어요." |
| final_goal | 명사형 종결 | ~실천하기/~먹기/~마시기 | "하루 6컵 이상 물 마시기 주 7회 이상 실천하기" |
| reason | 설명형 존댓말 | ~에요/~미칩니다 혼용 | "혈압 관리에 긍정적인 영향을 미칩니다." |
| action_plans | 합쇼체 | ~합니다/~하세요 혼용 | "두부로 대체합니다." / "메뉴를 정해두세요." |

### current_status 세부 규칙
- 반드시 "~에요", "~이에요", "~있어요", "~같아요" 등 부드러운 서술체로 종결
- 긍정적 변화가 있으면 인정한 뒤 개선 여지 언급
- ❌ 금지: "~있습니다", "~됩니다" (딱딱한 합쇼체), "나쁜 습관" (판단형)

### reason 세부 규칙
- 건강 근거 설명 시: "~에 도움을 주며", "~에 긍정적인 영향을 미칩니다" 패턴 사용
- 개인 패턴 설명 시: "~이에요", "~있으니 ~써주세요" 패턴 사용
- 2~4문장. 건강검진 수치와의 연결을 반드시 포함
- ❌ 금지: "~해야 합니다" (당위형), "~하지 않으면 위험합니다" (위협형)

### action_plans 세부 규칙
- 반드시 3개 항목 생성
- 3개 중 최소 1개는 "~합니다" (행동 진술), 최소 1개는 "~하세요" (권유)로 종결
- 대안 제시형 문장 권장: "A 대신 B로 대체합니다", "~할 때 ~하세요"
- ❌ 금지: "~하지 마세요" (부정 명령 시작), "~하십시오" (격식체)
</SYSTEM>

<INPUT_DATA>
{json_data_str}
</INPUT_DATA>

<GOAL_DATA>
{goals}

<CODEBOOK_DIET>
{diet_var_def}
</CODEBOOK_DIET>

<CODEBOOK_LIFESTYLE>
{life_var_def}
</CODEBOOK_LIFESTYLE>

<INSTRUCTIONS>

## STEP 1: 목표 영역 확인

목표 데이터에 "목표 1", "목표 2", "목표 3"이 이미 지정되어 있습니다.
이 영역을 **절대 변경하지 마세요.** 그대로 따르세요.

- 식습관 문항(D1~D19)이 지정된 경우 → 해당 문항에 대한 식습관 목표 생성
- 생활습관(physical_activity, alcohol, smoking)이 지정된 경우 → 해당 생활습관 목표 생성

---

## STEP 2: final_goal 문구 작성

모든 목표 문구는 아래 **통일 포맷**을 따릅니다:

### 포맷 규칙
```
Increase 목표: [행동] (주 X회)
Decrease 목표: [대상] 줄이기 (주 X회 이하) 또는 (주 0회)
```

### 작성 절차
1. 코드북에서 해당 문항의 **가장 건강한 응답**을 찾는다
2. 그 응답에 포함된 빈도/양을 **주 단위 숫자**로 환산한다
3. 아래 포맷에 맞춰 문구를 완성한다

### Increase 목표 (부족한 것을 늘리는 경우)
포맷: `[구체적 음식명] [행동] (주 X회)`

⚠️ **D5(밥양) 특수 규칙**: D5는 "줄이기"가 아니라 **"적정량 유지하기"** 성격입니다.
- 반드시 Increase(direction="Increase")로 처리하세요.
- "밥 한 끼 1공기 이하로 먹기 (주 7회)" — 매일 적정량을 지켰는지 긍정적으로 추적합니다.
- ❌ 절대 금지: "밥 1공기 초과 섭취 줄이기 (주 0회)" ← 이렇게 Decrease로 바꾸지 마세요.

| 문항 | final_goal 예시 | direction |
|------|----------------|-----------|
| D5 밥양 | 밥 한 끼 1공기 이하로 먹기 (주 7회) | Increase |
| D7 단백질 | 고기·생선·두부·달걀 반찬 하루 1끼 이상 먹기 (주 7회) | Increase |
| D8 채소 | 김치 제외 채소반찬 하루 2끼 이상 먹기 (주 14회) | Increase |
| D9 유제품 | 우유·유제품 하루 1회 이상 먹기 (주 7회) | Increase |
| D10 과일 | 과일 먹기 (주 7회) | Increase |
| D11 물 | 물 하루 6컵 이상 마시기 (주 7회) | Increase |
| 신체활동 | 중강도 이상 운동 하기 (주 5회 이상, 1회 30분 이상) | Increase |

### Decrease 목표 (과다한 것을 줄이는 경우)
포맷: `[구체적 음식명] 줄이기 (주 X회 이하)` 또는 `(주 0회)`

| 문항 | final_goal 예시 |
|------|----------------|
| D12 튀김 | 튀김·부침개 줄이기 (주 2회 이하) |
| D13 고지방육류 | 삼겹살·갈비 등 기름진 고기 줄이기 (주 2회 이하) |
| D14 가공식품 | 라면·즉석식품 등 가공식품 줄이기 (주 2회 이하) |
| D15 소금추가 | 소금·간장 추가 사용 줄이기 (주 0회) |
| D16 짠음식 | 국물·젓갈·장아찌 등 짠 음식 줄이기 (주 2회 이하) |
| D17 단음식 | 케이크·과자·아이스크림 등 단 음식 줄이기 (주 0회) |
| D18 음료 | 콜라·주스 등 단 음료 줄이기 (주 2회 이하) |
| 음주 | 음주 줄이기 (주 0회) |
| 흡연 | 흡연 줄이기 (주 0회) |

### 문구 작성 시 금지 표현
- ❌ "~섭취", "~실천하기", "~제한하기" → 딱딱함
- ❌ "튀김류", "고지방 육류", "인스턴트 식품" → 추상적
- ✅ "~먹기", "~마시기", "~줄이기" → 쉬운 동사 사용

---

## STEP 3: direction 결정

- Increase: 현재 부족하거나 적정량 유지가 필요 → 늘리거나 유지하는 목표 (D5~D11, 신체활동)
  - ⚠️ D5(밥양)은 "적정량 유지" 성격이지만 반드시 Increase로 처리
- Decrease: 현재 과다 → 줄여야 하는 목표 (D3, D4, D12~D18, 음주, 흡연)

---

## STEP 4: daily_goal 계산

- Increase: ceil(주간 목표 횟수 ÷ 7) → "X회"
  - 예: 주 7회 → "1회", 주 14회 → "2회", 주 5회 → "1회"
- Decrease: 항상 "0회"

---

## STEP 5: current_status 작성

참여자에게 직접 보여주는 텍스트입니다.
반드시 **공감형 서술체(~에요/~이에요)**로 작성하세요.

작성 절차:
1. 코드북에서 해당 응답 값의 정의를 찾는다
2. 쉬운 말로 바꿔 쓴다
3. 관련 건강검진 이상 소견이 있으면 추가한다

예시:
- "현재 튀김·부침개를 하루 2회 이상 드시고 있어요."
- "현재 짜게 드시는 편이에요. 수축기 혈압이 145mmHg로 정상 기준을 넘고 있어서, 나트륨 관리가 특히 중요해요."
- "채소 반찬을 주 3~6회 정도 드시고 있어요. 조금만 더 늘리면 좋겠어요."

⚠️ 어투 주의:
- ✅ "~드시고 있어요", "~중요해요", "~좋겠어요"
- ❌ "~드시고 있습니다", "~중요합니다" (합쇼체 금지)

---

## STEP 6: reason 작성

이 목표가 왜 이 사용자에게 중요한지, **건강검진 수치와 연결**하여 2~4문장으로 설명하세요.
**설명형 존댓말(~에요/~미칩니다 혼용)**을 사용합니다.

예시:
- "중성지방 수치가 180mg/dL로 기준(150mg/dL)을 초과하고 있어요. 튀김류 과다 섭취는 중성지방 수치 악화의 주요 원인이며, 조리법 개선이 중성지방 관리에 긍정적인 영향을 미칩니다."
- "충분한 물 섭취는 신진대사를 원활하게 하고 체내 노폐물 배출에 도움을 주며, 혈액순환 개선 및 혈압 관리에 긍정적인 영향을 미칩니다."

⚠️ 어투 주의:
- ✅ "~있어요", "~미칩니다", "~도움을 줍니다" 혼용 가능
- ❌ "~해야 합니다" (당위형), "~하지 않으면 위험합니다" (위협형)

---

## STEP 7: action_plans 작성

3개의 구체적 행동 계획. "무엇을 + 어떻게/언제" 가 드러나야 합니다.
반드시 **합쇼체(~합니다/~하세요 혼용)**로 작성하세요.

### 어투 혼용 규칙
- 3개 항목 중 최소 1개는 "~합니다" (행동 진술), 최소 1개는 "~하세요" (권유)
- 아래 패턴을 참고하세요:

| 패턴 | 종결 어미 | 예시 |
|------|----------|------|
| 대안 제시 | ~합니다 | "삼겹살, 갈비 등 대신 닭가슴살, 생선, 두부로 대체합니다." |
| 상황별 권유 | ~하세요 | "주말 외식 시 미리 저지방 메뉴를 정해두세요." |
| 조리법 제안 | ~합니다 | "육류 섭취 시 삶거나 찌는 조리법을 선택합니다." |
| 습관 형성 | ~하세요 | "물을 마시기 쉬운 곳에 두고 습관적으로 마시세요." |
| 보충 제안 | ~좋습니다 | "수분 섭취를 돕기 위해 보리차나 허브차를 마시는 것도 좋습니다." |

- ❌ "건강한 음식을 먹습니다." (추상적)
- ❌ "기름진 고기를 먹지 마세요." (부정 명령 시작)

나이, 성별, 현재 식습관 수준을 고려하여 현실성 있게 작성하세요.

---

## STEP 8: 메타데이터

- `strategy`: 모든 목표에 `"초기설정"` 기입
- `substituted_from`: 모든 목표에 `null`

---

## 식습관이 이미 양호한 경우

해당 문항 응답이 최적값에 근접하면, "유지·심화" 목표로 전환하세요.
- 예: D8 채소 응답 3 (하루 1끼) → "김치 제외 채소반찬 하루 2끼 이상 먹기 (주 14회)"
- current_status에 "현재 양호한 수준이나, 한 단계 더 높일 여지가 있음"을 명시

</INSTRUCTIONS>

<OUTPUT_FORMAT>
반드시 제공된 JSON Schema(GoalList) 구조에 맞추어 유효한 JSON만 출력하세요.
마크다운 백틱이나 설명 텍스트 없이 순수한 JSON 문자열만 출력합니다.
</OUTPUT_FORMAT>
    """


    if feedback and previous_goal:
        refinement_prompt = f"""

        ----------------------------------------------------------------
        🚨 [중요: 수정 요청] 🚨
        당신은 이전에 목표를 생성했으나, 검증 과정에서 다음과 같은 지적을 받았습니다.
        반드시 아래 피드백을 반영하여 **다시** 작성하세요.
        
        [이전 생성 결과]
        {previous_goal}
        
        [받은 피드백 (지적 사항)]
        {feedback}
        
        위 [받은 피드백]을 해결할 수 있도록 목표를 수정하거나 다시 작성해주세요.
        ----------------------------------------------------------------
        """
        prompt += refinement_prompt
    
    try:
        # 4. API 호출 및 응답 생성        
        
        # Generation Config에 Pydantic 스키마 주입
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=GoalList, 
        )
        
        response = model.generate_content(
            prompt, 
            generation_config=generation_config
        )

        return response.text
    except Exception as e:
        print(f"❌ API 호출 중 오류 발생: {e}")
        return None  # None 반환으로 변경 (에러 구분 용이)



def initial_goal_node(state: PatientState):
    """LLM 기반 개인 맞춤 목표 3개 생성"""

    genai.configure(api_key=GOOGLE_API_KEY)    
    model = genai.GenerativeModel(API_MODEL)
    
    diet_var_def = uts.load_json(os.path.join(json_data_path, 'diet_variable_definition.json'))
    life_var_def = uts.load_json(os.path.join(json_data_path, 'life_variable_definition.json'))

    health_analysis_results = state['health_analysis_results']
    health_analysis_results_dict = json.loads(health_analysis_results)
    goals = get_goal_region(health_analysis_results_dict['분석 결과']['위험 요인 순위'])

    feedback = state.get('initial_goal_feedback')
    previous_goal = state.get('initial_goal')

    if feedback and "PASS" in feedback:
        feedback = None

    initial_goal = generate_action_plan(model, goals, health_analysis_results, diet_var_def, life_var_def, feedback=feedback, previous_goal=previous_goal)

    # API 호출 실패 시 retry_count 증가 (빈 결과 방지)
    if initial_goal is None:
        retry_count = state.get('retry_count', 0) + 1
        return {
            'initial_goal': json.dumps({"goals": []}),
            'initial_goal_feedback': f"API 호출 실패 (retry {retry_count}/{MAX_RETRY})",
            'retry_count': retry_count
        }
  
    return {
        'goals': goals,
        'initial_goal': initial_goal
    }  

def initial_goal_reflection_node(state: PatientState):

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(API_MODEL)

    # 코드북 로드 (reflection에서도 코드북 참조 필요)
    diet_var_def = uts.load_json(os.path.join(json_data_path, 'diet_variable_definition.json'))
    life_var_def = uts.load_json(os.path.join(json_data_path, 'life_variable_definition.json'))

    prompt = f"""
    당신은 AI가 생성한 '건강 목표'의 논리적 오류와 포맷 준수 여부를 검사하는 엄격한 감사관(Auditor)입니다.
    
    아래 [검사 기준]을 하나라도 위반하면 구체적인 피드백을 출력하세요.
    모든 기준을 완벽히 통과했다면 오직 'PASS'라고만 출력하세요.

    [검사 기준]

    1. **단일 문항 매핑**: 하나의 목표는 하나의 식습관/생활습관 문항에 집중해야 한다.
    2. **우선순위 — 목표 영역 일치**: 생성된 목표 1, 2, 3이 입력 데이터(json_output)의 "목표 1", "목표 2", "목표 3" 영역과 **정확히 일치**하는지 확인한다. 임의로 다른 문항으로 대체했다면 위반이다.
    3. **최종 목표 적합성**: 식습관 목표는 제공된 코드북(정의)에서 도출 가능한 내용이어야 한다. 코드북에 없는 임의의 수치를 사용했다면 위반이다.
    4. **최종 목표 포맷 및 논리**: '최종 목표'는 "주 X회 이상/이하"이라는 구체적인 수치와 기간이 포함되어야 한다.
        - '제한' 목표의 경우 "0회 이하"라는 표현은 비논리적이므로 금지한다. 대신 "주 0회" 또는 "~하지 않기" 등의 올바른 한국어 표현을 사용했는가?
        - 사용자의 목표가 현재 응답보다 더 건강하지 않은 비논리적인 목표는 금지한다.
    5. **Daily 목표 포맷**: 
       - Increase 목표일 경우 주 목표량을 7일로 나눈 올림값 "X회" 형식이여야 한다.
       - Decrease 목표일 경우 "0회" 텍스트가 포함되어야 한다.
    6. **형식 준수**: 모든 목표(1, 2, 3)에 대해 `final_goal`, `daily_goal`, `direction`, `reason`, `action_plans`, `current_status` 항목이 반드시 존재해야 한다.
    7. **문항 ID 포함 여부 (필수)**: `related_item` 항목에는 반드시 `D1`, `D8`, `L_PA` 와 같이 문항 ID가 포함되어야 한다.
    8. **direction 필드 검사**: `direction` 값은 "Increase" 또는 "Decrease" 중 하나여야 한다. Increase인데 목표가 줄이기이거나, Decrease인데 목표가 늘리기인 경우 비논리적이므로 위반이다.
        - **D5(밥양) 특수 규칙**: D5는 반드시 direction="Increase"여야 한다. "밥 초과 섭취 줄이기"처럼 Decrease로 처리했다면 위반이다.
    9. **reason 필드 검사**: `reason`에 건강 검진 수치와의 연관성이 1문장 이상 포함되어야 한다. 단순 "건강에 좋다" 수준은 부족하다.
    10. **생활습관 목표 적합성**: 생활습관 목표(신체활동/음주/흡연)인 경우, 생활습관 코드북의 정의를 참조하여 `current_status`가 작성되었는지 확인한다.
    11. **action_plans 구체성**: 각 action_plan이 추상적이지 않고 구체적 행동(무엇을, 어떻게)을 포함하는지 확인한다.
    12. **current_status 어투 검사**: `current_status`가 공감형 서술체(~에요/~이에요/~있어요)로 종결되는지 확인한다. "~있습니다", "~됩니다" 등 합쇼체로 종결된 경우 위반이다.
    13. **reason 어투 검사**: `reason`이 설명형 존댓말(~에요/~미칩니다 혼용)로 작성되었는지 확인한다. "~해야 합니다"(당위형), "~하지 않으면 위험합니다"(위협형)가 포함된 경우 위반이다.
    14. **action_plans 어투 검사**: `action_plans` 3개 항목 중 최소 1개는 "~합니다"(행동 진술), 최소 1개는 "~하세요"(권유)로 종결되어야 한다. 모두 같은 어미로 끝나거나, "~하지 마세요"(부정 명령 시작), "~하십시오"(격식체)가 사용된 경우 위반이다.

    [식습관 변수 정의 (코드북)]
    {diet_var_def}

    [생활습관 변수 정의 (코드북)]
    {life_var_def}

    [사용자 입력 데이터 및 SHAP 분석 결과]
    {state['health_analysis_results']}

    [목표 영역]
    {state['goals']}
    
    [생성된 목표]
    {state['initial_goal']}
    """

    try:
        feedback = model.generate_content(prompt).text
        
        if "PASS" in feedback.upper():
            goal_str = state.get('initial_goal', '{}')
            try:
                goal_json = json.loads(goal_str)
                goal_json['week'] = state.get('week', 1)
                
                # 기존 리스트 복사 후 새로운 목표 추가
                accumulated = state.get('accumulated_weekly_goal', [])
                new_accumulated = accumulated.copy() if accumulated else []
                new_accumulated.append(goal_json)
                
                return {
                    'initial_goal_feedback': feedback,
                    'accumulated_weekly_goal': new_accumulated # State 업데이트
                }
            except json.JSONDecodeError:
                pass

        return {'initial_goal_feedback': feedback}
    
    except Exception as e:
        # API 에러 시에도 retry로 보내되, 횟수 체크는 router에서 수행
        print(f"❌ Reflection API 호출 중 오류 발생: {e}")
        return {'initial_goal_feedback': f"REFLECTION_ERROR: {e}"}


def initial_goal_router(state: PatientState):
    retry_count = state.get('retry_count', 0)
    
    # 최대 재시도 횟수 초과 시 강제 종료
    if retry_count >= MAX_RETRY:
        print(f"\n[Goal Router] 최대 재시도 횟수({MAX_RETRY}) 초과. 현재 결과로 종료합니다.")
        return "end"
    
    feedback = state.get('initial_goal_feedback', '')
    
    if "PASS" in feedback:
        return "end"
    else:
        print(f"\n[Goal Router] Retry ({retry_count + 1}/{MAX_RETRY}): {feedback[:200]}...")
        return "retry"