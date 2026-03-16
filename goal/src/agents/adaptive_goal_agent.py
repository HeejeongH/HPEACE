# adaptive_goal_agent.py
import os
import json

import google.generativeai as genai

from lib import utils as uts
from lib.config import API_MODEL, GOOGLE_API_KEY, MAX_RETRY, json_data_path

from agents.state import PatientState
from agents.schema import GoalList
from agents.guardrail import get_guardrail_prompt, validate_goals

def adaptive_goal_node(state: PatientState):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(API_MODEL)

    # 1. State에서 필요한 데이터 로드
    week = state.get('week')
    initial_goal = state.get('initial_goal', '')
    prev_adaptive_goal = state.get('adaptive_goal', initial_goal)
    feedback_report = state.get('feedback_report', '')
    stats_summary = state.get('statistical_summary', '')
    accumulated_achievements = state.get('accumulated_achievements', [])
    reflection_msg = state.get('adaptive_goal_feedback')
    
    meta_path = os.path.join(json_data_path, 'week_survey_meta.json')
    week_meta = uts.load_json(meta_path)
    
    meta_summary = "\n".join([f"- {k}: {v.get('question', '')}" for k, v in week_meta.items()])
    current_week_achievements = [ach for ach in accumulated_achievements if ach.get('week') == week]
    past_ach_str = "\n".join([f"Week {p['week']} - {p['id']}: {p['rate']}%" for p in current_week_achievements])

    # 2. 환자의 질병 데이터 기반 가드레일 텍스트 생성
    disease_data = state.get('disease_data', {})
    guardrail_text = get_guardrail_prompt(disease_data)

    # 3. 적응형 목표 생성
    generation_prompt = f"""
<ROLE>
당신은 대사증후군 예방을 위해 사용자의 달성률과 행동 패턴에 맞춰 목표를 동적으로 재설계하는 적응형 건강 코치입니다.
아래 데이터를 분석하고, 적응 규칙에 따라 다음 주차 목표 3개를 JSON으로 생성하세요.

## 필드별 어투 규칙 (반드시 준수)

모든 텍스트 필드는 아래 지정된 화법으로만 작성하세요. 다른 어미를 사용하면 위반입니다.

| 필드 | 화법 | 종결 어미 | 예시 |
|------|------|----------|------|
| current_status | 공감형 서술체 | ~에요/~이에요 | "지난주 57% 달성했어요. 기존 습관에서 크게 줄었지만, 아직 목표에는 미달이에요." |
| final_goal | 명사형 종결 | ~먹기/~마시기/~줄이기 | "과일 먹기 (주 3회)" |
| reason | 설명형 존댓말 | ~에요/~미칩니다 혼용 | "달성률이 올라가고 있어요. 이 속도로 계속 실천해 보세요." |
| action_plans | 합쇼체 | ~합니다/~하세요 혼용 | "두부로 대체합니다." / "메뉴를 정해두세요." |

### current_status 세부 규칙
- 반드시 "~에요", "~이에요", "~있어요" 등 부드러운 서술체로 종결
- 지난주 달성률 수치를 반드시 포함
- 긍정적 변화가 있으면 인정한 뒤 남은 과제 언급
- ❌ 금지: "~있습니다", "~됩니다" (딱딱한 합쇼체), "나쁜 습관" (판단형)

### reason 세부 규칙
- 어떤 전략(유지/하향/교체)을 선택했는지 + 근거(달성률, 행동 패턴)를 설명
- "~에요"와 "~미칩니다/~줍니다"를 혼용 가능
- 2~4문장 작성
- ❌ 금지: "~해야 합니다" (당위형), "~하지 않으면 위험합니다" (위협형)

### action_plans 세부 규칙
- 반드시 3개 항목 생성
- 3개 중 최소 1개는 "~합니다" (행동 진술), 최소 1개는 "~하세요" (권유)로 종결
- 지난주 피드백의 행동 패턴(예: 주말 과식, 야식 습관)을 반영
- ❌ 금지: "~하지 마세요" (부정 명령 시작), "~하십시오" (격식체)
</ROLE>

<INPUT_DATA>
1. 궁극적 초기 목표 (최종 도착지):
{initial_goal}

2. 직전 주차 목표 (지난주 부여된 목표):
{prev_adaptive_goal}

3. 달성률 추이 및 통계:
{past_ach_str}
{stats_summary}

4. 이번 주 COM-B 행동 분석 리포트:
{feedback_report}

5. 사용 가능한 행동 변수 풀 (코드북 D1~L3):
{meta_summary}
</INPUT_DATA>

<INSTRUCTIONS>

## STEP 1: 각 목표의 달성률을 확인하고 적응 전략을 선택하세요

### 달성률 산출 기준
- **Increase 목표** ("~먹기", "~마시기", "~하기"): 해당 행동을 **수행한 날** = 달성일
- **Decrease 목표** ("~줄이기"): 해당 음식을 **섭취하지 않은 날** = 달성일
- **주간 달성률** = 달성일 수 / 7

### 적응 규칙표 (우선순위 순서대로 적용)

| 우선순위 | 조건 | 전략 | 설명 |
|---------|------|------|------|
| 1 | 달성률 ≤ 1/7 (15% 이하) **AND** 전주도 ≤ 1/7 | **교체** | 2주 연속 사실상 비실행 → 코드북 내 다른 변수로 교체 |
| 2 | 달성률 ≤ 3/7 (50% 이하) | **하향** | 목표 행동은 유지, 난이도(횟수)만 조절 |
| 3 | 달성률 4/7 ~ 5/7 (50% 초과 ~ 75% 이하) | **유지** | 직전 목표 횟수 그대로 유지 |
| 4 | 달성률 ≥ 6/7 (75% 초과) **AND** 전주도 ≥ 6/7 | **유지 (습관 강화)** | 목표 횟수 유지 + 습관 강화 피드백 제공 |

- 위 표에 해당하지 않는 경우(예: 달성률 ≥ 6/7이지만 전주는 아닌 경우)는 **유지** 전략을 적용합니다.
- **상향 전략은 적용하지 않습니다**: 본 연구의 목표(final_goal)는 각 항목의 최종 권장 수준으로 설정되어 있어 목표 상한(ceiling)이 존재하므로, 고달성 시 상향이 아닌 습관 강화를 유도합니다.

### 전략별 세부 규칙

**교체 전략**:
- 반드시 코드북(D1~D19, L1~L3)에 존재하는 변수만 선택하세요. 코드북에 없는 임의 행동(산책, 명상 등)은 절대 불가합니다.
- SHAP 위험요인 순위에서 현재 3개 목표 다음 순위(4순위) 항목을 대체 후보로 선정합니다.
- 교체은 영구적 포기가 아닌 전술적 우회입니다. 우회 목표에서 높은 달성률을 보이면 이후 주차에 원래 목표로 복귀할 수 있습니다.
- 과거 달성률이 1주 치만 있으면 "2주 연속" 조건 미충족이므로 교체 사용 금지 → 하향을 쓰세요.

**하향 전략**:
- Increase 목표(예: 운동 늘리기): 주간 목표 횟수를 감소 (예: 주 7회 → 주 4회)
- Decrease 목표(예: 단 음식 줄이기): 주간 허용 횟수를 증가 (예: 주 0회 → 주 2회 이하). 건강 악화가 아닌 현실적 타협입니다.
- 너무 쉽거나 너무 어렵지 않도록 적절히 조절하세요.

**유지 (습관 강화) 전략**:
- 목표 횟수는 변경하지 않습니다.
- 2주 연속 75% 초과 달성 중이므로, 형성 중인 습관을 공고히 하는 방향의 피드백을 제공합니다.
- current_status와 reason에 습관 형성 진행 상황을 언급하세요.

{guardrail_text}

---

## STEP 2: final_goal 문구 작성

모든 목표 문구는 아래 통일 포맷을 따릅니다:

### 포맷 규칙
```
Increase 목표: [행동] (주 X회)
Decrease 목표: [대상] 줄이기 (주 X회 이하) 또는 (주 0회)
```

### Increase 예시
| final_goal | direction |
|------------|-----------|
| 고기·생선·두부·달걀 반찬 하루 1끼 이상 먹기 (주 7회) | Increase |
| 김치 제외 채소반찬 하루 2끼 이상 먹기 (주 14회) | Increase |
| 과일 먹기 (주 7회) | Increase |
| 물 하루 6컵 이상 마시기 (주 7회) | Increase |
| 중강도 이상 운동 하기 (주 5회 이상, 1회 30분 이상) | Increase |

### Decrease 예시
| final_goal | direction |
|------------|-----------|
| 튀김·부침개 줄이기 (주 2회 이하) | Decrease |
| 삼겹살·갈비 등 기름진 고기 줄이기 (주 2회 이하) | Decrease |
| 라면·즉석식품 등 가공식품 줄이기 (주 2회 이하) | Decrease |
| 소금·간장 추가 사용 줄이기 (주 0회) | Decrease |
| 케이크·과자·아이스크림 등 단 음식 줄이기 (주 0회) | Decrease |
| 음주 줄이기 (주 0회) | Decrease |

### 난이도 조절 시 final_goal 예시
- 하향: "과일 먹기 (주 7회)" → "과일 먹기 (주 3회)"
- 하향: "단 음식 줄이기 (주 0회)" → "단 음식 줄이기 (주 2회 이하)"

### 금지 표현
- ❌ "~섭취", "~실천하기", "~제한하기" → 딱딱함
- ❌ "~먹지 않기 (주 0회)" → 중복 ("않기"에 이미 0회 의미 포함)
- ✅ "~먹기", "~마시기", "~줄이기" → 쉬운 동사 사용

---

## STEP 3: daily_goal 계산

- Increase: ceil(주간 목표 횟수 ÷ 7) → "X회"
  - 예: 주 7회 → "1회", 주 14회 → "2회", 주 3회 → "1회"
- Decrease: 항상 "0회"

---

## STEP 4: current_status 작성

참여자에게 직접 보여주는 텍스트입니다.
반드시 **공감형 서술체(~에요/~이에요)**로 작성하세요.
지난주 달성률과 행동 패턴을 반영하여 작성하세요.

예시:
- "지난주 57% 달성했어요. 기존 습관(하루 2회 이상)에서 크게 줄었지만, 아직 목표에는 미달이에요."
- "지난주 채소반찬을 주 10회 드셔서 목표(주 14회)에 점차 가까워지고 있어요."
- "지난주 튀김·부침개를 주 4회 드셨어요. 목표(주 2회 이하)보다 조금 많은 편이에요."

⚠️ 어투 주의:
- ✅ "~달성했어요", "~가까워지고 있어요", "~미달이에요"
- ❌ "~드셨습니다", "~초과하고 있습니다" (합쇼체 금지)

---

## STEP 5: reason 작성

이번 주 이 전략을 선택한 이유를 2~4문장으로 설명하세요.
**설명형 존댓말(~에요/~미칩니다 혼용)**을 사용합니다.
어떤 전략(유지/하향/교체/강화)을 선택했는지, 근거(달성률, 행동 패턴)는 무엇인지 명시하세요.

예시:
- "지난주 달성률이 30%로 목표 대비 부족했어요. 난이도를 하향하여 주 0회에서 주 2회 이하로 완화하면, 작은 성공 경험을 쌓는 데 도움을 줍니다."
- "2주 연속 달성률 14% 이하로, 직접적인 행동 변화가 어려운 상황이에요. 코드북 내 D11(물 섭취)로 교체하여 건강 습관 기반을 먼저 다지는 것이 효과적입니다."
- "57% 달성률은 기존 습관 대비 큰 개선이에요. 특히 주말에 섭취가 늘어나는 패턴이 있으니, 주말 식단 계획에 신경 써주세요."
- "2주 연속 85% 이상 달성하셨어요. 이제 이 행동이 습관으로 자리잡고 있어요. 현재 목표를 유지하면서 꾸준히 실천하시면 좋겠습니다."

⚠️ 어투 주의:
- ✅ "~부족했어요", "~도움을 줍니다", "~효과적입니다" 혼용 가능
- ❌ "~해야 합니다" (당위형), "~하지 않으면 위험합니다" (위협형)

---

## STEP 6: action_plans 작성

3개의 구체적 행동 계획. "무엇을 + 어떻게/언제"가 드러나야 합니다.
반드시 **합쇼체(~합니다/~하세요 혼용)**로 작성하세요.
지난주 피드백에서 나온 행동 패턴(예: 주말 과식, 야식 습관)을 반영하여 작성하세요.

### 어투 혼용 규칙
- 3개 항목 중 최소 1개는 "~합니다" (행동 진술), 최소 1개는 "~하세요" (권유)

| 패턴 | 종결 어미 | 예시 |
|------|----------|------|
| 대안 제시 | ~합니다 | "삼겹살, 갈비 등 대신 닭가슴살, 생선, 두부로 대체합니다." |
| 상황별 권유 | ~하세요 | "주말 외식 시 미리 저지방 메뉴를 정해두세요." |
| 조리법 제안 | ~합니다 | "육류 섭취 시 삶거나 찌는 조리법을 선택합니다." |
| 습관 형성 | ~하세요 | "물을 마시기 쉬운 곳에 두고 습관적으로 마시세요." |
| 보충 제안 | ~좋습니다 | "수분 섭취를 돕기 위해 보리차나 허브차를 마시는 것도 좋습니다." |

- ❌ "건강한 음식을 먹습니다." (추상적)
- ❌ "기름진 고기를 먹지 마세요." (부정 명령 시작)

---

## STEP 7: strategy & substituted_from 메타데이터

- `strategy`: 선택한 전략을 기입
  - `"유지"` / `"하향"` / `"교체"` / `"강화"` 중 하나
  - 상향 전략은 본 연구에서 사용하지 않습니다.
- `substituted_from`:
  - `"교체"`인 경우에만 → 직전 주 실패한 기존 목표의 문항 ID (예: "D16")
  - 나머지 전략 → 반드시 `null`

</INSTRUCTIONS>

<OUTPUT_FORMAT>
반드시 제공된 JSON Schema(GoalList) 구조에 맞추어 유효한 JSON만 출력하세요.
마크다운 백틱이나 설명 텍스트 없이 순수한 JSON 문자열만 출력합니다.
</OUTPUT_FORMAT>
    """

    if reflection_msg:
        generation_prompt += f"""
        \n----------------------------------------------------------------
        🚨 [중요: 검증 실패에 따른 재작성 요청] 🚨
        이전 생성 결과가 다음 이유로 거절되었습니다: 
        {reflection_msg}
        
        추가로 요청된 데이터 도구 분석 결과가 있다면 위 INPUT_DATA의 3번 항목에 추가되었습니다.
        위 피드백과 추가 데이터를 반영하여 목표를 반드시 수정하십시오.
        ----------------------------------------------------------------
        """

    try:
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=GoalList, 
        )
        
        response = model.generate_content(
            generation_prompt,
            generation_config=generation_config
        )

        # 배너 알림용 동적 조정 여부 및 문구 생성 로직
        banner_info = {
            "is_adjusted": False,
            "title": "목표 점검 완료",
            "desc": "이번 주도 현재 목표를 잘 유지해봐요!"
        }
        
        try:
            goals_data = json.loads(response.text)
            adjusted_strategies = []
            
            for g in goals_data.get('goals', []):
                st = g.get('strategy', '')
                if st in ['교체', '하향', '유지', '강화']:
                    adjusted_strategies.append(st)
                    
            if adjusted_strategies:
                banner_info["is_adjusted"] = True
                banner_info["title"] = "새로운 맞춤형 목표 도착 🎯"
                
                # 적용된 전략에 따라 디테일한 알림 문구(desc) 분기 처리
                if "교체" in adjusted_strategies:
                    banner_info["desc"] = "🔄 실천에 더 적합한 새로운 행동으로 목표가 교체되었어요!"
                elif "하향" in adjusted_strategies:
                    banner_info["desc"] = "❤️ 부담을 줄여 목표를 조금 낮췄어요."
                elif "강화" in adjusted_strategies:
                    banner_info["desc"] = "💪 훌륭해요! 목표를 꾸준히 달성하여 습관 강화 단계로 넘어갑니다."
                else:
                    banner_info["desc"] = "지난주 달성률을 반영하여 목표가 새롭게 조정되었습니다."
                    
        except json.JSONDecodeError:
            pass # JSON 파싱 실패 시 기본값(False) 유지 (Reflection Node에서 처리됨)

        return {
            'adaptive_goal': response.text,
            'goal_adjustment_banner': banner_info # State에 배너 데이터 추가
        }
        
    except Exception as e:
        print(f"❌ Adaptive Goal API 호출 중 오류 발생: {e}")
        retry_count = state.get('retry_count', 0) + 1
        return {
            'adaptive_goal': json.dumps({"goals": []}),
            'adaptive_goal_feedback': f"API 호출 실패 (retry {retry_count}/{MAX_RETRY})",
            'retry_count': retry_count
        }

def adaptive_goal_reflection_node(state: PatientState):
    
    generated_goal_str = state.get('adaptive_goal', '{}')
    disease_data = state.get('disease_data', {})

    # 1. 하드코딩된 Python Rule Check (1차 방어선)
    try:
        goals_data = json.loads(generated_goal_str)
        goals_list = goals_data.get('goals', [])
        
        is_valid, rule_feedback = validate_goals(goals_list, disease_data)
        if not is_valid:
            print(f"[Auditor] 🔴 기계적 가드레일 위반 감지: {rule_feedback}")
            return {'adaptive_goal_feedback': rule_feedback} 
            
    except json.JSONDecodeError:
        return {'adaptive_goal_feedback': '생성된 결과가 유효한 JSON 포맷이 아닙니다.'}
    
    # 2. LLM 기반 의미론적(Semantic) 검사 (2차 방어선)
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(API_MODEL)

    achievements = state.get('accumulated_achievements', [])    
    current_week_achievements = [ach for ach in achievements if ach.get('week') == state.get('week')]

    prompt = f"""
<ROLE>
당신은 AI가 생성한 적응형 건강 목표의 논리적 타당성과 포맷 준수 여부를 검사하는 감사관입니다.
아래 검사 기준을 하나라도 위반하면 구체적인 피드백을 출력하세요.
모든 기준을 통과했다면 오직 'PASS'라고만 출력하세요.
</ROLE>

<CHECKLIST>

1. **적응 전략 타당성**: 달성률에 따라 올바른 전략이 선택되었는가?
   - 달성률 ≤ 1/7 (15% 이하)이 2주 연속: 교체. 단, 과거 데이터 1주 치뿐이면 교체 사용 불가 → 하향 사용
   - 달성률 ≤ 3/7 (50% 이하): 하향
   - 달성률 4/7 ~ 5/7 (50% 초과 ~ 75% 이하): 유지
   - 달성률 ≥ 6/7 (75% 초과)이 2주 연속: 유지 (강화)
   - 그 외: 유지
   - ❗ 상향 전략은 본 연구에서 사용하지 않음. 상향이 적용되었다면 위반

2. **상향 금지 확인**: strategy가 "상향"으로 설정되지 않았는가? 상향이 사용된 경우 위반이다.

3. **하향 방향 일관성**:
   - Increase 목표 하향 → 주간 횟수 감소 (맞는가?)
   - Decrease 목표 하향 → 주간 허용 횟수 증가 (맞는가?)

4. **코드북 일치성**: 모든 `related_id`가 코드북(D1~D19, L1~L3)에 존재하는가?

5. **final_goal 포맷 준수**:
   - Increase: "[행동] (주 X회)" 형태인가?
   - Decrease: "[대상] 줄이기 (주 X회 이하)" 또는 "(주 0회)" 형태인가?
   - "~섭취", "~실천하기", "~제한하기" 같은 금지 표현이 없는가?
   - "~먹지 않기 (주 0회)" 같은 중복 표현이 없는가?

6. **daily_goal 정합성**:
   - Increase: ceil(주간 횟수 ÷ 7) 값과 일치하는가?
   - Decrease: "0회"인가?

7. **direction 논리 일관성**: Increase인데 "줄이기"이거나, Decrease인데 "먹기"인 경우 위반

8. **strategy & substituted_from 정합성**:
   - "교체"일 때만 substituted_from에 값이 있는가?
   - "유지"/"하향"/"강화"일 때 substituted_from이 null인가?
   - strategy가 "상향"이면 위반 (본 연구에서 상향은 사용하지 않음)

9. **reason 구체성**: 달성률 수치와 선택한 전략명이 reason에 포함되어 있는가?

10. **current_status 어투 검사**: `current_status`가 공감형 서술체(~에요/~이에요/~있어요)로 종결되는지 확인한다. "~있습니다", "~됩니다" 등 합쇼체로 종결된 경우 위반이다.

11. **reason 어투 검사**: `reason`이 설명형 존댓말(~에요/~미칩니다 혼용)로 작성되었는지 확인한다. "~해야 합니다"(당위형), "~하지 않으면 위험합니다"(위협형)가 포함된 경우 위반이다.

12. **action_plans 어투 검사**: `action_plans` 3개 항목 중 최소 1개는 "~합니다"(행동 진술), 최소 1개는 "~하세요"(권유)로 종결되어야 한다. 모두 같은 어미로 끝나거나, "~하지 마세요"(부정 명령 시작), "~하십시오"(격식체)가 사용된 경우 위반이다.

</CHECKLIST>

<TOOL_REQUEST_GUIDE>
통계적 근거가 부족하여 재작성이 필요한 경우, 피드백에 아래 명령어를 포함하세요:
- 장기 추세: `REQUEST_TREND: [문항ID]`
- 변수 간 상관관계: `REQUEST_SPEARMAN: [문항ID1, 문항ID2]`
</TOOL_REQUEST_GUIDE>

<CONTEXT>
- 초기 목표: {state.get('initial_goal')}
- 직전 주차 달성률: {current_week_achievements}
- 피드백 리포트: {state.get('feedback_report')}
- 통계 요약: {state.get('statistical_summary')}
</CONTEXT>

<TARGET>
{state.get('adaptive_goal')}
</TARGET>
    """

    try:
        response = model.generate_content(prompt)
        feedback = response.text.strip()
        
        if feedback.upper() == "PASS" or ("PASS" in feedback.upper() and len(feedback) < 10):
            # print("[Auditor] 🟢 적응형 목표 검증 통과!")
            goal_str = state.get('adaptive_goal', '{}')
            try:
                goal_json = json.loads(goal_str)
                goal_json['week'] = state.get('week', 1)
                
                accumulated = state.get('accumulated_weekly_goal', [])
                new_accumulated = accumulated.copy() if accumulated else []
                new_accumulated.append(goal_json)
                
                return {
                    'adaptive_goal_feedback': 'PASS',
                    'accumulated_weekly_goal': new_accumulated # State 업데이트
                }
            except json.JSONDecodeError:
                return {'adaptive_goal_feedback': 'PASS'}
            
        else:
            # print(f"[Auditor] 🔴 적응형 목표 보완 요청 발생!")
            # if "REQUEST_" in feedback:
            #     print("         (추가 데이터 도구 분석을 추천했습니다.)")
            return {'adaptive_goal_feedback': feedback}
            
    except Exception as e:
        print(f"❌ Reflection API 호출 중 오류 발생: {e}")
        return {'adaptive_goal_feedback': f"REFLECTION_ERROR: {e}"}
    
def adaptive_goal_router(state: PatientState):
    retry_count = state.get('retry_count', 0)
    
    # 최대 재시도 횟수 초과 시 강제 종료
    if retry_count >= MAX_RETRY:
        print(f"\n[Adaptive Router] 최대 재시도 횟수({MAX_RETRY}) 초과. 현재 결과로 종료합니다.")
        return "end"
    
    feedback = state.get('adaptive_goal_feedback', '')
    
    if 'PASS' in feedback.upper(): 
        return "end"
    else:
        print(f"\n[Adaptive Router] Retry ({retry_count + 1}/{MAX_RETRY}): {feedback[:200]}...")
        return "retry"