# [인력양성] Agentic AI 기반 맞춤형 식생활 관리 목표 설정

## 시스템 실행 방법 (시뮬레이션 테스트)
UI 설계 전, 콘솔에서 직접 시뮬레이션을 돌려보며 주차별로 데이터가 어떻게 변하는지 확인해 보시는 것을 권장드립니다.

```Bash
# 필수 패키지 설치
pip install -r requirements.txt

# 중재군 B (적응형 목표 조정 AI) 실행
python run_multi_agents.py --group B

# 중재군 A (초기 목표 12주 고정) 실행
python run_multi_agents.py --group A
```
Tip: 콘솔에서 Enter 키를 누를 때마다 가상의 1주일이 지나가며 새로운 데이터와 AI 피드백이 생성됩니다.

## 그룹별 UI 처리 안내
실증 연구 디자인에 따라 세 그룹의 로직이 다릅니다. UI에서도 분기 처리가 필요할 것 같습니다.

- 대조군: 건강 분석 리포트 + 식습관·생활습관 모니터링
  - 리포트 및 건강 기록 화면만 렌더링. 목표·피드백 UI 없음
- 중재군 A: 대조군 + 개인 맞춤 목표 3개 (12주 고정) + 목표 달성률 기록
  - 1주 차에 설정된 목표가 12주 동안 변하지 않음. 목표 조정 UI 불필요. 달성률 수치만 렌더링 (행동 패턴 기반 피드백, COM-B 리포트 없음)
  - 목표 strategy는 `"고정"`으로 고정 표시됨
- 중재군 B: 중재군 A + 행동 패턴 기반 피드백 + 적응형 목표 조정
  - 매주 피드백 리포트 제공. 목표의 내용과 strategy가 동적으로 변함

## 시스템 구조

시스템은 크게 1주 차(초기 진단 및 정적 목표 설정)와 2주 차 이상(주간 모니터링, 피드백 및 목표 적응)으로 나뉩니다. 시뮬레이션 환경에서는 가상 데이터 생성 모듈을 통해 사용자의 입력과 주간 실천 데이터를 대리하며, 모든 상태는 agents/state.py의 PatientState 객체에 누적됩니다. 시각화는 lib/display.py를 통해 콘솔 환경에 모바일 UI/UX 형태로 출력됩니다.

### [가상 데이터 생성] 시뮬레이션 입력 모듈 (virtual_input.py)

실제 앱/웹 프론트엔드가 연동되기 전, AI 시스템의 로직을 테스트하기 위해 환자의 초기 데이터와 매주 누적되는 실천 데이터를 가상으로 생성하는 역할을 담당합니다.

- `get_patient_input()`: 실증 연구 데이터(`total_only_raw.xlsx`)에서 특정 환자(target_idx)의 건강 검진 및 문진 데이터(식습관 D1~D19, 생활습관 L1~L3, 질환 유무 등)를 불러와 초기 상태(State)를 구성합니다.

- `generate_data_from_multimodal_model()`: 2주 차부터 매주 실행되며, 월요일부터 일요일까지 7일 치의 식습관 및 생활습관 실천 데이터를 생성합니다. 주차(week_num)가 지날수록 점진적으로 실천율이 개선되는 로직(improvement 변수)이 반영되어 있어 적응형 목표 로직을 테스트하기에 적합합니다.

### [1주차] 초기 진단 및 목표 설정 화면

사용자의 건강검진 데이터와 초기 설문 결과를 바탕으로 위험도를 분석하고 초기 목표를 제안합니다.

- 🔍 나의 건강 분석 결과 (`health_analysis_results`)
    - 건강 점수화 (영역별 위험도): 식습관, 신체활동, 흡연, 음주 4가지 영역의 위험도가 0.0 ~ 1.0 사이의 정규화된 값으로 산출됩니다. (UI 출력 시 int(v*100) 처리하여 퍼센트로 렌더링)
    - 💊 대사증후군 검진 결과: 사용자가 보유한 대사증후군 위험 요인 (예: 허리둘레, 혈압 등)

- 🎯 첫 주차 목표 카드 (`initial_goal`)
    - 3개의 개인화된 목표가 JSON 형태로 제공됩니다. (아래 JSON 구조 참고)

### [2주차 ~] 주간 리포트 및 적응형 목표 화면

매주 가상 데이터(virtual_input.py)를 수집하여 달성률을 평가하고, UI 컴포넌트 형태(목표, 피드백, 변화, 캘린더 탭)로 콘솔에 렌더링합니다.

- 🎯 목표 (`display_goal`)
    - 이번 주 실천할 3개의 목표 정보(목표명, 기준, 행동 지침, 선정 이유 등)를 출력합니다.
    
    - 2주 차 이상부터는 전주(week-1)의 목표 달성률을 조회하여 "지난주 XX% 달성 -> 이번 주 진행 중"과 같이 현재 상태를 보여주며, 목표가 변경되었을 경우 이를 반영하여 출력합니다. (중재군 B는 adaptive_goal, 중재군 A는 initial_goal 활용)

- 💬 피드백 (`display_feedback`)
    - 📝 행동 요약 (`feedback_report` → `behavior_summary`)
    
    - 🧠 행동 분석 리포트 (`feedback_report` → `com_b_analysis`)
        - COM-B 모델(역량, 기회, 동기)에 기반한 AI의 주간 행동 분석 텍스트.
    
    - 📊 이번 주 분석 인사이트 (`weekly_insights`): 통계 분석 결과(weekly_insights)를 뱃지(Badge) 형태의 모바일 UI처럼 출력합니다. 추세(📈/📉), T-test(🆕 주말/평일), 장기 추세(📅), 상관관계(🔗 연관성) 등 다양한 패턴과 7주 연속 100% 달성 시 축하 뱃지(🏆)를 제공합니다.

- 📈 변화 (`display_change`)
    - 📐 나의 목표 변화: 최근 4주 동안 각 목표 슬롯의 달성률 추이를 퍼센트 수치와 함께 시각적인 프로그레스 바 형태로 보여줍니다. 목표가 상향/하향/교체된 내역도 박스 하단에 직관적으로 렌더링됩니다.
    
    - 🔄 목표 조정 타임라인: 1주 차부터 현재 주차까지 목표가 어떻게 변화해왔는지 역순 트리(Tree) UI 형태로 보여줍니다. 🟢(현재), 🔵(과거), 🟣(1주차) 아이콘과 함께 전략 뱃지([하향], [신규 교체] 등), 그리고 해당 주차의 성과 요약 칩(Chip)을 한눈에 확인할 수 있습니다.

- 📅 캘린더 (`display_calendar`)
    - 🔥 연속 달성 기록: 매일매일의 평균 점수를 바탕으로 현재 및 최장 연속 100% 달성 기록(Streak)을 보여줍니다.

    - 📅 주차별 달성 캘린더 (해빗 트래커): Github 잔디 심기처럼 요일별 달성 점수에 따라 직관적인 신호등 이모지(🟢 완벽, 🟡 양호, 🟠 노력 필요, 🔴 부족)를 캘린더 형태로 매핑하여 출력합니다.

## 핵심 UI 컴포넌트별 JSON 데이터 구조

프론트엔드 연동 시 AI가 반환하는 주요 JSON 스키마입니다. (참고: `agents/schema.py`)

### 목표 카드 (Goal Card) Data

AI가 생성하는 목표 상세 데이터입니다. 이 데이터를 바탕으로 목표 리스트 컴포넌트를 만듭니다.

```JSON
{
  "goals": [
    {
      "goal_title": "김치 제외 채소반찬 하루 2끼 이상 먹기",
      "goal_measure": "채소",
      "related_id": "D8",
      "direction": "Increase",
      "is_more_than": true,
      "weekly_target_num": 14,
      "daily_goal": "2회",
      "final_goal": "김치 제외 채소반찬 하루 2끼 이상 먹기 (주 14회)",
      "current_status": "채소 반찬을 주 3~6회 정도 드시고 있어요. 조금만 더 늘리면 좋겠어요.",
      "reason": "중성지방 수치가 180mg/dL로 기준을 초과하고 있어요. 채소 섭취를 늘리면 식이섬유 보충에 도움을 주며, 혈중 지질 관리에 긍정적인 영향을 미칩니다.",
      "action_plans": [
        "외식 시 비빔밥이나 쌈밥을 선택하세요.",
        "매 끼니 채소 반찬을 가장 먼저 먹습니다.",
        "방울토마토나 오이를 간식으로 준비해 두세요."
      ],
      "strategy": "하향",
      "substituted_from": null
    }
  ]
}
```

#### 주요 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `goal_title` | string | 목표명 (예: 채소반찬 하루 2끼 이상 먹기) |
| `goal_measure` | string | 측정 대상 한글명 (예: 채소, 튀김·부침개, 신체활동) |
| `related_id` | string | 관련 문항 ID (예: D8, L1, D16) |
| `direction` | string | `"Increase"` 또는 `"Decrease"` |
| `is_more_than` | boolean | Increase → `true`, Decrease → `false` |
| `weekly_target_num` | integer | 주간 목표 횟수 (정수) |
| `daily_goal` | string | 일일 목표 텍스트 (Increase: `"X회"`, Decrease: `"0회"`) |
| `final_goal` | string | 최종 목표 문구 (Increase: `"[행동] (주 X회)"`, Decrease: `"[대상] 줄이기 (주 X회 이하)"`) |
| `current_status` | string | 현재 상태 요약 (공감형 서술체: ~에요/~이에요) |
| `reason` | string | 목표 선정 이유 (설명형 존댓말: ~에요/~미칩니다 혼용) |
| `action_plans` | string[] | 구체적 실천 방안 3개 (합쇼체: ~합니다/~하세요 혼용) |
| `strategy` | string | 적응 전략: `"초기설정"` / `"유지"` / `"강화"` / `"하향"` / `"교체"` (중재군 A는 `"고정"`) |
| `substituted_from` | string \| null | `"교체"` 전략일 때만 기존 목표 문항 ID (그 외 `null`) |

### 주간 피드백 리포트 (Feedback Report) Data

매주 사용자의 성과를 분석하는 리포트 화면 컴포넌트용 데이터입니다. (`feedback_report` 필드에 JSON 문자열로 저장)

```JSON
{
  "behavior_summary": "이번 주 채소 섭취 목표는 80% 달성했지만, 주말에 튀김류 섭취가 급격히 증가하는 패턴을 보였습니다.",
  "com_b_analysis": {
    "capability": "채소를 챙겨 먹는 양 조절 능력은 점차 향상되고 있습니다.",
    "opportunity": "주말 가족 외식 환경이 튀김류 섭취 증가의 주된 원인으로 보입니다.",
    "motivation": "스트레스를 기름진 음식으로 보상받으려는 습관이 남아 있습니다."
  }
}
```

### 통계 및 인사이트 (Insights) Data

대시보드 상단에 알림이나 하이라이트 카드로 띄워줄 데이터입니다. (`weekly_insights` 필드에 리스트로 저장되고 `accumulated_weekly_insights`에 누적됨)

```JSON
[
  {
    "type": "ttest",
    "target_id": "D12",
    "name": "튀김·부침개",
    "higher": "주말",
    "p_value": 0.012
  },
  {
    "type": "trend",
    "target_id": "D11",
    "name": "물",
    "trend": "증가",
    "slope": 0.25
  },
  {
    "type": "correlation",
    "target_id": "D11",
    "other_id": "D8",
    "target_name": "물",
    "other_name": "채소",
    "correlation": 0.65,
    "p_value": 0.003
  },
  {
    "type": "long_term_trend",
    "target_id": "D11",
    "name": "물",
    "trend": "장기적 증가",
    "slope": 0.0312,
    "p_value": 0.021
  }
]
```

### 목표 조정 알림 (`goal_adjustment_banner`) Data

```JSON
{
  "is_adjusted": true,
  "title": "새로운 맞춤형 목표 도착 🎯",
  "desc": "❤️ 부담을 줄이기 위해 목표 난이도가 조금 낮아졌어요."
}
```

전략별 `desc` 문구:
- 교체: `"🔄 실천에 더 적합한 새로운 행동으로 목표가 교체되었어요!"`
- 하향: `"❤️ 부담을 줄여 목표를 조금 낮췄어요."`
- 강화: `"💪 훌륭해요! 목표를 꾸준히 달성하여 습관 강화 단계로 넘어갑니다."`
- 변동 없음(유지): `is_adjusted: false`, `desc: "이번 주도 현재 목표를 잘 유지해봐요!"`


## State 주요 변수 참고 (`agents/state.py`)

| 변수명 | 타입 | 설명 |
|--------|------|------|
| `health_analysis_results` | string (JSON) | Health Analysis Agent 결과 |
| `initial_goal` | string (JSON) | 초기 목표 3개 (GoalList 스키마) |
| `adaptive_goal` | string (JSON) | 적응형 목표 3개 (GoalList 스키마) |
| `accumulated_weekly_goal` | List[Dict] | 매주 목표 누적 리스트 (week 포함) |
| `accumulated_achievements` | List[Dict] | 주차별 목표 달성률 누적 기록 (`{week, id, rate}`) |
| `daily_logs` | List[Dict] | 일별 평균 달성 점수 누적 (`{week, day, average_score}`) |
| `weekly_insights` | List[Dict] | 현재 주차의 통계 인사이트 |
| `accumulated_weekly_insights` | List[Dict] | 매주 인사이트 누적 리스트 |
| `streak_data` | Dict | 연속 달성 기록 (`{current_streak, max_streak}`) |
| `feedback_report` | string (JSON) | 피드백 리포트 (FeedbackReport 스키마) |
| `accumulated_weekly_feedback_reports` | List[string] | 누적 주간 피드백 리포트 |
| `goal_adjustment_banner` | Dict | 목표 조정 알림 배너 데이터 |
| `statistical_summary` | string | 통계 분석 결과 텍스트 (LLM 입력용) |


## 참고

### API

`src/lib/config.py`에서 `GOOGLE_API_KEY` 변수에 API Token을 입력하여 사용하실 수 있습니다.

### UI/UX 예시

`목표 관리_중재군 A.html`과 `목표 관리_중재군 B.html`을 함께 보내드립니다. `run_multi_agents.py`에서 쓰이는 변수들이 어떻게 시각화되면 좋을지 **레이아웃 참고용**으로 봐주시면 됩니다.

### 데이터

- 건강 검진: 강남검진센터 연동 (허리둘레, 혈압, 공복혈당, 중성지방, HDL-C)
- 건강 문진 (식습관 설문 19문항 + 생활습관 설문): 현재 텍스트 기반 → 추후 이미지 기반 입력으로 전환 예정 (서봉원 교수님 연구실 작업 필요)
  - [Note] `data/건강 문진 및 기록.xlsx`에서 건강 기록 시트와 같이 19개 식습관 (D1~D19)과 3개의 생활 습관 (L1~L3) 데이터가 수집된다고 생각하여 구현한 시스템입니다.

### 가드레일 (`agents/guardrail.py`)

LLM이 생성한 목표에 대해 다음 두 단계 안전장치가 적용됩니다:
1. **사전 주입 (Pre-generation)**: 환자의 대사증후군 위험 요인에 따라 금기 행동을 프롬프트에 명시 (`get_guardrail_prompt`)
2. **사후 검증 (Post-generation)**: 생성된 목표를 Python 코드로 기계적 검증 (`validate_goals`)
   - 전략 라벨 유효성 (`초기설정/유지/강화/하향/교체`)
   - `substituted_from` 정합성 (교체일 때만 값 존재)
   - 질환별 금기 문항 방향 충돌 검사
   - 어투 기본 검사 (current_status 합쇼체 금지)

### 기타 문의 사항

특정 화면(예: 레이더 차트, 타임라인) 구현 시 필요한 데이터가 부족하거나 JSON 스키마 변경이 필요하다면 언제든 편하게 요청해주세요 ! 

감사합니다 :) 🙇‍♀️
