# guardrail.py
from typing import List, Dict, Any

# ==========================================
# 1. MetS 질환별 금기 문항 매트릭스 (Contraindication Matrix)
# ==========================================
# prohibit_more_than: True -> 해당 행동을 '증가(권장)'시키는 목표 금지
# prohibit_more_than: False -> 해당 행동을 '감소(제한)'시키는 목표 금지

METS_CONTRAINDICATIONS = {
    'Increased waist circumference': { # 복부 비만
        'D4': {'prohibit_more_than': True, 'reason': '복부비만: 식외 간식(D4) 증가 우회 금지'},
        'D13': {'prohibit_more_than': True, 'reason': '복부비만: 지방육(D13) 섭취 증가 우회 금지'},
        'D14': {'prohibit_more_than': True, 'reason': '복부비만: 인스턴트(D14) 증가 우회 금지'},
        'D17': {'prohibit_more_than': True, 'reason': '복부비만: 단 음식(D17) 증가 우회 금지'},
        'L3': {'prohibit_more_than': True, 'reason': '복부비만: 음주(L3) 증가 우회 금지'}
    },
    'Elevated blood pressure': { # 고혈압
        'D15': {'prohibit_more_than': True, 'reason': '고혈압: 소금/간장 추가(D15) 증가 우회 금지'},
        'D16': {'prohibit_more_than': True, 'reason': '고혈압: 짠 음식/국물(D16) 증가 우회 금지'}
    },
    'Impaired fasting glucose': { # 고혈당
        'D6': {'prohibit_more_than': True, 'reason': '고혈당: 정제 곡류/면류(D6) 섭취 증가 우회 금지'},
        'D17': {'prohibit_more_than': True, 'reason': '고혈당: 단 음식(D17) 섭취 증가 우회 금지'}
    },
    'Elevated triglycerides': { # 고중성지방혈증
        'L3': {'prohibit_more_than': True, 'reason': '고중성지방: 음주(L3) 증가 우회 절대 금지'},
        'D13': {'prohibit_more_than': True, 'reason': '고중성지방: 지방육(D13) 섭취 증가 우회 금지'},
        'D17': {'prohibit_more_than': True, 'reason': '고중성지방: 단 음식(D17) 증가 우회 금지'}
    },
    'Decreased HDL-C': { # 낮은 HDL
        'L1': {'prohibit_more_than': False, 'reason': '낮은 HDL: 신체 활동(L1)을 감소시키는 우회 금지'}
    }
}

# ==========================================
# 2. 사전 주입용 프롬프트 생성기 (Pre-generation)
# ==========================================
def get_guardrail_prompt(disease_data: Dict[str, int]) -> str:
    """환자의 질환 유무(1: 보유, 0: 정상)를 바탕으로 엄격한 제약 조건을 텍스트로 반환"""
    warnings = []
    
    for disease_key, is_active in disease_data.items():
        # 값이 1(True)이고 매트릭스에 정의된 질환인 경우
        if is_active == 1 and disease_key in METS_CONTRAINDICATIONS:
            rules = METS_CONTRAINDICATIONS[disease_key]
            for var_id, rule in rules.items():
                direction = "증가/권장(is_more_than: true)" if rule['prohibit_more_than'] else "감소/제한(is_more_than: false)"
                warnings.append(f"- [{disease_key}] 환자 주의: 문항 ID '{var_id}'를 {direction}하는 우회 전략 절대 금지. ({rule['reason']})")
                
    if not warnings:
        return ""
        
    prompt = "\n🚨 [임상적 절대 금기 사항 (Rule-Based Guardrail)] 🚨\n"
    prompt += "이 사용자는 대사증후군(MetS) 위험 요인을 가지고 있습니다. 교체 전략을 사용할 때 아래의 타협은 의학적으로 매우 위험하므로 시스템이 절대 금지합니다:\n"
    prompt += "\n".join(warnings)
    prompt += "\n\n💡 (허용 예시: 음주(L3) 제한 목표가 힘들 때, 음주를 유지하되 수분 섭취(D11)를 늘리거나 채소 섭취(D8)를 늘리는 식의 건강한 우회 변수를 찾으세요.)\n"
    
    return prompt

# ==========================================
# 3. 사후 검증기 (Post-generation Validator)
# ==========================================
def validate_goals(goals: List[Dict[str, Any]], disease_data: Dict[str, int]) -> tuple[bool, str]:
    """LLM이 생성한 JSON 목표를 Python 코드로 기계적 검증"""
    
    VALID_STRATEGIES = {'초기설정', '유지', '강화', '하향', '교체'}
    
    # --- 검증 1: 전략 라벨 유효성 ---
    for goal in goals:
        strategy = goal.get('strategy', '')
        if strategy not in VALID_STRATEGIES:
            return False, f"[Rule 위반 감지] 유효하지 않은 strategy '{strategy}'. 허용값: {VALID_STRATEGIES}"
    
    # --- 검증 2: substituted_from 정합성 ---
    for goal in goals:
        strategy = goal.get('strategy', '')
        sub_from = goal.get('substituted_from')
        if strategy == '교체' and not sub_from:
            return False, f"[Rule 위반 감지] strategy가 '교체'인데 substituted_from이 비어 있습니다. (related_id: {goal.get('related_id')})"
        if strategy != '교체' and sub_from:
            return False, f"[Rule 위반 감지] strategy가 '{strategy}'인데 substituted_from에 값({sub_from})이 있습니다. '교체' 외에는 null이어야 합니다."
    
    # --- 검증 3: 질환별 금기 문항 방향 충돌 ---
    for disease_key, is_active in disease_data.items():
        if is_active != 1 or disease_key not in METS_CONTRAINDICATIONS:
            continue
            
        rules = METS_CONTRAINDICATIONS[disease_key]
        
        for goal in goals:
            g_id = goal.get('related_id')
            g_is_more_than = goal.get('is_more_than')
            
            if g_id in rules:
                rule = rules[g_id]
                # 금기된 방향과 LLM이 설정한 방향이 충돌하면 에러 발생
                if rule['prohibit_more_than'] == g_is_more_than:
                    error_msg = f"[Rule 위반 감지] {rule['reason']}. LLM이 {g_id}를 {'증가' if g_is_more_than else '감소'}시키는 위험한 목표를 생성했습니다."
                    return False, error_msg
    
    # --- 검증 4: 어투 기본 검사 (current_status 합쇼체 금지) ---
    forbidden_endings = ('있습니다.', '됩니다.', '입니다.', '합니다.', '됩니다.')
    for goal in goals:
        status = goal.get('current_status', '')
        if status.rstrip().endswith(forbidden_endings):
            return False, f"[어투 위반 감지] current_status가 합쇼체(~있습니다/~됩니다)로 종결됨. 공감형 서술체(~에요/~이에요)로 수정 필요. (related_id: {goal.get('related_id')})"
                    
    return True, "PASS"