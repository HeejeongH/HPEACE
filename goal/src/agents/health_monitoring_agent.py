# health_monitoring_agent.py
import os
import copy
import json

import pandas as pd
import numpy as np
from scipy import stats

from typing import TypedDict, List, Dict, Any, Optional

from agents.state import PatientState
from lib.virtual_input import generate_data_from_multimodal_model
from lib import utils as uts
from lib.config import json_data_path


def calculate_streaks(daily_logs: List[Dict[str, Any]]) -> Dict[str, int]:
    """일일 달성 로그를 순회하며 현재 연속 달성 일수와 최장 기록을 계산합니다."""
    current_streak = 0
    max_streak = 0
    
    for log in daily_logs:
        if log.get('average_score', 0) == 100.0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
            
    return {
        "current_streak": current_streak,
        "max_streak": max_streak
    }

def calc_achievement_rate(weekly_df: pd.DataFrame, target_col: str, weekly_target_num: float, is_more_than: bool = True) -> Dict[str, Any]:
    """
    1. Total & Daily 달성률 계산 도구
    target_col(예: 'D11', 'L1')의 이번 주 데이터와 목표치를 비교하여 달성률을 반환합니다.
    """
    if target_col not in weekly_df.index:
        return {"error": f"{target_col} 데이터가 없습니다."}
    
    row = weekly_df.loc[target_col]
    days = ['월', '화', '수', '목', '금', '토', '일']
    
    daily_actuals = {day: float(row.get(day, 0)) for day in days}
    
    total_actual = sum(daily_actuals.values())
    total_goal = float(weekly_target_num)
        
    daily_rates = {}

    if is_more_than:
        if total_goal > 0:
            total_rate = min((total_actual / total_goal) * 100.0, 100.0)
        else:
            total_rate = 100.0 if total_actual >= 0 else 0.0
            
        daily_goal = total_goal / len(days)
        for day, actual in daily_actuals.items():
            if daily_goal > 0:
                daily_rates[day] = min((actual / daily_goal) * 100.0, 100.0)
            else:
                daily_rates[day] = 100.0 if actual >= 0 else 0.0

    else:
        if total_actual <= total_goal:
            total_rate = 100.0 
        else:
            if total_goal == 0:
                total_rate = 0.0
            else:
                excess_ratio = (total_actual - total_goal) / total_goal
                total_rate = max(0.0, 100.0 - (excess_ratio * 100.0))
        
        for day, actual in daily_actuals.items():
            if actual == 0:
                daily_rates[day] = 100.0
            else:
                daily_rates[day] = 0.0

    return {
        "target_col": target_col,
        "total_actual": total_actual,
        "total_goal": total_goal,
        "total_rate": round(total_rate, 2),
        "daily_actual": daily_actuals,
        "daily_rates": {k: round(v, 2) for k, v in daily_rates.items()}
    }

def calc_ttest(accumulated_data: List[pd.DataFrame], target_col: str) -> Dict[str, Any]:
    """2. T-test (평일 vs 주말) 도구 - 누적 데이터 이용"""
    if not accumulated_data:
        return {"error": "누적 데이터가 없습니다."}
    
    all_weekdays = []
    all_weekends = []
    
    for df in accumulated_data:
        if target_col in df.index:
            row = df.loc[target_col]
            all_weekdays.extend([row.get(d, 0) for d in ['월', '화', '수', '목', '금']])
            all_weekends.extend([row.get(d, 0) for d in ['토', '일']])
            
    if len(all_weekdays) < 2 or len(all_weekends) < 2:
        return {"error": "T-test를 수행하기엔 샘플 수가 부족합니다."}
        
    t_stat, p_val = stats.ttest_ind(all_weekdays, all_weekends, equal_var=False)
    
    return {
        "target": target_col,
        "weekday_mean": round(np.mean(all_weekdays), 2),
        "weekend_mean": round(np.mean(all_weekends), 2),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "is_significant": bool(p_val < 0.05)
    }

def calc_correlation(accumulated_data: List[pd.DataFrame], col_x: str, col_y: str) -> Dict[str, Any]:
    """3. Pearson/Spearman 상관계수 도구 - 누적 데이터 이용"""
    if not accumulated_data:
        return {"error": "누적 데이터가 없습니다."}
    
    x_vals, y_vals = [], []
    days = ['월', '화', '수', '목', '금', '토', '일']
    
    for df in accumulated_data:
        if col_x in df.index and col_y in df.index:
            for day in days:
                x_vals.append(df.loc[col_x, day])
                y_vals.append(df.loc[col_y, day])
                
    if len(x_vals) < 3:
        return {"error": "상관분석을 위한 샘플 수가 부족합니다."}

    if np.std(x_vals) == 0 or np.std(y_vals) == 0:
        return {"error": "변수 값의 변화(분산)가 없어 상관관계를 계산할 수 없습니다."}
        
    pearson_corr, p_p = stats.pearsonr(x_vals, y_vals)
    spearman_corr, p_s = stats.spearmanr(x_vals, y_vals)
    
    return {
        "col_x": col_x, 
        "col_y": col_y,
        "pearson": round(pearson_corr, 3), 
        "pearson_p": round(p_p, 4),
        "spearman": round(spearman_corr, 3), 
        "spearman_p": round(p_s, 4),
        "is_significant": bool(p_s < 0.05)
    }

def calc_trend(weekly_df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """4. 추세선(기울기) 계산 도구 - 이번 주 데이터 이용 (월~일 시계열)"""
    if target_col not in weekly_df.index:
        return {"error": f"{target_col} 데이터가 없습니다."}
        
    row = weekly_df.loc[target_col]
    days = ['월', '화', '수', '목', '금', '토', '일']
    y = [row.get(day, 0) for day in days]
    x = np.arange(len(y))
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    trend_type = "증가" if slope > 0.1 else ("감소" if slope < -0.1 else "유지")
    
    return {
        "target": target_col,
        "slope": round(slope, 3),
        "trend": trend_type,
        "r_squared": round(r_value**2, 3),
        "p_value": round(p_value, 4)
    }

def calc_long_term_trend(accumulated_data: List[pd.DataFrame], target_col: str) -> Dict[str, Any]:
    """[Tool 1] 장기 행동추세 회귀 분석기 (1주차~현재까지의 모든 일일 데이터 시계열 연결)"""
    if not accumulated_data:
        return {"error": "누적 데이터가 없습니다."}
    
    y = []
    days = ['월', '화', '수', '목', '금', '토', '일']
    for df in accumulated_data:
        if target_col in df.index:
            row = df.loc[target_col]
            y.extend([row.get(day, 0) for day in days])
            
    if len(y) < 7:
        return {"error": "추세를 분석하기에 데이터가 부족합니다."}

    if np.std(y) == 0:
        return {"error": "값의 변화(분산)가 없어 추세를 계산할 수 없습니다."}
        
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    trend_type = "장기적 증가" if slope > 0.05 else ("장기적 감소" if slope < -0.05 else "정체/유지")
    
    return {
        "target": target_col,
        "slope": round(slope, 4),
        "trend": trend_type,
        "p_value": round(p_value, 4),
        "is_significant": bool(p_value < 0.05)
    }

def parse_goals_from_text(goal_json_str: str) -> List[Dict[str, Any]]:
    """JSON Schema 구조화된 응답을 직접 파싱합니다."""
    parsed_goals = []
    
    if not goal_json_str:
        return parsed_goals

    try:
        data = json.loads(goal_json_str)
        goals = data.get('goals', [])
        
        for g in goals:
            parsed_goals.append({
                'id': g.get('related_id', 'UNKNOWN'),
                'weekly_target_num': g.get('weekly_target_num', 0),
                'is_more_than': g.get('is_more_than', True)
            })
            
    except json.JSONDecodeError as e:
        print(f"[System] JSON 파싱 에러 (LLM이 JSON 형식을 어김): {e}")
        
    return parsed_goals


def health_monitoring_node_for_groupA(state: PatientState):
    """(중재군 A) 주간 데이터를 수집하고 달성률을 계산하는 노드"""

    current_week = state.get('week', 2)
    data_week = current_week - 1  # 2주차에 실행되면 분석할 데이터는 '1주차'임

    # 1. 가상의 멀티모달 모델을 통해 '지난주(data_week)' 식습관/생활습관 데이터 생성
    new_weekly_data = generate_data_from_multimodal_model(data_week)
    if new_weekly_data is None:
        return {'feedback_report': '{"error": "주간 데이터가 없습니다."}'}

    state['weekly_data'] = new_weekly_data
    state['accumulated_weekly_data'].append(new_weekly_data)

    initial_goal = state.get('initial_goal', '{}')
    
    # [UI 연동] 중재군 A의 타임라인 기록을 위해 이번 주차 목표를 '고정'으로 복사 저장
    try:
        goal_json = json.loads(initial_goal)
        current_week_goal = copy.deepcopy(goal_json)
        current_week_goal['week'] = current_week
        for g in current_week_goal.get('goals', []):
            g['strategy'] = '고정'
            
        accumulated = state.get('accumulated_weekly_goal', [])
        new_accumulated = accumulated.copy() if accumulated else []
        new_accumulated.append(current_week_goal)
        state['accumulated_weekly_goal'] = new_accumulated
    except Exception:
        pass
    
    # 3. 달성률 계산 로직
    meta_path = os.path.join(json_data_path, 'week_survey_meta.json')
    html_name_dict = uts.load_json(meta_path)
    valid_vars = [idx for idx in new_weekly_data.index if idx.startswith('D') or idx.startswith('L')]
    
    parsed_goals = parse_goals_from_text(initial_goal)
    
    if parsed_goals:
        days = ['월', '화', '수', '목', '금', '토', '일']
        weekly_daily_scores = {day: 0 for day in days} 
        goal_count = len(parsed_goals)

        for g in parsed_goals:
            target_id = g['id']
            if target_id == "UNKNOWN" or target_id not in valid_vars:
                continue

            res = calc_achievement_rate(new_weekly_data, target_id, g['weekly_target_num'], g['is_more_than'])
            
            if "error" not in res:
                for day in days:
                    weekly_daily_scores[day] += res['daily_rates'].get(day, 0)
                
                current_rate = res.get('total_rate', 0)
                
                # 달성률을 저장할 때는 지난주(data_week) 기록으로 저장해야 차트가 정상적으로 그려짐
                state['accumulated_achievements'].append({
                    'week': data_week, 
                    'id': target_id,
                    'rate': current_rate
                })

        if goal_count > 0:
            for day in days:
                state['daily_logs'].append({
                    "week": data_week,
                    "day": day,
                    "average_score": weekly_daily_scores[day] / goal_count
                })

        state['streak_data'] = calculate_streaks(state['daily_logs'])

    return state

def health_monitoring_node(state: PatientState):
    """(중재군 B) 주간 데이터를 수집하고 통계분석을 하는 에이전트"""

    current_week = state.get('week', 2)
    data_week = current_week - 1  # 현재 주차보다 1주 전의 데이터를 평가

    new_weekly_data = generate_data_from_multimodal_model(data_week)
    
    if new_weekly_data is None:
        return {'feedback_report': '{"error": "주간 데이터가 없습니다."}'}

    state['weekly_data'] = new_weekly_data
    state['accumulated_weekly_data'].append(new_weekly_data)

    current_goal = state.get('adaptive_goal') if state.get('adaptive_goal') else state.get('initial_goal', '목표 없음')

    diet_var_def = uts.load_json(os.path.join(json_data_path, 'diet_variable_definition.json'))
    life_var_def = uts.load_json(os.path.join(json_data_path, 'life_variable_definition.json'))
    
    html_name_dict = {}
    for item in diet_var_def + life_var_def:
        if 'id' in item and 'html_name_ko' in item:
            html_name_dict[item['id']] = item['html_name_ko']

    valid_vars = [idx for idx in new_weekly_data.index if idx.startswith('D') or idx.startswith('L')]
    
    raw_past_achievements = state.get('accumulated_achievements', [])
    past_achievements = [p for p in raw_past_achievements if p.get('week') != data_week]

    summary_lines = []
    summary_lines.append("### [통계적 분석 결과 (Statistical Facts)] ###")
    current_week_insights = []

    parsed_goals = parse_goals_from_text(current_goal)
    
    if parsed_goals:
        days = ['월', '화', '수', '목', '금', '토', '일']
        weekly_daily_scores = {day: 0 for day in days} 
        goal_count = len(parsed_goals)

        summary_lines.append("\n[1. 설정된 목표별 달성률 (Achievement)]")
        for g in parsed_goals:
            target_id = g['id']
            if target_id == "UNKNOWN" or target_id not in valid_vars:
                continue

            res = calc_achievement_rate(new_weekly_data, target_id, g['weekly_target_num'], g['is_more_than'])
            
            if "error" not in res:
                for day in days:
                    weekly_daily_scores[day] += res['daily_rates'].get(day, 0)
                
                current_rate = res.get('total_rate', 0)
                q_short = html_name_dict.get(target_id, target_id) 
                achieve_type = "이상(More)" if g['is_more_than'] else "제한(Less/0)"
                
                # 달성률을 저장할 때는 지난주(data_week) 기록으로 저장
                state['accumulated_achievements'].append({
                    'week': data_week,
                    'id': target_id,
                    'rate': current_rate
                })

                past_rates = [p for p in past_achievements if p['id'] == target_id]
                summary_lines.append(f"- {target_id}({q_short}): 주간 달성률 {current_rate}% (유형: {achieve_type}, 주간목표: {g['weekly_target_num']} -> 실제 총합: {res.get('total_actual', 0)})")
                
                if past_rates:
                    history_str = " -> ".join([f"{p['week']}주차: {p['rate']}%" for p in past_rates])
                    summary_lines.append(f"  * 📈 [과거 달성률 추이]: {history_str} -> 이번 주({data_week}주차): {current_rate}%")
                else:
                    summary_lines.append(f"  * 🆕 (이번 주에 처음 설정된 목표입니다)")

        if goal_count > 0:
            for day in days:
                state['daily_logs'].append({
                    "week": data_week, 
                    "day": day,
                    "average_score": weekly_daily_scores[day] / goal_count
                })

        state['streak_data'] = calculate_streaks(state['daily_logs'])
    else:
        summary_lines.append("\n[1. 설정된 목표별 달성률]\n- 명시적인 수치를 파싱하지 못했습니다.")

    # --- 추세선 ---
    trends = []
    for var in valid_vars:
        res = calc_trend(new_weekly_data, var)
        if "error" not in res and res['p_value'] < 0.1:
            q = html_name_dict.get(var, var)
            trends.append(f"- {var}({q}): 주 후반으로 갈수록 섭취/행동이 '{res['trend']}'하는 경향 (기울기: {res['slope']})")
            current_week_insights.append({
                "type": "trend", "target_id": var, "name": q,
                "trend": res['trend'], "slope": res['slope']
            })

    if trends:
        summary_lines.append("\n[2. 이번 주 행동 추세 (Trend)]")
        summary_lines.extend(trends)

    # --- 평일 vs 주말 차이 ---
    ttests = []
    accumulated_data = state['accumulated_weekly_data'].copy()
    for var in valid_vars:
        res = calc_ttest(accumulated_data, var)
        if "error" not in res and res['is_significant']:
            q = html_name_dict.get(var, var)
            higher = "평일" if res['weekday_mean'] > res['weekend_mean'] else "주말"
            ttests.append(f"- {var}({q}): {higher}에 섭취/행동량이 유의미하게 높음 (평일 {res['weekday_mean']} vs 주말 {res['weekend_mean']}, p={res['p_value']})")
            current_week_insights.append({
                "type": "ttest", "target_id": var, "name": q,
                "higher": higher, "p_value": res['p_value']
            })

    if ttests:
        summary_lines.append("\n[3. 평일 vs 주말 행동 차이 (T-test)]")
        summary_lines.extend(ttests)

    # --- 상관관계 ---
    if parsed_goals:
        correlations = []
        for g in parsed_goals:
            target_id = g['id']
            if target_id == "UNKNOWN" or target_id not in valid_vars: continue
            target_q = html_name_dict.get(target_id, target_id) 

            for other_var in valid_vars:
                if target_id == other_var: continue 
                res = calc_correlation(accumulated_data, target_id, other_var)
                if "error" not in res and res['is_significant']:
                    other_q = html_name_dict.get(other_var, other_var) 
                    corr_val = res['spearman']
                    direction = "양(+)" if corr_val > 0 else "음(-)"
                    correlations.append(f"- [{target_id}]와 [{other_var}]: {direction}의 상관관계 (Spearman rho={corr_val}, p={res['spearman_p']})")
                    current_week_insights.append({
                        "type": "correlation", "target_id": target_id, "other_id": other_var,
                        "target_name": target_q, "other_name": other_q,
                        "correlation": corr_val, "p_value": res['spearman_p']
                    })

        if correlations:
            summary_lines.append("\n[4. 목표 변수와 다른 변수 간의 상관관계 (Spearman)]")
            summary_lines.extend(correlations)
            
    # --- 장기 추세 ---
    if current_week >= 4:
        long_term_trends = []
        for g in parsed_goals:
            target_id = g['id']
            if target_id == "UNKNOWN" or target_id not in valid_vars: continue
            res = calc_long_term_trend(accumulated_data, target_id)
            if "error" not in res and res['is_significant']:
                q = html_name_dict.get(target_id, target_id)
                # 텍스트 출력도 data_week 기준으로 표시
                long_term_trends.append(f"- {target_id}({q}): 1주차부터 {data_week}주차 현재까지 '{res['trend']}'하는 유의미한 장기 추세 확인 (p={res['p_value']})")
                current_week_insights.append({
                    "type": "long_term_trend", "target_id": target_id, "name": q,
                    "trend": res['trend'], "slope": res['slope'], "p_value": res['p_value']
                })

        if long_term_trends:
            summary_lines.append(f"\n[5. 장기 행동 추세 (1~{data_week}주차 누적)]")
            summary_lines.extend(long_term_trends)

    state['weekly_insights'] = current_week_insights
    
    if state.get('accumulated_weekly_insights') is None:
        state['accumulated_weekly_insights'] = []
    state['accumulated_weekly_insights'].append(current_week_insights)

    state['statistical_summary'] = "\n".join(summary_lines)

    return state