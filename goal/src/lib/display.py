# display.py
import json
from agents.state import PatientState


# ---------------------------------------------------------
# 목표 탭
# ---------------------------------------------------------

def display_goal(state: PatientState):
    print("\n" + "="*50)
    print(" 🎯 목표")
    print("="*50)

    week = state.get('week', 0)
    
    if week == 1:
        print("\n[🔍 나의 건강 분석 결과]")
        print("검진 데이터와 생활습관을 분석하여,\n가상 개선이 필요한 영역을 찾았습니다.")
        
        health_analysis_results = state.get('health_analysis_results')
        health_analysis_results_json = json.loads(health_analysis_results)
        
        goals_rank = health_analysis_results_json['분석 결과']['영역']
        for i, (k, v) in enumerate(goals_rank.items()):
            print(f"({i}) {k}: {int(v*100)}%")

        print(f"\n[🎯 {week}주차 초기 정적 목표 (12주 유지)]")
        initial_goal_str = state.get('initial_goal', '{}')
        try:
            goal_json = json.loads(initial_goal_str)

            for i, goal in enumerate(goal_json['goals']):
                print(f"[목표 {i+1}]: {goal['goal_title']} ({goal['goal_measure']})")
                print(f"| {goal['goal_measure']} | {goal['weekly_target_num']} |")

                print("📌 현재 상태")
                print("측정 전")

                print("🎯 최종 목표")
                print(goal['final_goal'])

                print("🔍 왜 이 목표가 선정되었나요?")
                print(goal['reason'])

                print("🔄 조정 이력")
                print(goal['strategy'])

                print("💡 이렇게 실천해보세요!")
                for a in goal['action_plans']:
                    print(f'- {a}')

        except Exception as e:
            print(f"목표 파싱 실패: {e}")

    else:
        print("\n[🔍 나의 건강 분석 결과]")
        print("검진 데이터와 생활습관을 분석하여,\n가상 개선이 필요한 영역을 찾았습니다.")
        
        try:
            health_analysis_results = state.get('health_analysis_results')
            health_analysis_results_json = json.loads(health_analysis_results)
            
            goals_rank = health_analysis_results_json['분석 결과']['영역']
            for i, (k, v) in enumerate(goals_rank.items()):
                print(f"({i}) {k}: {int(v*100)}%")
        except:
            pass

        # 1. 2주 차 이상은 adaptive_goal을 우선적으로 불러옵니다. (중재군 A는 initial_goal)
        goal_str = state.get('adaptive_goal')
        if not goal_str:
            goal_str = state.get('initial_goal', '{}')

        try:
            goal_json = json.loads(goal_str)
            achievements = state.get('accumulated_achievements', [])    
            
            # 2. 현재 주차(week)가 아닌, '지난주(week-1)'를 기준으로 조회합니다.
            prev_week = week - 1

            for i, goal in enumerate(goal_json.get('goals', [])):
                print(f"[목표 {i+1}]: {goal.get('goal_title', '')} ({goal.get('goal_measure', '')})")
                print(f"| {goal.get('goal_measure', '')} | {goal.get('weekly_target_num', '')} |")

                print("📌 현재 상태")
                
                # 3. 리스트 인덱스[i]에 의존하지 않고, ID(예: D11, L1)를 기반으로 안전하게 매칭
                prev_rate = None
                for ach in achievements:
                    if ach.get('week') == prev_week and ach.get('id') == goal.get('related_id'):
                        prev_rate = ach.get('rate')
                        break

                if prev_rate is not None:
                    print(f"지난주 {int(prev_rate)}% 달성 -> 이번 주 진행 중")
                else:
                    # 교체 전략 등으로 인해 지난주 기록이 없는 신규 목표인 경우
                    print("새롭게 설정된 목표 (이번 주 진행 중)")

                print("🎯 최종 목표")
                print(goal.get('final_goal', ''))

                print("🔍 왜 이 목표가 선정되었나요?")
                print(goal.get('reason', ''))

                print("🔄 조정 이력")
                print(goal.get('strategy', ''))

                print("💡 이렇게 실천해보세요!")
                for a in goal.get('action_plans', []):
                    print(f'- {a}')

        except Exception as e:
            print(f"목표 파싱 실패: {e}")


# ---------------------------------------------------------
# 피드백 탭
# ---------------------------------------------------------

def display_feedback(state: PatientState):
    print("\n" + "="*50)
    print(" 💬 피드백")
    print("="*50)

    feedback_report_str = state.get('feedback_report', '{}')
    try:
        # 1. JSON 텍스트를 파이썬 딕셔너리로 변환
        feedback_report = json.loads(feedback_report_str)
        print("\n📝 행동 요약")
        print(feedback_report.get('behavior_summary', '첫 주차이므로 아직 피드백이 없어요. 이번 주부터 열심히 실천해보세요! 💪'))

        print("\n🧠 COM-B 행동 변화 분석")
        com_b = feedback_report.get('com_b_analysis', {})
        print(f"💪 능력(C):\n{com_b.get('capability', '')}\n")
        print(f"🌍 기회(O):\n{com_b.get('opportunity', '')}\n")
        print(f"❤️ 동기(M):\n{com_b.get('motivation', '')}\n")                        
        
    except Exception as e:
        print(f"피드백 파싱 실패: {e}")
        print(feedback_report_str)

    print(f"\n📊 이번 주 분석 인사이트")

    def print_ui_row(icon, text, badge, width=54):
        """한글(2칸)과 영문/숫자(1칸)의 폭을 계산하여 배지를 우측 정렬하는 헬퍼 함수"""
        display_width = sum(2 if ord(c) > 0x1100 else 1 for c in text)
        padding = max(1, width - display_width)
        print(f" {icon}  {text}" + " " * padding + f"{badge}")

    streak_data = state.get('streak_data')
    if streak_data and streak_data.get('current_streak', 0) >= 7:
        weeks = streak_data['current_streak'] // 7
        print_ui_row("🏆", f"목표 {weeks}주 연속 100% 달성! 완벽하게 실천 중!", "[ 완료 ]")


# ---------------------------------------------------------
# 변화 탭
# ---------------------------------------------------------

def get_rate_for_week(achievements, week_num, related_id):
    """특정 주차, 특정 문항의 달성률을 찾아 반환합니다."""
    for ach in achievements:
        if ach.get('week') == week_num and ach.get('id') == related_id:
            return ach.get('rate')
    return None

def get_emoji_for_goal(related_id, title=""):
    """문항 ID나 제목을 기반으로 어울리는 이모지를 반환합니다."""
    rid = str(related_id).upper()
    if "11" in rid or "물" in title: return "💧"
    if "13" in rid or "육류" in title or "고기" in title: return "🥩"
    if "8" in rid or "채소" in title: return "🥗"
    if "16" in rid or "15" in rid or "소금" in title or "짠" in title: return "🧂"
    if "L1" in rid or "운동" in title or "활동" in title: return "🏃"
    return "🎯"

def display_change(state: PatientState):
    print("\n" + "="*50)
    print(" 📈 변화")
    print("="*50)

    current_week = state.get('week', 1)
    goal_history = state.get('accumulated_weekly_goal', [])
    
    # 누적 달성률
    achievements = state.get('accumulated_achievements', [])

    if not goal_history:
        print("\n아직 기록된 목표 변화가 없습니다.")
        return

    # =========================================================
    # 1. 나의 목표 변화 (최근 4주) - 프로그레스 바 UI 형태
    # =========================================================
    print("\n[📐 나의 목표 변화]")
    
    start_week = max(1, current_week - 3)
    end_week = current_week
    weeks_to_show = list(range(start_week, end_week + 1))
    
    # 보통 목표는 3개(Slot 0, 1, 2)이므로 각 슬롯별로 추이를 추적
    max_slots = max((len(gh.get('goals', [])) for gh in goal_history), default=0)
    
    for slot_idx in range(max_slots):
        slot_records = []
        measures_seen = []
        latest_strategy = ""
        latest_change_desc = ""

        # 해당 슬롯의 최근 4주 데이터 수집
        for w in weeks_to_show:
            # 해당 주차의 목표 데이터 찾기
            w_data = next((gh for gh in goal_history if gh.get('week') == w), None)
            g_item = None
            rate = None
            
            if w_data and slot_idx < len(w_data.get('goals', [])):
                g_item = w_data['goals'][slot_idx]
                rate = get_rate_for_week(achievements, w, g_item.get('related_id'))
                
                measure = g_item.get('goal_measure', g_item.get('goal_title', ''))
                if measure not in measures_seen:
                    measures_seen.append(measure)
                    
                # 최신 주차의 전략 및 변경 사항 기록
                if w == current_week:
                    latest_strategy = g_item.get('strategy', '유지')
                    if latest_strategy not in ['유지', '초기설정']:
                        latest_change_desc = f"⬆ {w}주차: {g_item.get('final_goal')} ({latest_strategy})"
                        
            slot_records.append((w, g_item, rate))

        # 데이터가 아예 없는 슬롯은 건너뜀
        if not measures_seen:
            continue

        # 슬롯 제목 결정 (목표가 교체되었다면 "XX 관련 목표 변화"로 표시)
        latest_goal = next((r[1] for r in reversed(slot_records) if r[1]), {})
        icon = get_emoji_for_goal(latest_goal.get('related_id'), measures_seen[-1])
        
        if len(measures_seen) > 1:
            title = f"{icon} {measures_seen[-1]} 관련 목표 변화"
        else:
            title = f"{icon} {measures_seen[0]}"

        print(f"\n{title}")

        # 박스 UI 및 라벨 UI 그리기
        boxes = []
        labels = []
        for w, g_item, rate in slot_records:
            label = f"{w}주".center(10)
            
            if g_item is None:
                # 목표가 없던 빈 주차
                boxes.append(" [  --  ] ".center(12))
            else:
                if rate is not None:
                    r_str = f"{int(rate)}%"
                    # 3개 이상 목표 교체 시 짧은 이름 추가 표시
                    if len(measures_seen) > 1:
                        short_name = g_item.get('goal_measure', '')[:2]
                        box_content = f"{short_name} {r_str}"
                    else:
                        box_content = f"{r_str}"
                    boxes.append(f" [{box_content:^6}] ".center(12))
                else:
                    # 이번 주차 (아직 달성률 미계산)
                    boxes.append(" [ 진행중 ] ".center(12))
            
            labels.append(label)

        print("".join(boxes))
        print("".join(labels))
        
        # 하단에 목표 변경 사항 하이라이트 표시
        if latest_change_desc:
            print(f"  {latest_change_desc}")


    # =========================================================
    # 2. 목표 조정 타임라인 (전체 주차 역순)
    # =========================================================
    print("\n\n[🔄 목표 조정 타임라인]")
    
    for idx, week_data in enumerate(reversed(goal_history)):
        w = week_data.get('week', 1)
        goals = week_data.get('goals', [])
        
        # 1) 타임라인 노드 제목 및 뱃지 요약
        strategies = [g.get('strategy', '유지') for g in goals]
        unique_strats = set(strategies) - {'유지', '초기설정'}
        
        if w == current_week:
            node_icon = "🟢"
            strat_txt = f" - 목표 {' + '.join(unique_strats)}" if unique_strats else " - 목표 유지 중"
            print(f"{node_icon} {w}주차 (현재){strat_txt}")
        elif w == 1:
            node_icon = "🟣"
            print(f"{node_icon} {w}주차 - 맞춤형 초기 목표 설정")
        else:
            node_icon = "🔵"
            # 해당 주차 달성률이 모두 100%인지 확인
            w_rates = [ach.get('rate', 0) for ach in achievements if ach.get('week') == w]
            if w_rates and all(r >= 100.0 for r in w_rates):
                print(f"{node_icon} {w}주차 - 전 목표 100% 달성!")
            else:
                print(f"{node_icon} {w}주차 - 주간 결과 반영")

        # 2) 해당 주차의 세부 목표 리스트 출력
        for g in goals:
            strat = g.get('strategy', '유지')
            badge = ""
            if strat == '교체': badge = "[신규 교체]"
            elif strat == '초기설정': badge = ""
            else: badge = f"[{strat}]"

            measure = g.get('goal_measure', g.get('goal_title', ''))
            target = g.get('final_goal', '')
            icon = get_emoji_for_goal(g.get('related_id'), measure)
            
            print(f"    {icon} {measure}: {target}  {badge}")

        # 3) 해당 주차가 완료된 과거라면 달성 결과 칩(Chip) 출력
        if w < current_week:
            w_achievements = [ach for ach in achievements if ach.get('week') == w]
            if w_achievements:
                chip_strs = []
                for ach in w_achievements:
                    # ID를 기반으로 목표 이름 찾기
                    m_name = "목표"
                    for g in goals:
                        if g.get('related_id') == ach['id']:
                            m_name = g.get('goal_measure', m_name)
                            break
                    chip_strs.append(f"{m_name} {int(ach['rate'])}%")
                
                print(f"    {' · '.join(chip_strs)}")
        
        # 4) 타임라인 연결선 (맨 마지막 1주차 아래에는 그리지 않음)
        if idx < len(goal_history) - 1:
            print("    │")
            
    print()


# ---------------------------------------------------------
# 캘린더 탭
# ---------------------------------------------------------

def get_score_emoji(score: float) -> str:
    """달성률 점수에 따라 직관적인 색상 이모지를 반환합니다."""
    if score >= 99.9:
        return "🟢"  # 100% 완벽 달성
    elif score >= 60.0:
        return "🟡"  # 60% 이상 (양호)
    elif score >= 30.0:
        return "🟠"  # 30% 이상 (노력 필요)
    else:
        return "🔴"  # 30% 미만 (부족)
    
def display_calendar(state: PatientState):
    print("\n" + "="*50)
    print(" 📅 캘린더")
    print("="*50)

    # 1. 연속 달성 기록 (Streak)
    if state.get('streak_data') is not None:
        print("\n[🔥 연속 달성 기록]")
        print(f"현재 연속: {state['streak_data']['current_streak']}일")
        print(f"최장 기록: {state['streak_data']['max_streak']}일")

        achievements = state.get('accumulated_achievements', [])
        if achievements:
            total_achievements = [a['rate'] for a in achievements]
            if total_achievements:
                total_achievement = sum(total_achievements) / len(total_achievements)
                print(f"종합 달성률: {total_achievement:.02f}%")

    # 2. 주차별 일일 달력 (Habit Tracker 형태)
    daily_logs = state.get('daily_logs')
    if daily_logs:
        print("\n[📅 주차별 달성 캘린더]")
        print("범례: 🟢 100% | 🟡 60% 이상 | 🟠 30% 이상 | 🔴 30% 미만\n")

        # 주차(week)별로 데이터 그룹화
        weeks_data = {}
        for log in daily_logs:
            w = log['week']
            if w not in weeks_data:
                weeks_data[w] = {}
            weeks_data[w][log['day']] = log['average_score']

        days_order = ['월', '화', '수', '목', '금', '토', '일']

        # 누적된 주차별로 달력 UI 그리기
        for w in sorted(weeks_data.keys()):
            print(f" [ {w}주차 ]")
            
            # 요일 헤더 출력
            header = "  ".join(days_order)
            print(f"  {header}")

            # 요일별 이모지 출력
            emojis = []
            for d in days_order:
                # 해당 요일의 데이터가 없으면 0점 처리 (또는 빈칸 처리 가능)
                score = weeks_data[w].get(d, 0.0)
                emojis.append(get_score_emoji(score))
            
            emoji_row = " ".join(emojis)
            print(f" {emoji_row}\n")

    # (통계적) 인사이트
    def print_ui_row(icon, text, badge, width=54):
        """한글(2칸)과 영문/숫자(1칸)의 폭을 계산하여 배지를 우측 정렬하는 헬퍼 함수"""
        display_width = sum(2 if ord(c) > 0x1100 else 1 for c in text)
        padding = max(1, width - display_width)
        print(f" {icon}  {text}" + " " * padding + f"{badge}")

    weekly_insights = state.get('weekly_insights', [])
    if weekly_insights:
        for insight in weekly_insights:
            if isinstance(insight, dict):
                i_type = insight.get('type')

                # (1) 주간 추세선 (Trend)
                if i_type == 'trend':
                    icon = "📈" if insight.get('slope', 0) > 0 else "📉"
                    # 개선/주의 배지를 달기 위해 텍스트 구성
                    trend_val = insight.get('trend', '추세')
                    badge = f"[ {trend_val} ]"
                    text = f"{insight.get('name')}: 주 후반으로 갈수록 {trend_val}하는 추세"
                    print_ui_row(icon, text, badge)

                # (2) 평일 vs 주말 차이 (T-test)
                elif i_type == 'ttest':
                    icon = "🆕"
                    higher = insight.get('higher', '')
                    badge = f"[ {higher} ↑ ]"
                    text = f"{insight.get('name')}: {higher}에 유의미하게 높음 (p={insight.get('p_value', 0):.3f})"
                    print_ui_row(icon, text, badge)

                # (3) 장기 행동 추세 (Long Term Trend)
                elif i_type == 'long_term_trend':
                    icon = "📅"
                    trend = insight.get('trend', '')
                    trend_short = trend.replace("장기적 ", "")
                    badge = f"[ 장기 {trend_short} ]"
                    text = f"{insight.get('name')}: 점차 {trend_short}되는 장기 추세"
                    print_ui_row(icon, text, badge)

                # (4) 상관관계 (Correlation)
                elif i_type == 'correlation':
                    icon = "🔗"
                    badge = "[ 연관성 ]"
                    target = insight.get('target_name', '')
                    other = insight.get('other_name', '')
                    text = f"{target} ↔ {other} 연관성 발견"
                    print_ui_row(icon, text, badge)
            else:
                # 딕셔너리가 아닌 단순 문자열일 경우 (기존 호환성)
                print(f" 💡 {insight}")
    else:
        print("  아직 감지된 통계적 패턴이 없습니다.")