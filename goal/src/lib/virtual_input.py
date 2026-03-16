import os
import numpy as np
import pandas as pd

from lib.config import diet_cols, life_cols, conf_cols, disease_cols, diseases_value_cols


def get_patient_input(target_idx: int, data_dir: str):
    '''
    실제 실증에서는 target_row 데이터가 입력으로 들어오지만, 
    여기선 기존 데이터의 한 행을 가져와서 대리로 사용.
    즉, 실제 실증에서는 삭제될 함수 
    '''
    total_df = pd.read_excel(os.path.join(data_dir, "total_only_raw.xlsx"))
    if 'Sex' in total_df.columns:
        total_df['Sex'] = total_df['Sex'].map({'M': 0, 'F': 1, 'Male': 0, 'Female': 1}).fillna(0).astype(int)

    target_row = total_df.iloc[target_idx] 
    return {
        'r_id': str(target_row['R-ID']),
        'full_data': target_row[diet_cols + life_cols + conf_cols + disease_cols].astype(float).fillna(0).astype(int).to_dict(),
        'diet_data': target_row[diet_cols].astype(float).fillna(0).astype(int).to_dict(),
        'life_data': target_row[life_cols].astype(float).fillna(0).astype(int).to_dict(),
        'conf_data': target_row[conf_cols].astype(float).fillna(0).astype(int).to_dict(),
        'disease_data': target_row[disease_cols].astype(float).fillna(0).astype(int).to_dict(),
        'disease_value_data': target_row[diseases_value_cols].astype(float).fillna(0).to_dict(),
    }


def generate_data_from_multimodal_model(week_num: int) -> pd.DataFrame:
    np.random.seed(42 + week_num) # 주차별로 일관되게 약간씩 변하는 패턴 생성
    
    improvement = min((week_num - 1) / 4.0, 1.0) # 주차가 지날수록 개선(0.0 ~ 1.0)
    
    ids = [f'D{i}' for i in range(1, 20)] + [f'L{i}' for i in range(1, 4)]
    data = {'id': ids}
    days = ['월', '화', '수', '목', '금', '토', '일']
    
    for day in days:
        day_data = []
        is_weekend = 1 if day in ['토', '일'] else 0 
        
        # 임의의 논리적 데이터 생성 (예: 식사 끼니, 물 섭취 증가 / 간식 감소)
        day_data.append(np.random.choice([2, 3]) if improvement > 0.5 else np.random.choice([1, 2, 3])) # D1
        day_data.append(1) # D2
        day_data.append(np.random.choice([0, 1], p=[0.5 + improvement*0.3, 0.5 - improvement*0.3])) # D3
        day_data.append(np.random.choice([0, 1], p=[0.3 + improvement*0.5, 0.7 - improvement*0.5])) # D4
        day_data.append(1) # D5
        
        for _ in range(6, 11): day_data.append(np.random.choice([1, 2])) # D6~D10
        day_data.append(np.random.randint(int(1 + improvement*3), int(3 + improvement*4) + 1)) # D11(물)
        
        bad_prob = max(0.1, 0.6 - improvement*0.4)
        for _ in range(12, 20): day_data.append(1 if np.random.rand() < bad_prob else 0) # D12~D19
        
        day_data.append(1 if np.random.rand() < (0.2 + improvement*0.6) else 0) # L1(운동)
        day_data.append(0) # L2(담배)
        day_data.append(1 if is_weekend and np.random.rand() < bad_prob else 0) # L3(술)
        
        data[day] = day_data

    df = pd.DataFrame(data).set_index('id')
    df['합계'] = df[['월', '화', '수', '목', '금', '토', '일']].sum(axis=1)
    return df
