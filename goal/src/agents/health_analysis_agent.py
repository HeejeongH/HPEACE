import os
import json
import joblib

import pandas as pd
import numpy as np

import torch

from agents.state import PatientState

from lib.config import checkpoint_path, artifact_path
from lib.config import target_goal_cols
from lib.model import AutoEncoder as AE
from lib.utils import numpy_converter


def health_analysis_node(state: PatientState):

    # ============================================================
    # 1. Get latent vector via pretrained auto-encoder model
    # ============================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    diet_data = state['diet_data']
    X_raw = np.array(list(diet_data.values())).reshape(1, -1)

    scaler = checkpoint['scaler']
    X_scaled = scaler.transform(X_raw)
    
    model = AE(input_dim=config['input_dim']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        full_latent, *others = model(torch.tensor(X_scaled, dtype=torch.float32).to(device))
        X_latent = full_latent.cpu().numpy()

    # ============================================================
    # 2. Clustering via pretrained k-means model
    # ============================================================
    km = checkpoint['km']
    cluster_id = int(km.predict(X_latent)[0]) + 1 # 클러스터 인덱스 시작을 1로 설정!!

    # ============================================================
    # 3. Compute SHAP value via pretrained Logistic Regression model
    # ============================================================
    artifacts = joblib.load(artifact_path)

    shap_model = artifacts['model']
    explainer = artifacts['explainer']
    train_columns = artifacts['train_columns']
    df_cols = artifacts['input_features']
    dtypes_map = artifacts['dtypes_map']
    
    life_data = state['life_data']
    conf_data = state['conf_data']
    df_list = list(conf_data.values()) + list(life_data.values()) + [cluster_id]
    
    df = pd.DataFrame([df_list], columns=df_cols)    
    for col, dtype in dtypes_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)    

    df_encoded = pd.get_dummies(df)  # One-Hot Encoding
    df_aligned = df_encoded.reindex(columns=train_columns, fill_value=0)  # Reindexing
    pred_prob = shap_model.predict_proba(df_aligned)[:, 1][0]  # Logistic Regression 예측
    shap_values = explainer.shap_values(df_aligned)

    if isinstance(shap_values, list):
        # [0]: Negative Class SHAP, [1]: Positive Class SHAP
        shap_values = shap_values[1]
    
    # 차원 축소 (1, n_features) -> (n_features,)
    if len(shap_values.shape) == 2:
        shap_values = shap_values[0]
    
    # SHAP 값을 Series로 변환
    shap_series = pd.Series(shap_values, index=train_columns)

    # ============================================================
    # 4. 개인별 위험 요인 순위화 
    # ============================================================
    r_id = state['r_id']

    ind_shap_result = {}
    
    # 4. 각 컬럼별로 순회하며 매칭되는 SHAP 열 찾기
    for col in target_goal_cols:
        if col == 'Cluster' :
            original_value = cluster_id
        else:
            original_value = life_data[col]
            
        # 데이터에 따라 값이 정수형일 수도, 문자열일 수도 있으므로 문자열로 변환하여 조합
        matching_col_name = f"{col}_{original_value}"
        
        # (3) 해당 SHAP 컬럼이 데이터에 존재하는지 확인 후 값 추출
        if matching_col_name in train_columns:
            col_idx = train_columns.index(matching_col_name)
            shap_value = shap_series.iloc[col_idx]
            ind_shap_result[col] = {
                'original_value': original_value,
                'matched_column': matching_col_name,
                'shap_value': shap_value
            }
        else:
            ind_shap_result[col] = {
                'original_value': original_value,
                'matched_column': matching_col_name,
                'shap_value': "Column Not Found" # 매칭되는 SHAP 컬럼이 없을 경우
            }
    
    sorted_keys = sorted(ind_shap_result, key=lambda k: ind_shap_result[k]['shap_value'], reverse=True)
    # print(f"정렬 결과: {sorted_keys}")
    
    # 이미 잘하고 있는(값이 0이거나 양호한) 항목은 목표 후보에서 제외 필터링
    # 예: Alcohol Consumption이 0이면 목표로 선정하지 않음
    filtered_keys = []
    for k in sorted_keys:
        val = ind_shap_result[k]['original_value']
        
        # 제외 조건 (사용자 정의 필요)
        # 예: 술(Alcohol)이나 담배(Smoking)가 0이면 개선할 게 없으므로 제외
        if k in ['Alcohol Consumption', 'Current smoking'] and val == 0:
            continue
        
        # 예: 운동(Physical activity)이 이미 1(중강도) 이상이면 제외할지 여부 결정 (정책에 따라 다름)
        # if k == 'Physical activity' and val >= 1: continue 

        filtered_keys.append(k)
    
    # 만약 필터링 후 3개가 안 되면, 원래 리스트에서 채워넣기 (에러 방지)
    remaining = [k for k in sorted_keys if k not in filtered_keys]
    filtered_keys.extend(remaining)

    # ============================================================
    # 결과 정리
    # ============================================================

    pa_val = life_data.get('Physical activity', 0)
    smoke_val = life_data.get('Current smoking', 0)
    alc_val = life_data.get('Alcohol Consumption', 0)

    keyword_mapping = {
        'Increased waist circumference': '허리둘레',
        'Elevated blood pressure': '혈압',
        'Impaired fasting glucose': '공복혈당',
        'Elevated triglycerides': '중성지방', 
        'Decreased HDL-C': 'HDL-C',
    }
    diagnosis = {keyword_mapping.get(d, d): v for d, v in state['disease_data'].items()}
    diagnosis_value = {keyword_mapping.get(d, d): v for d, v in state['disease_value_data'].items()}

    health_analysis_results = {
        "R-ID": r_id,
        "설문 응답": {
            "식습관": diet_data.copy(),
            "생활습관": life_data.copy(),
        },
        "건강 검진": {
            "진단": diagnosis,
            "수치": diagnosis_value,
        },
        "분석 결과": {
            "식습관 클러스터": int(cluster_id),
            "raw_shap_values": ind_shap_result,
            "영역": {
                '식습관': round(int(cluster_id) / 4.0, 2),
                '신체활동': round(int(pa_val) / 3.0, 2),
                '흡연': float(smoke_val),
                '음주': round(int(alc_val) / 3.0, 2)
            },
            "위험 요인 순위": filtered_keys
        }
    }
    health_analysis_results_json = json.dumps(health_analysis_results, default=numpy_converter, ensure_ascii=False, indent=4)

    return {"health_analysis_results": health_analysis_results_json}