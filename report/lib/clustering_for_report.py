import os
import shap
import joblib
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from sklearn.linear_model import LogisticRegression

from lib.model import EmbeddingAutoEncoder as EAE
from lib.model import AutoEncoder as AE
from lib.model_mets import AutoEncoder as MAE
from lib.config import diet_columns, disease_cols, disease_short_cols, confounders

warnings.filterwarnings('ignore')


def predict_total_cluster(new_user_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {dsc: {} for dsc in disease_short_cols}
    details = {dsc: {} for dsc in disease_short_cols}
    for i, dsc in enumerate(disease_short_cols):

        checkpoint = torch.load(f'data/{dsc}/{dsc.lower()}_cluster_emb_model.pth', map_location=device, weights_only=False)

        config = checkpoint['config']
        kmeans_model = checkpoint['km']
        # print(dsc)
        if dsc == 'MetS':
            model = MAE(input_dim=config['input_dim']).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            input_df = pd.DataFrame([new_user_data])
            for col in diet_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            X_raw = input_df[diet_columns].values
            X_scaled = checkpoint['scaler'].transform(X_raw)
            
            with torch.no_grad():
                full_latent, *others = model(torch.tensor(X_scaled, dtype=torch.float32).to(device))
                latent_vec = full_latent.cpu().numpy()
                
            original_cluster_id = int(kmeans_model.predict(latent_vec)[0]) 
            final_cluster_id = original_cluster_id + 1 
            
        elif dsc == 'IWC':
            cardinalities = config.get('cardinalities')
            if cardinalities is None:
                raise ValueError("Checkpoint does not contain cardinalities for Embedding Model")
        
            model = EAE(cardinalities=cardinalities, disease_short_name=dsc).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            encoders = checkpoint['encoder']

            input_df = pd.DataFrame([new_user_data])
            for col in diet_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            X_raw_df = input_df[diet_columns].fillna(input_df[diet_columns].mode().iloc[0])

            X_encoded = X_raw_df.copy() 
            for col in diet_columns:
                le = encoders[col]
                X_encoded[col] = X_encoded[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)

            X_final = X_encoded.values
            with torch.no_grad():
                full_latent, *others = model(torch.tensor(X_final, dtype=torch.float32).to(device))
                latent_vec = full_latent.cpu().numpy()
            
            original_cluster_id = int(kmeans_model.predict(latent_vec)[0]) 
            final_cluster_id = original_cluster_id + 1 
            
        else:
            model = AE(input_dim=config['input_dim'], disease_short_name=dsc).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            input_df = pd.DataFrame([new_user_data])
            for col in diet_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            X_raw = input_df[diet_columns].values
            X_scaled = checkpoint['preprocessor'].transform(X_raw)

            with torch.no_grad():
                full_latent, *others = model(torch.tensor(X_scaled, dtype=torch.float32).to(device))
                latent_vec = full_latent.cpu().numpy()

            original_cluster_id = int(kmeans_model.predict(latent_vec)[0]) 
            final_cluster_id = original_cluster_id + 1 # 클러스터 인덱스 시작은 1!!
            
        
        aor_value = get_cluster_aor(dsc, final_cluster_id)
        pred_prob, shap_dict = predict_and_explain_lr(dsc, new_user_data, final_cluster_id)
        
        details[dsc] = {
            'disease_name': disease_cols[i],
            'final_cluster_id': final_cluster_id,
            'aor': aor_value,
            'shap': shap_dict, 
            "original_id": original_cluster_id,
            "latent_vector": latent_vec,
            "disease_prediction_probability": pred_prob,
        }

        # 정제 (aor, shap rank)
        if isinstance(aor_value, str):
            if "Reference Group" in aor_value:
                clean_aor = 1.0
            else:
                try:
                    # '1.878 (1.748-2.019) p=<0.001' -> '1.878' -> 1.878
                    clean_aor = float(aor_value.split()[0])
                except ValueError:
                    clean_aor = aor_value # 파싱 실패 시 원본 유지
        else:
            clean_aor = float(aor_value)
            
        target_shap_keys = ['Cluster', 'Physical activity', 'Alcohol Consumption', 'Current smoking']
        
        # 관심 있는 키만 추출
        filtered_shap = {k: shap_dict[k] for k in target_shap_keys if k in shap_dict}
        
        # 영향력의 크기(절댓값)가 큰 순서대로 내림차순 정렬
        sorted_shap = dict(sorted(filtered_shap.items(), key=lambda item: abs(item[1]), reverse=True))
        shap_rank = {k: f'{i+1}등' for i, k in enumerate(sorted_shap.keys())}
        
        results[dsc] = {
            'disease_name': disease_cols[i],
            'final_cluster_id': final_cluster_id,
            'aor': clean_aor,
            'shap': shap_rank, 
        }

    return results, details

def get_cluster_aor(disease_short_name, final_cluster_id):
    """저장된 엑셀 파일에서 특정 클러스터의 AOR 값을 읽어옵니다."""
    try:
        aor_df = pd.read_excel(f'data/{disease_short_name}/03_AOR_Results.xlsx', sheet_name='AOR_Analysis')
        
        target_row = aor_df[aor_df['Model'] == 'All_Confounders'].iloc[0]
        
        col_name = f"Cluster_{final_cluster_id}_AOR"
        
        if col_name in target_row:
            aor_value = target_row[col_name]
            return aor_value
        else:
            return "Reference Group (AOR=1.0)" # 1번 클러스터인 경우
            
    except Exception as e:
        return f"AOR 조회 실패: {e}"
    
def predict_and_explain_lr(disease_short_name, user_full_data, cluster_id):
    """
    저장된 LR 모델과 LinearExplainer를 불러와 질병 발병 확률과 SHAP 기여도를 계산합니다.
    """
    artifact_path = f'data/{disease_short_name}/{disease_short_name}_LR_SHAP_Artifact.pkl'
    artifacts = joblib.load(artifact_path)
    
    model_lr = artifacts['model']
    explainer_lr = artifacts['explainer']
    train_columns = artifacts['train_columns']
    input_features = artifacts['input_features'] # ['Cluster', 'Age', 'Sex', ...]
    
    user_data_with_cluster = user_full_data.copy()
    user_data_with_cluster['Cluster'] = cluster_id
    
    # 🚨 1. 필수 변수가 모두 포함되어 있는지 엄격하게 검사
    missing_cols = [col for col in input_features if col not in user_data_with_cluster]
    if missing_cols:
        raise ValueError(f"SHAP 계산을 위한 필수 교란변수가 누락되었습니다: {missing_cols}")

    input_df = pd.DataFrame([user_data_with_cluster])
    input_df = input_df[input_features].copy()
    
    if 'Sex' in input_df.columns:
        # ※ 주의: 0이 남성(M), 1이 여성(F)인 기준으로 작성되었습니다. 필요시 수정하세요.
        sex_map = {0: 'M', 1: 'F', '0': 'M', '1': 'F'}
        input_df['Sex'] = input_df['Sex'].map(lambda x: sex_map.get(x, x))
    
    categorical_confs = [
        'Sex', 'Education Level', 'Marital Status', 'Household Income', 
        'Physical activity', 'Alcohol Consumption', 'Current smoking'
    ]
    cols_to_encode = ['Cluster'] + categorical_confs
    
    for col in cols_to_encode:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
            
    X_encoded = pd.get_dummies(input_df, columns=cols_to_encode, dtype=int)
    
    for col in train_columns:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
            
    X_encoded = X_encoded[train_columns]
    
    pred_prob = float(model_lr.predict_proba(X_encoded)[0, 1])
    shap_values = explainer_lr.shap_values(X_encoded)
    
    shap_dict = {}
    row_data = X_encoded.iloc[0] # 현재 유저의 0, 1 더미 데이터 행
    
    for feat, val in zip(train_columns, shap_values[0]):
        is_categorical = False
        base_name = feat
        
        # 현재 피처가 카테고리형 변수에서 파생된 것인지 확인 (예: 'Sex_1' -> 'Sex')
        for cat_col in cols_to_encode:
            if feat.startswith(cat_col + '_'):
                is_categorical = True
                base_name = cat_col
                break
        
        if is_categorical:
            # 유저가 해당하는 카테고리(값이 1)인 경우에만 저장
            if row_data[feat] == 1:
                shap_dict[base_name] = round(float(val), 4)
        else:
            # 나이(Age)와 같은 연속형 변수는 항상 저장
            shap_dict[base_name] = round(float(val), 4)
    
    return pred_prob, shap_dict


# if __name__ == "__main__":
#     new_user_input = {
#         'Meal Frequency': 1, 
#         'Meal Portion Size': 1, 
#         'Eating Out Frequency': 5, 
#         'Rice Portion Size': 1, 
#         'Snacking Frequency': 0, 
#         'Grain Products': 1, 
#         'Protein Foods': 2, 
#         'Vegetables': 3, 
#         'Dairy Products': 1, 
#         'Fruits': 1, 
#         'Fried Foods': 0, 
#         'High Fat Meat': 1, 
#         'Processed Foods': 0, 
#         'Water Intake': 5, 
#         'Coffee Consumption': 2, 
#         'Sugar-Sweetened Beverages': 0, 
#         'Additional Salt Use': 1, 
#         'Salty Food Consumption': 1, 
#         'Sweet Food Consumption': 1,
#         'Age': 40,
#         'Sex': 'F',
#         'Education Level': 3,
#         'Marital Status': 1,
#         'Household Income': 2,
#         'Physical activity': 1,
#         'Alcohol Consumption': 0,
#         'Current smoking': 0,
#     }
#     results, details = predict_total_cluster(new_user_input)
#     print(results)
