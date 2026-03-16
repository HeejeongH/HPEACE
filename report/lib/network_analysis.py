import os
import numpy as np
import pandas as pd

import networkx as nx
from scipy import stats
from sklearn.covariance import graphical_lasso
from sklearn.model_selection import KFold

from lib.config import FOOD_GROUPS

# ============================================================================
# GGM 네트워크 기반 동적 허브 추출 로직
# ============================================================================
def nonparanormal_skeptic_transform(X):
    n_samples, n_features = X.shape
    Z = np.zeros_like(X, dtype=float)
    for j in range(n_features):
        ranks = np.argsort(np.argsort(X[:, j])) + 1
        F_hat = (ranks - 0.5) / n_samples
        Z[:, j] = stats.norm.ppf(F_hat)
    return Z

def nonparanormal_correlation_matrix(data):
    X = data.values
    Z = nonparanormal_skeptic_transform(X)
    return np.corrcoef(Z.T)

def graphical_lasso_cv_loglik(X, corr_matrix, alphas=None, cv_folds=5):
    n_samples, n_features = X.shape    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = np.zeros(len(alphas))
    
    for alpha_idx, alpha in enumerate(alphas):
        fold_scores = []
        for train_idx, test_idx in kf.split(X):
            try:
                X_train, X_test = X[train_idx], X[test_idx]
                train_corr = np.corrcoef(X_train.T) + np.eye(n_features) * 1e-6
                _, precision_train = graphical_lasso(train_corr, alpha=alpha, max_iter=100)
                
                test_corr = np.corrcoef(X_test.T) + np.eye(n_features) * 1e-6
                sign, logdet = np.linalg.slogdet(precision_train)
                if sign > 0:
                    log_lik = logdet - np.trace(test_corr @ precision_train)
                    fold_scores.append(log_lik)
            except Exception:
                continue
                
        cv_scores[alpha_idx] = np.mean(fold_scores) if len(fold_scores) > 0 else -np.inf
    
    valid_indices = np.where(np.isfinite(cv_scores))[0]
    best_alpha = alphas[valid_indices[np.argmax(cv_scores[valid_indices])]] if len(valid_indices) > 0 else alphas[0]
            
    cov_matrix = corr_matrix.copy()
    try:
        _, best_precision = graphical_lasso(cov_matrix, alpha=best_alpha, max_iter=100)
    except:
        best_alpha = 0.01
        _, best_precision = graphical_lasso(cov_matrix, alpha=best_alpha, max_iter=100)
    
    partial_corr = np.zeros_like(best_precision)
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                partial_corr[i, j] = 1.0
            else:
                denom = np.sqrt(best_precision[i, i] * best_precision[j, j])
                if denom > 0:
                    partial_corr[i, j] = -best_precision[i, j] / denom
    return best_alpha, best_precision, partial_corr

def get_user_group_info(user_data):
    """user_data에서 성별, 연령대, 대사증후군 여부를 파악"""
    sex = 0 if user_data.get('Sex', 0) in [0, 'M', 'Male'] else 1
    age = float(user_data.get('Age', 40))
    
    if age < 40:
        age_group = '청년층(19-39세)'
    elif age < 60:
        age_group = '중년층(40-59세)'
    else:
        age_group = '장년층(60-74세)'
        
    mets = 1 if user_data.get('MetS', 0) == 1 else 0
    
    return sex, age_group, mets

def extract_top_hubs_for_user(user_data: dict, data_dir: str, top_k: int = 5):
    """사용자의 속성 집단 데이터를 기반으로 GGM 네트워크를 구축해 Top 허브 추출"""
    sex_val, age_group_val, mets_val = get_user_group_info(user_data)
    print(f"-> 추출된 사용자 그룹: 성별({sex_val}), 연령대({age_group_val}), MetS({mets_val})")
    
    # 전체 데이터 로드 후 그룹 필터링
    total_df = pd.read_excel(os.path.join(data_dir, "total_only_raw.xlsx"))
    
    # Sex 매핑 통일 (0: 남성, 1: 여성)
    if 'Sex' in total_df.columns:
        total_df['Sex'] = total_df['Sex'].map({'M': 0, 'F': 1, 'Male': 0, 'Female': 1, 0: 0, 1: 1}).fillna(0).astype(int)
        
    # Age Group 생성
    total_df['Age_Group'] = total_df['Age'].apply(lambda x: '청년층(19-39세)' if x < 40 else ('중년층(40-59세)' if x < 60 else '장년층(60-74세)'))
    
    # 그룹 필터링
    mask = (total_df['Sex'] == sex_val) & (total_df['Age_Group'] == age_group_val) & (total_df['MetS'] == mets_val)
    group_data = total_df[mask].copy()
    
    if len(group_data) < 50:
        print(f"⚠️ 해당 그룹의 샘플 수가 부족합니다 (n={len(group_data)}). 기본 허브를 반환합니다.")
        return ['Processed Foods', 'High Fat Meat', 'Salty Food Consumption']

    print(f"-> 그룹 데이터(n={len(group_data)})로 GGM 네트워크 분석 중...")
    
    X = group_data[FOOD_GROUPS].copy().fillna(group_data[FOOD_GROUPS].mean())
    X_array = X.values
    corr_matrix = nonparanormal_correlation_matrix(X)
    
    # 속도를 위해 StARS 대신 CV 방식 사용 (Alphas 범위 축소로 속도 최적화)
    alphas = np.logspace(-3.5, -1, 10) 
    _, _, partial_corr = graphical_lasso_cv_loglik(X_array, corr_matrix, alphas=alphas, cv_folds=3)
    
    G = nx.Graph()
    for food in FOOD_GROUPS:
        G.add_node(food)
        
    for i, food1 in enumerate(FOOD_GROUPS):
        for j, food2 in enumerate(FOOD_GROUPS):
            if i < j and abs(partial_corr[i, j]) >= 0.05: # min_correlation
                G.add_edge(food1, food2, weight=abs(partial_corr[i, j]))
                
    # 연결정도 중심성 계산 및 상위 3개 추출
    degree_cent = nx.degree_centrality(G)
    top_hubs = [node for node, cent in sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    
    return top_hubs

def get_network_image_path(user_data: dict, image_dir: str = "result/networks/images") -> str:
    """
    user_data를 분석하여 해당하는 네트워크 이미지 파일 경로를 반환합니다.
    반환 예시: "result/networks/images/network_Male_Young_MetS(+).png"
    """
    # 1. 성별 파악 (0, 'M', 'Male' -> 'Male' / 그 외 -> 'Female')
    sex_val = user_data.get('Sex', 0)
    sex_str = "Male" if sex_val in [0, 'M', 'Male'] else "Female"
        
    # 2. 연령대 파악 (Young: <40, Middle: 40~59, Old: 60+)
    age = float(user_data.get('Age', 40))
    if age < 40:
        age_str = "Young"
    elif age < 60:
        age_str = "Middle"
    else:
        age_str = "Old"
        
    # 3. 대사증후군(MetS) 여부 파악 (1 -> MetS(+), 0 -> MetS(-))
    mets_val = user_data.get('MetS', 0)
    mets_str = "MetS(+)" if mets_val == 1 else "MetS(-)"
    
    # 4. 파일명 조합
    filename = f"network_{sex_str}_{age_str}_{mets_str}.png"
    file_path = os.path.join(image_dir, filename)
    
    # 파일 존재 여부 확인 (예외 처리)
    if not os.path.exists(file_path):
        print(f"⚠️ [경고] 해당 그룹의 네트워크 이미지가 존재하지 않습니다: {file_path}")
        # 필요시 기본(Default) 이미지 경로를 반환하도록 수정할 수 있습니다.
        return None
        
    return file_path