import os
import argparse
import warnings

import numpy as np
import pandas as pd

from lib.config import diet_columns, confounders, diet_num_of_responses
from lib.clustering_for_report import predict_total_cluster
from lib.network_analysis import extract_top_hubs_for_user, get_network_image_path

warnings.filterwarnings('ignore')


# ============================================================================
# 유틸리티 
# ============================================================================
def make_json_serializable(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_json_serializable(i) for i in data]
    return data

def get_patient_input(target_idx: int, data_dir: str):
    total_df = pd.read_excel(os.path.join(data_dir, "total_only_raw.xlsx"))
    if 'Sex' in total_df.columns:
        total_df['Sex'] = total_df['Sex'].map({'M': 0, 'F': 1, 'Male': 0, 'Female': 1}).fillna(0).astype(int)

    target_row = total_df.iloc[target_idx] 
    return target_row[diet_columns + confounders].astype(float).fillna(0).astype(int).to_dict()

def normalize_diet(user_data: dict):
    '''레이더 차트를 위한 정규화'''
    results = {}
    for k, v in user_data.items():
        if k in diet_columns: 
            num_of_responses = diet_num_of_responses[k]
            ratio = int(v) / (int(num_of_responses) + 1e-8)
            results[k] = round(ratio, 2)
    return results

def report(args):
    print(f"사용자 {args.target_index}가 선택되었습니다.")

    # 사용자의 설문 응답 데이터를 가져옵니다
    user_data = get_patient_input(args.target_index, args.data_dir)

    # 19개 식습관 종합 평가
    print('\n[19개 식습관 종합 평가]')
    results = normalize_diet(user_data)
    print(f"- 레이더 차트를 위한 식습관 응답 데이터 정규화 결과:\n{results}\n")

    top_hubs = extract_top_hubs_for_user(user_data, args.data_dir, top_k=5)
    print(f"-> 도출된 위험 허브 식습관: {top_hubs}\n")

    image_path = get_network_image_path(user_data, image_dir=args.data_dir + "network_img/")
    if image_path:
        print(f"✅ 환자에게 보여줄 네트워크 시각화: {image_path}")

    # MetS 전체 종합 분석 및 MetS 개별 지표 상세 분석
    results, details = predict_total_cluster(user_data)
    clean_results = make_json_serializable(results)
    print(f'\n[MetS 전체 종합 분석 및 MetS 개별 지표 상세 분석]\n{clean_results}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--target_index', type=int, default=0)
    args = parser.parse_args()

    report(args=args)

