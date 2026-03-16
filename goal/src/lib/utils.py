import os
import json
import random
import numpy as np
import torch


def set_seed(s:int, verbose=True):
    random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if verbose:
        print(f"Seed is set to {s}.")

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        return loaded_data
    except FileNotFoundError:
        print(f"오류: {file_path} 파일을 찾을 수 없습니다.")
    except json.JSONDecodeError:
        print("오류: 파일 형식이 올바른 JSON이 아닙니다.")

def numpy_converter(obj):
    # 정수형 처리 (np.int_ 등 삭제된 별칭 제거)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    
    # 실수형 처리 (np.float_ 삭제됨 -> np.float64 사용)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    
    # 불리언 처리
    elif isinstance(obj, np.bool_):
        return bool(obj)
        
    # 배열(ndarray) 처리 (혹시 몰라 추가)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
        
    raise TypeError(f"Type {type(obj)} is not serializable")