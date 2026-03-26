# data/interfaces.py
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime

class IModelData(ABC):
    """三维模型数据契约"""
    
    @abstractmethod
    def get_3Dframe_dim(self) -> Tuple[int, int, int]: ...
    
    @abstractmethod
    def get_3Dframe_size(self, certain_actnum=None) -> Tuple[float, float, float]: ...
    
    @abstractmethod
    def get_model_actnum(self) -> np.ndarray: ...
    
    @abstractmethod
    def get_origin(self, certain_actnum=None) -> Tuple[float, float, float]: ...
    
    @abstractmethod
    def get_top_2d_centers(self) -> Tuple[np.ndarray, np.ndarray]: ...
    
    @abstractmethod
    def get_cell_centers(self, I_list: List[int], J_list: List[int], K_list: List[int]) -> np.ndarray: ...
    
    @abstractmethod
    def get_static_fields(self, field_names: List[str], certain_actnum=None) -> np.ndarray: ...
    
    @abstractmethod
    def get_dynamic_steps(self) -> List[int]: ...
    
    @abstractmethod
    def get_dynamic_datetimes(self) -> Dict[int, datetime]: ...
    
    @abstractmethod
    def get_dynamic_fields(self, idx_step: int, field_names: List[str], certain_actnum=None) -> np.ndarray: ...

    # 用于 3D 渲染的接口 (Zarr 替身也需要实现这些来画图)
    @abstractmethod
    def get_pv_static_mesh(self, field_names: Union[str, List[str]], certain_actnum=None) -> Any: ...
    
    @abstractmethod
    def get_pv_dynamic_mesh(self, time_step: int, field_names: Union[str, List[str]], certain_actnum=None) -> Any: ...


class IWellData(ABC):
    """井控与物理轨迹数据契约"""
    
    # 静态属性：要求子类必须存在 static_wells 和 trj_data 字典
    static_wells: dict
    trj_data: dict

    @abstractmethod
    def get_steps_info(self, step_indices) -> pd.DataFrame: ...
    
    @abstractmethod
    def get_step_to_active_wells_map(self, target_steps: list, target_wells: list = None) -> dict: ...
    
    @abstractmethod
    def get_wells_by_step_idx_and_name(self, step_indices, well_names, cluster_by) -> pd.DataFrame: ...
    
    @abstractmethod
    def get_pv_well_tracks(self, model: IModelData, step_idx: int, well_names=None, **kwargs) -> Tuple[List[dict], List[dict]]: ...


class IWellSmry(ABC):
    """单井/全场动态产量契约"""
    
    df_field: Optional[pd.DataFrame]
    df_well: Optional[pd.DataFrame]

    @abstractmethod
    def get_well_data(self, well_name: str) -> pd.DataFrame: ...