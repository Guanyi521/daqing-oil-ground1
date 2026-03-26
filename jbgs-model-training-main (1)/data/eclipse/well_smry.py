import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
from opm.io.ecl import ESmry
from data.interfaces import IWellSmry

class EclipseWellSmry(IWellSmry):
    """
    极速提取 Eclipse 模拟器 Summary 产量结果的类。
    将不同作用域 (Field, Region, Group, Well, Connection) 的结果隔离为独立的宽表 DataFrame。
    """
    
    def __init__(self, project_dir: Union[str, Path], project_name: str):
        """
        初始化并一次性解析所有的 Summary 数据。
        
        Args:
            project_dir: 项目目录路径
            project_name: 项目名称
        """
        file_path = Path(project_dir) / f"{project_name}.SMSPEC"
        if file_path.exists():
            self.file_path = Path(file_path)
        else:
            raise FileNotFoundError(f"未找到文件: {self.file_path}")
            
        self.smry = ESmry(str(self.file_path))
        self.keys = list(self.smry.keys())
        
        # 各个域的 DataFrame，若无对应数据则为 None
        self.df_field: Optional[pd.DataFrame] = None
        self.df_region: Optional[pd.DataFrame] = None
        self.df_group: Optional[pd.DataFrame] = None
        self.df_well: Optional[pd.DataFrame] = None
        self.df_conn: Optional[pd.DataFrame] = None
        
        self._parse_all_data()

    def _parse_all_data(self) -> None:
        """内部核心：分类解析所有的 Key，构建基准时间表并生成各域 DataFrame"""
        if not self.keys:
            return

        # 1. 提取基础时间轴 (保证所有表都有同一套 step_idx, time_days, date)
        time_days = self.smry['TIME'] if 'TIME' in self.keys else np.arange(len(self.smry))
        df_time = pd.DataFrame({
            'step_idx': np.arange(len(time_days)),
            'time_days': time_days
        })
        
        # 如果存在年月日，一键向量化转为 Datetime
        if all(k in self.keys for k in ['YEAR', 'MONTH', 'DAY']):
            try:
                # pandas 的 to_datetime 字典解析法，速度极快
                df_time['date'] = pd.to_datetime({
                    'year': self.smry['YEAR'],
                    'month': self.smry['MONTH'],
                    'day': self.smry['DAY']
                }, errors='coerce')
            except Exception:
                pass

        # 2. 按前缀与层级分类所有的 Keys
        field_keys, region_keys, group_keys, well_keys, conn_keys = [], [], [], [], []
        
        for k in self.keys:
            # 过滤掉全局控制属性
            if k in ('TIME', 'YEAR', 'MONTH', 'DAY', 'TIMESTEP') or k.startswith('WLIST'):
                continue
                
            parts = k.split(':')
            prop = parts[0]
            
            # 没有冒号，一般是全场 (Field) 属性，如 FOPR, FOPT
            if len(parts) == 1:
                if prop.startswith('F'): 
                    field_keys.append(k)
            # 有冒号的按首字母区分域
            elif len(parts) >= 2:
                prefix = prop[0]
                if prefix == 'W':
                    well_keys.append(k)
                elif prefix == 'C':
                    conn_keys.append(k)
                elif prefix == 'G':
                    group_keys.append(k)
                elif prefix == 'R':
                    region_keys.append(k)

        # 3. 构建各个域的数据立方体
        self.df_field = self._build_field_df(field_keys, df_time)
        self.df_well = self._build_entity_df(well_keys, df_time, 'well')
        self.df_region = self._build_entity_df(region_keys, df_time, 'region')
        self.df_group = self._build_entity_df(group_keys, df_time, 'group')
        self.df_conn = self._build_entity_df(conn_keys, df_time, 'conn_id')

        # 主动释放底层对象引用，防止 OOM
        del self.smry
        self.keys.clear()

    def _build_field_df(self, keys: List[str], df_time: pd.DataFrame) -> Optional[pd.DataFrame]:
        """构建全场属性宽表"""
        if not keys: return None
        data = {k: self.smry[k] for k in keys}
        df = pd.DataFrame(data)
        
        # 将时间列插在最前面
        return pd.concat([df_time, df], axis=1)

    def _build_entity_df(self, keys: List[str], df_time: pd.DataFrame, id_col_name: str) -> Optional[pd.DataFrame]:
        """
        高度优化的实体级宽表生成器。
        将 'WOPR:W1', 'WWPR:W1' 等拆分为 [step_idx, date, well, WOPR, WWPR]
        """
        if not keys: return None
        
        # 分组组装: entity_dict = {'W1': {'WOPR': array, 'WWPR': array}}
        entity_dict = {}
        for k in keys:
            parts = k.split(':', 1)
            prop, entity = parts[0], parts[1]
            if entity not in entity_dict:
                entity_dict[entity] = {}
            entity_dict[entity][prop] = self.smry[k]

        dfs = []
        for entity, prop_data in entity_dict.items():
            # 1. 为该实体生成宽表
            df_entity = pd.DataFrame(prop_data)
            # 2. 贴上时间标签
            for col in df_time.columns:
                df_entity[col] = df_time[col]
            # 3. 贴上实体标识 (如 well='W1')
            df_entity[id_col_name] = entity
            dfs.append(df_entity)

        if not dfs: return None
        
        # 4. 光速拼接
        final_df = pd.concat(dfs, ignore_index=True)
        
        # 5. 整理列顺序：使 step, time_days, date, id_col 排在最前面
        base_cols = list(df_time.columns) + [id_col_name]
        other_cols = [c for c in final_df.columns if c not in base_cols]
        return final_df[base_cols + other_cols]

    def get_well_data(self, well_name: str) -> pd.DataFrame:
        """
        快捷接口：获取单口井的动态产量宽表。
        
        Args:
            well_name: 井名
            
        Returns:
            pd.DataFrame: 该井的产量表。如果模型中无此井，返回空 DataFrame。
        """
        if self.df_well is None:
            return pd.DataFrame()
        return self.df_well[self.df_well['well'] == well_name].copy()



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # =============================================================================
    # 测试入口：1:1 完美复刻原始 esmry.py 的画图逻辑
    # =============================================================================

    def plot_production_curves_from_df(conseq: EclipseWellSmry, plot_type='well_oil_rate', wells=None):
        """
        基于 DataFrame 接口复刻的画图函数。代码逻辑大幅简化！
        """
        plt.figure(figsize=(10, 6))
        x_col = 'time_days'
        xlabel = 'Time (days)'
        
        # ---------------- 1. 全场级别绘图 ----------------
        if plot_type.startswith('field_'):
            df = conseq.df_field
            if df is None or df.empty:
                print(f"没有提取到 Field 级别数据，无法绘制 {plot_type}")
                return
                
            if plot_type == 'field_rate':
                if 'FOPR' in df.columns: plt.plot(df[x_col], df['FOPR'], label='Field Oil Rate')
                if 'FWPR' in df.columns: plt.plot(df[x_col], df['FWPR'], label='Field Water Rate')
                ylabel = 'Field Production Rate'
                
            elif plot_type == 'field_cum':
                if 'FOPT' in df.columns: plt.plot(df[x_col], df['FOPT'], label='Cumulative Oil')
                if 'FWPT' in df.columns: plt.plot(df[x_col], df['FWPT'], label='Cumulative Water')
                ylabel = 'Field Cumulative Production'
                
            else:
                raise ValueError(f"Unknown plot_type: {plot_type}")

        # ---------------- 2. 单井级别绘图 ----------------
        elif plot_type.startswith('well_'):
            df = conseq.df_well
            if df is None or df.empty:
                print(f"没有提取到 Well 级别数据，无法绘制 {plot_type}")
                return

            # 确定需要绘制的井列表
            target_wells = wells if wells else df['well'].unique()

            # 利用 Pandas groupby 优雅地遍历每一口井
            for well, w_df in df.groupby('well'):
                if well not in target_wells:
                    continue
                    
                # 确保时间轴是有序的
                w_df = w_df.sort_values(x_col)
                
                if plot_type == 'well_oil_rate' and 'WOPR' in w_df.columns:
                    plt.plot(w_df[x_col], w_df['WOPR'], label=well)
                    ylabel = 'Well Oil Rate'
                    
                elif plot_type == 'well_water_rate' and 'WWPR' in w_df.columns:
                    plt.plot(w_df[x_col], w_df['WWPR'], label=well)
                    ylabel = 'Well Water Rate'
                    
                elif plot_type == 'well_oil_cum' and 'WOPR' in w_df.columns:
                    # 完美复刻原有的 dt 积分逻辑
                    dt = np.concatenate(([0], np.diff(w_df[x_col])))
                    cum = np.cumsum(w_df['WOPR'] * dt)
                    plt.plot(w_df[x_col], cum, label=well)
                    ylabel = 'Cumulative Well Oil'
                    
                elif plot_type == 'well_water_cum' and 'WWPR' in w_df.columns:
                    dt = np.concatenate(([0], np.diff(w_df[x_col])))
                    cum = np.cumsum(w_df['WWPR'] * dt)
                    plt.plot(w_df[x_col], cum, label=well)
                    ylabel = 'Cumulative Well Water'

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(plot_type.replace('_', ' ').title())
        plt.legend()
        plt.tight_layout()
        plt.show()

    proj_dir = Path('../')
    proj_name = 'BYEPD93'
    smry_file = proj_dir / f"{proj_name}.SMSPEC"

    # 1. 实例化新接口
    print("正在基于 Pandas DataFrame 解析产量数据...")
    conseq = EclipseWellSmry(smry_file)
    print("解析完成！")
    
    # 2. 依次调用 6 种经典图表，效果与老脚本绝对一致！
    plot_production_curves_from_df(conseq, plot_type='well_oil_rate')
    plot_production_curves_from_df(conseq, plot_type='well_water_rate')
    plot_production_curves_from_df(conseq, plot_type='well_oil_cum')
    plot_production_curves_from_df(conseq, plot_type='well_water_cum')
    plot_production_curves_from_df(conseq, plot_type='field_rate')
    plot_production_curves_from_df(conseq, plot_type='field_cum')