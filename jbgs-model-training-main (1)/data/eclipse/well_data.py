# 主要读取 Eclipse 项目 .Data 文件中的井控、射孔、井位
import pyvista as pv
import bisect
import re
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from data.interfaces import IWellData

class EclipseWellData(IWellData):

    def __init__(self, 
                 project_dir: str | Path, project_name: str, 
                 sch_file=None, ev_file=None, vol_files=None, trj_file=None):
        
        dat_file = Path(project_dir) / f"{project_name}.DATA"
        self.dates = []            
        self.static_wells = {}     
        self._event_records = []   
        self.status = set()
        self.modes = set()
        self.keywords = set()

        # 用于存储从 .trj 解析出来的高精度三维坐标字典
        self.trj_data = {}
        
        # =================================================================
        # 🌟 智能推断引擎：如果外部没有把文件传全，则主动去 .DATA 里嗅探
        # =================================================================
        if any(f is None for f in [sch_file, ev_file, trj_file]) or not vol_files:
            sniffed_files = self._sniff_include_files(dat_file)
            
            # 按需补全
            if sch_file is None and sniffed_files['sch']:
                sch_file = sniffed_files['sch']
            if ev_file is None and sniffed_files['ev']:
                ev_file = sniffed_files['ev']
            if not vol_files and sniffed_files['vol']:
                vol_files = sniffed_files['vol']
                
            # 轨迹文件双重 Fallback：如果在 INCLUDE 没找到，尝试在同级目录下找同名 .trj
            if trj_file is None:
                trj_file = sniffed_files['trj']
                if not trj_file:
                    for f in Path(project_dir).iterdir():
                        if f.is_file() and f.suffix.lower() == '.trj' and f.stem.lower() == project_name.lower():
                            trj_file = f
                            break
                            
            print(f"🔍 [智能推断] SCH: {Path(sch_file).name if sch_file else 'None'} | "
                  f"EV: {Path(ev_file).name if ev_file else 'None'} | "
                  f"TRJ: {Path(trj_file).name if trj_file else 'None'}")
        # =================================================================

        # 1. 解析主文件 (可能包含些基础信息)
        self._parse_file(dat_file)
        
        # 2. 解析井位和射孔事件文件 (重点！)
        if ev_file:
            self._parse_file(ev_file)
            
        # 3. 解析生产历史文件 (可能有多个)
        if vol_files:
            for vol_file in vol_files:
                self._parse_file(vol_file)
                
        # 4. 解析动态调度文件 (OPEN/SHUT等)
        if sch_file:
            self._parse_file(sch_file)
            
        # 5. 解析物理轨迹文件 (.trj)
        if trj_file:
            self._parse_trj(trj_file)

        # 6. 统一转换为 DataFrame 便于后续查询
        if self._event_records:
            self.df_events = pd.DataFrame(self._event_records)
        else:
            self.df_events = pd.DataFrame()
            
        del self._event_records 

    # =================================================================
    # 🌟 辅助嗅探方法 (防弹级路径解析)
    # =================================================================
    def _sniff_include_files(self, dat_file: Path) -> dict:
        """
        提取 DATA 文件中的 INCLUDE 关键字，并根据后缀名自动分发。
        完美兼容 Windows/Linux 斜杠，且无视大小写。
        """
        res = {'sch': None, 'ev': None, 'vol': [], 'trj': None}
        if not dat_file.exists():
            return res

        with open(dat_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # 1. 剔除所有 Eclipse 注释 (-- 开头)，防止读到被注释掉的旧文件
        text = re.sub(r"--.*$", "", text, flags=re.MULTILINE)

        # 2. 匹配 INCLUDE 后面的文件名 (兼容带引号和不带引号)
        pattern = re.compile(r"\bINCLUDE\b\s+(?:['\"](.*?)['\"]|([^\s/]+))", re.IGNORECASE)
        
        for m in pattern.finditer(text):
            fname = m.group(1) or m.group(2)
            if not fname: 
                continue
                
            # 暴力提取纯文件名，兼容 \ 和 / 
            pure_name = fname.strip().replace('\\', '/').split('/')[-1]
            
            # 在 .DATA 所在目录下无视大小写搜寻实体文件
            resolved_path = None
            for f in dat_file.parent.iterdir():
                if f.is_file() and f.name.lower() == pure_name.lower():
                    resolved_path = f
                    break
                    
            if not resolved_path:
                continue
                
            # 3. 根据后缀名自动路由归类
            ext = resolved_path.suffix.lower()
            if ext in ['.sch', '.inc']:
                res['sch'] = resolved_path
            elif ext in ['.ev', '.evt']:
                res['ev'] = resolved_path
            elif ext == '.vol':
                res['vol'].append(resolved_path)
            elif ext == '.trj':
                res['trj'] = resolved_path
                
        return res

    def _parse_trj(self, filepath):
        """专门解析高精度物理井轨迹文件的模块"""

        target_path = Path(filepath)
        if not target_path.exists():
            print(f"⚠️ 警告: 找不到轨迹文件 {target_path}")
            return

        with open(target_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        current_well = None
        in_trajectory = False
        coords = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('--'): continue
                
            if line.startswith('WELLNAME'):
                current_well = line.split()[1]
                coords = []
            elif line == 'TRAJECTORY':
                in_trajectory = True
            elif line.startswith('END_TRAJECTORY'):
                in_trajectory = False
                parts = line.split()
                if len(parts) >= 5: # MD X Y Z ...
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    coords.append([x, y, z])
                
                # 当前井收集完毕，入库
                if current_well and len(coords) > 1:
                    self.trj_data[current_well] = np.array(coords)
                    
            elif in_trajectory:
                parts = line.split()
                if len(parts) >= 12:  # ENTRY_X, Y, Z 在 4, 5, 6
                    x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                    coords.append([x, y, z])

    def _parse_file(self, datafile):
        # ... (解析逻辑保持你的不变) ...
        month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                     'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

        def _expand_tokens(raw_tokens):
            expanded = []
            for t in raw_tokens:
                if '*' in t and t.replace('*', '').isdigit():
                    count = int(t.replace('*', ''))
                    expanded.extend(['DEFAULT'] * count)
                else:
                    expanded.append(t)
            return expanded

        with open(datafile, 'r') as f:
            text = f.read().upper()

        text = re.sub(r"--.*$", "", text, flags=re.MULTILINE)
        text = text.replace("'", "").replace('"', '')

        pattern = re.compile(
            r"^(WELSPECS|COMPDAT|WCONINJE|WCONHIST|WCONPROD|DATES)\b(.*?)(?=^\s*/)", 
            re.MULTILINE | re.DOTALL
        )

        for match in pattern.finditer(text):
            keyword = match.group(1)
            content = match.group(2)
            
            records = [line.split() for line in content.split('\n') if line.strip()]
            if not records:
                continue

            if keyword == "WELSPECS":
                for raw_tokens in records:
                    tokens = _expand_tokens(raw_tokens)
                    if not tokens: continue
                    well = tokens[0]
                    wtype = tokens[5] if len(tokens) > 5 and tokens[5] != "DEFAULT" else "PROD"
                    if well not in self.static_wells:
                        self.static_wells[well] = {"type": wtype, "welspecs_params": tokens[1:], "completions": []}
                    else:
                        self.static_wells[well]["type"] = wtype
                        self.static_wells[well]["welspecs_params"] = tokens[1:]

            elif keyword == "COMPDAT":
                for raw_tokens in records:
                    tokens = _expand_tokens(raw_tokens)
                    if len(tokens) >= 5:
                        well = tokens[0]
                        i_idx, j_idx = int(tokens[1]), int(tokens[2])
                        k1, k2 = int(tokens[3]), int(tokens[4])
                        
                        if well not in self.static_wells:
                            self.static_wells[well] = {"type": "UNKNOWN", "welspecs_params": [], "completions": []}
                        
                        for k in range(k1, k2 + 1):
                            comp_info = {"I": i_idx, "J": j_idx, "K": k, "raw_params": tokens[1:]}
                            self.static_wells[well]["completions"].append(comp_info)

            elif keyword in ("WCONINJE", "WCONHIST", "WCONPROD"):
                step_idx = len(self.dates)
                for raw_tokens in records:
                    tokens = _expand_tokens(raw_tokens)
                    if not tokens: continue
                    well = tokens[0]
                    
                    if keyword == "WCONINJE":
                        status = tokens[2] if len(tokens) > 2 and tokens[2] != "DEFAULT" else "OPEN"
                        mode = tokens[3] if len(tokens) > 3 else "DEFAULT"
                    else:
                        status = tokens[1] if len(tokens) > 1 and tokens[1] != "DEFAULT" else "OPEN"
                        mode = tokens[2] if len(tokens) > 2 else "DEFAULT"

                    status = status if status in ["OPEN", "SHUT"] else "OPEN"

                    self._event_records.append({
                        "step_idx": step_idx,
                        "well": well,
                        "keyword": keyword,
                        "status": status,
                        "mode": mode,
                        "raw_params": tokens[1:] 
                    })

                    self.modes.add(mode)
                    self.status.add(status)
                    self.keywords.add(keyword)

            elif keyword == "DATES":
                for tokens in records:
                    if len(tokens) >= 3:
                        try:
                            day = int(tokens[0])
                            month = month_map.get(tokens[1][:3], 1)
                            year = int(tokens[2].replace('/', ''))
                            self.dates.append(datetime(year, month, day))
                        except ValueError:
                            pass

    # ==========================================
    #       基于 Pandas 的光速查询 API (你的重构版)
    # ==========================================

    def __len__(self):
        return len(self.df_events)

    def get_dates(self):
        return list(self.dates)
    
    def get_modes(self):
        return list(self.modes)
    
    def get_states(self):
        return list(self.status)

    def get_keywords(self):
        return list(self.keywords)

    def get_well_names(self):
        return sorted(list(self.static_wells.keys()))
    
    def get_wells_by_step_idx_and_name_raw(self, step_indices=None, well_names=None, cluster_by=None):
        """
        统一的数据抽取与物理聚类底层方法。
        Args:
            step_indices: 截取到指定的最大时间步 (int, datetime, 或它们的 list)
            well_names: 筛选指定的井名 (str 或 list)
            cluster_by: 聚块排序的层级，例如 ['step_idx', 'well'] 或 ['well', 'step_idx']
        """
        if self.df_events.empty:
            return pd.DataFrame()
            
        df_filtered = self.df_events
        
        # 1. 时间步过滤逻辑 (step_idx <= max_step)
        if step_indices is not None:
            if isinstance(step_indices, (int, datetime)):
                step_indices = [step_indices]
                
            # 🌟 核心修改：如果是 datetime，直接用汇率接口做模糊/精确匹配
            if isinstance(step_indices[0], datetime):
                target_indices = self.datetime_to_step_idx(step_indices)
            else:
                target_indices = [i if i >= 0 else len(self.dates) + i for i in step_indices]
            
            if not target_indices:
                return pd.DataFrame()
                
            max_step = max(target_indices)
            df_filtered = df_filtered[df_filtered['step_idx'] <= max_step]
            
        # 2. 井名过滤逻辑 (well in well_names)
        if well_names is not None:
            if isinstance(well_names, str):
                target_wells = [well_names]
            else:
                target_wells = well_names
            df_filtered = df_filtered[df_filtered['well'].isin(target_wells)]
            
        if df_filtered.empty:
            return pd.DataFrame()
            
        # 显式 copy，防止后续操作触发 SettingWithCopyWarning
        subset_events = df_filtered.copy()
        
        # 3. 联合去重
        # 注意：无论是按什么顺序聚类，去重的联合唯一标识都是这两个字段，顺序无所谓
        unique_events = subset_events.drop_duplicates(subset=['step_idx', 'well'], keep='last')
        
        # 4. 黑科技：利用 Hash 表底层聚类结果，用 iloc O(1) 一次性物理聚类
        if cluster_by is None:
            cluster_by = ['step_idx', 'well'] # 默认后备顺序
        
        return unique_events


    def get_wells_by_step_idx_and_name(self, step_indices=None, well_names=None, cluster_by=None):
        """
        统一的数据抽取与物理聚类底层方法。
        Args:
            step_indices: 截取到指定的最大时间步 (int, datetime, 或它们的 list)
            well_names: 筛选指定的井名 (str 或 list)
            cluster_by: 聚块排序的层级，例如 ['step_idx', 'well'] 或 ['well', 'step_idx']
        """
        
        unique_events = self.get_wells_by_step_idx_and_name_raw(step_indices, well_names, cluster_by)
            
        grouped = unique_events.groupby(cluster_by, sort=False)
        clustered_indices = np.concatenate(list(grouped.indices.values()))
        result_df = unique_events.iloc[clustered_indices].reset_index(drop=True)
        
        return result_df


    def get_steps_info(self, step_indices=None):
        """
        截取到指定时间步的流水数据。
        注意：外部如果需要当步“快照”，请对返回的 df 执行 .drop_duplicates(subset=['well'], keep='last')
        """
        return self.get_wells_by_step_idx_and_name(
            step_indices=step_indices, 
            well_names=None, 
            cluster_by=['step_idx', 'well']
        )

    def datetime_to_step_idx(self, target_dts: list) -> list:
        """
        汇率转换：找 <= datetime 的最近 well step_idx
        利用目标序列的有序性，引入 lo 参数实现 O(M+N) 极速匹配
        """
        if not self.dates:
            return [0] * len(target_dts) 
        
        res = []
        lo = 0  # 🌟 核心优化：记忆上次的查找位置
        
        for dt in target_dts:
            if dt is None:
                res.append(-1)
                continue
                
            # 从上次找到的位置 'lo' 开始向后二分，大幅缩小搜索空间
            idx = bisect.bisect_right(self.dates, dt, lo=lo)
            lo = idx  # 🌟 更新起点
            
            res.append(idx - 1 if idx > 0 else 0)
            
        return res


    def get_wells_history(self, well_names=None):
        # 兼容原来默认获取全部井的逻辑
        if well_names is None:
            well_names = self.get_well_names()
            
        return self.get_wells_by_step_idx_and_name(
            step_indices=None,
            well_names=well_names,
            cluster_by=['well', 'step_idx']
        )
    
    def get_step_to_active_wells_map(self, target_steps: list, target_wells: list = None) -> dict:
        """
        利用单向状态机极速推演指定时间步的活井名单（完美处理 SHUT/OPEN 状态的跨步继承）。
        彻底消除 DataFrame 在循环中反复切片与去重的开销。
        
        Args:
            target_steps: 需要查询的目标时间步 int(step_idx) 列表 或 datetime(绝对时间) 列表
            target_wells: 候选井池。如果不传，则默认全局静态井。
        Returns:
            dict: {step_idx: [active_well_1, active_well_2, ...]}
        """

        if not target_steps: 
            return {}
            
        # 1. 拦截解析：将外部货币 (datetime) 统一兑换为内部索引 (step_idx)
        if target_wells is None:
            base_targets = set(self.get_well_names())
        else:
            base_targets = set(target_wells)

        is_datetime = isinstance(target_steps[0], datetime)
        internal_target_steps = self.datetime_to_step_idx(target_steps) if is_datetime else target_steps
            
        internal_step_to_wells = {}
        current_active = set()
        
        raw_events = self.df_events
        events_by_step = raw_events.groupby('step_idx') if not raw_events.empty else {}
        
        max_step = max(internal_target_steps) if internal_target_steps else 0
        internal_target_steps_set = set(internal_target_steps)
        
        # 2. 核心状态机推演 (基于内部的 step_idx)
        for step in range(max_step + 1):
            if step in events_by_step.groups:
                step_events = events_by_step.get_group(step)
                for _, row in step_events.iterrows():
                    w = row['well']
                    if w in base_targets:
                        status = str(row.get('status', 'OPEN')).upper()
                        if status != "SHUT":
                            current_active.add(w)
                        else:
                            current_active.discard(w) 
                            
            if step in internal_target_steps_set:
                internal_step_to_wells[step] = list(current_active)
                
        # 3. 🌟 完美归还：按照你传入的货币类型 (datetime/int) 原路返回字典
        result_map = {}
        for original_key, internal_step in zip(target_steps, internal_target_steps):
            result_map[original_key] = internal_step_to_wells.get(internal_step, [])
            
        return result_map
    
    # ==========================================
    #             重构后的周边功能
    # ==========================================

    def plot_timeline(self, width=100, well_names=None):
        """在控制台打印彩色井史进度条"""
        all_dates = self.get_dates()
        if not all_dates:
            print("错误: 未找到任何日期信息 (DATES)。")
            return
        
        target_wells = self.get_well_names() if well_names is None else (
            [well_names] if isinstance(well_names, str) else well_names
        )

        # 1. 拿取 DataFrame
        histories_df = self.get_wells_history(target_wells)
        max_n = max(len(n) for n in target_wells) if target_wells else 10
        total_steps = len(all_dates)
        
        RED, BLUE, GRAY, RESET = "\033[91m", "\033[94m", "\033[90m", "\033[0m"
        print(f"\n{'WELL':<{max_n}} | TIMELINE: {all_dates[0].year} -> {all_dates[-1].year}")
        print("-" * (max_n + width + 3))

        # 2. 从 DataFrame 里按井名提取数据
        for name in target_wells:
            # Pandas 优雅的条件筛选
            w_df = histories_df[histories_df['well'] == name] if not histories_df.empty else pd.DataFrame()
            
            if w_df.empty:
                print(f"{name:<{max_n}} | {' (No operational history)':<{width}}")
                continue
            
            states = [('NONE', 'UNK')] * total_steps
            for _, row in w_df.iterrows():
                s_idx = row['step_idx']
                if 0 <= s_idx < total_steps:
                    states[s_idx:] = [(row['status'], row['mode'])] * (total_steps - s_idx)

            line = ""
            well_static_type = self.static_wells.get(name, {}).get('type', 'PROD').upper()
            
            for i in range(width):
                idx = int(i * total_steps / width)
                s, m = states[idx]
                if s == 'OPEN':
                    is_injector = name.upper().startswith('W') or "WAT" in well_static_type or str(m) == 'RATE'
                    line += f"{BLUE if is_injector else RED}━{RESET}"
                elif s == 'SHUT':
                    line += f"{GRAY}─{RESET}"
                else:
                    line += " "
            
            print(f"{name:<{max_n}} | {line}")

    def get_pv_well_tracks(self, model, step_idx: int, well_names=None,
                           display_radius: float = 3.0, extend_track_length: float = 0.0,
                           show_perforation: bool = True, show_labels: bool = True, 
                           label_scale: int = 14):
        """
        生成特定时间步的 PyVista 井轨迹和射孔 3D 资产，不直接渲染，返回给外部 Plotter 使用。
        
        Args:
            model: EclipseModelData 模型对象
            step_idx: 时间步索引
            display_radius: 井轨迹线宽
            extend_track_length: 延长井轨迹到油藏以上（单位与网格坐标一致）
            show_perforation: 是否在射孔位置绘制高亮球体
            show_labels: 是否显示井口名称标签
            label_scale: 字体大小
            
        Returns:
            Tuple[List[dict], List[dict]]: 
                - render_items: 包含网格 (mesh) 和渲染参数 (kwargs) 的字典列表
                - label_items: 包含标签坐标 (points)、文本 (texts) 和渲染参数的字典列表
        """
        # 1. 获取截断到该时间步的全量流水分片并提取快照
        df_events = self.get_wells_by_step_idx_and_name(step_idx, well_names, cluster_by=['step_idx', 'well'])
        if df_events.empty:
            return [], []

        snapshot_df = df_events.drop_duplicates(subset=['well'], keep='last')

        # 2. 按颜色/状态分组提取坐标
        color_groups = {}  

        # 兜底：如果没有动态历史，但传入了 trj，也要画出来
        target_wells = set(snapshot_df['well'])
        if self.trj_data and well_names is None:
            target_wells = target_wells.union(self.trj_data.keys())

        for well in target_wells:
            # === 状态与颜色判定 ===
            if well in snapshot_df['well'].values:
                row = snapshot_df[snapshot_df['well'] == well].iloc[0]
                status = str(row['status']).upper()
                mode = str(row['mode']).upper()
                keyword = str(row['keyword']).upper() # 🌟 新增：直接提取控制它的关键字
            else:
                status, mode, keyword = "OPEN", "DEFAULT", ""

            wtype = str(self.static_wells.get(well, {}).get("type", "")).upper()

            if status == "SHUT":
                color = "gray"
            else:
                # 🌟 终极判断：只要控制关键字是 WCONINJE，或者名字包含 I/Z/W，就是注水井！
                is_injector = (
                    keyword == 'WCONINJE' or       # 最可靠的判断依据
                    well.upper().startswith('W') or 
                    well.upper().startswith('I') or 
                    well.upper().startswith('Z') or 
                    "WAT" in wtype or 
                    "INJ" in wtype
                )
                color = "blue" if is_injector else "red"

            if color not in color_groups:
                color_groups[color] = {"track_coords": [], "perf_coords": [], "labels": [], "label_coords": []}

            track_coords, perf_coords = None, None

            # === 坐标提取双引擎 ===
            if well in self.trj_data:
                # 引擎 A：TRJ 高精度物理坐标
                raw_pts = self.trj_data[well].copy()
                raw_pts[:, 2] *= -1 
                track_coords = raw_pts
                perf_coords = np.array([raw_pts[-1]]) 
            else:
                # 引擎 B：调用你新写的 model.get_cell_centers 极速批量提取
                comps = self.static_wells.get(well, {}).get("completions", [])
                if not comps: continue
                
                I_list = [c["I"] for c in comps]
                J_list = [c["J"] for c in comps]
                K_list = [c["K"] for c in comps]
                
                try:
                    # 🚀 利用你的 Numba API 批量获取坐标，无需手动减 1 (你的 API 里已经处理了 1-based)
                    coords = model.get_cell_centers(I_list, J_list, K_list)
                    
                    perf_coords = coords.copy()

                    if extend_track_length > 0:
                        wellhead = coords[0].copy()
                        wellhead[2] += extend_track_length 
                        track_coords = np.vstack([wellhead, coords]) 
                    else:
                        track_coords = coords
                except Exception as e:
                    print(f"⚠️ 井 {well} 坐标计算异常: {e}")
                    continue

            color_groups[color]["track_coords"].append(track_coords)
            if show_perforation:
                color_groups[color]["perf_coords"].append(perf_coords)
            color_groups[color]["labels"].append(well)
            color_groups[color]["label_coords"].append(track_coords[0])

        # 3. 开始构建 PyVista 资产
        render_items = []
        label_items = []

        for col, gdata in color_groups.items():
            if not gdata["track_coords"]: continue
            
            # --- 资产 A：井轨迹 (PolyData) ---
            track_points = np.vstack(gdata["track_coords"])
            lines = []
            offset = 0
            for c in gdata["track_coords"]:
                n = len(c)
                lines.append(n)
                lines.extend(range(offset, offset + n)) 
                offset += n

            track_mesh = pv.PolyData(track_points)
            track_mesh.lines = np.array(lines, dtype=np.int64)
            
            render_items.append({
                "mesh": track_mesh,
                "kwargs": {
                    "color": col,
                    "line_width": display_radius,
                    "render_lines_as_tubes": True,
                    "lighting": True
                }
            })

            # --- 资产 B：射孔球 (PolyData) ---
            # 【修改】使用专属的 perf_coords 数组画球，避免在延长的空气段也画上球
            if show_perforation and gdata["perf_coords"]:
                perf_points = np.vstack(gdata["perf_coords"])
                perf_mesh = pv.PolyData(perf_points) 
                render_items.append({
                    "mesh": perf_mesh,
                    "kwargs": {
                        "color": col,
                        "point_size": display_radius * 1.5,
                        "render_points_as_spheres": True,
                        "lighting": True
                    }
                })

            # --- 资产 C：浮空标签 ---
            if show_labels and gdata["label_coords"]:
                label_coords_np = np.vstack(gdata["label_coords"])
                label_coords_np[:, 2] += 30  # 井口往上偏移悬浮
                
                label_items.append({
                    "points": label_coords_np,
                    "texts": gdata["labels"],
                    "kwargs": {
                        "text_color": col,
                        "font_size": int(label_scale),
                        "point_color": col,
                        "point_size": 0,  
                        "shape_opacity": 0.4,
                        "shape": 'rounded_rect' 
                    }
                })

        return render_items, label_items


def calculate_num_windows(well_history_len, nt_width, shift_width):
    if well_history_len < nt_width:
        return 1 if well_history_len != 0 else 0
    else:
        num_windows = (well_history_len - nt_width) // shift_width + 1
        if (well_history_len - nt_width) % shift_width != 0:
            num_windows += 1
        return num_windows

def calculate_total_tokens(well_data):
    total_windows = 0
    nt_width, shift_width, tokens_per_step = 10, 1, 1024
    
    # 1. 获取全局 df
    history_df = well_data.get_wells_history()
    if history_df is None or history_df.empty:
        print("未找到井动态历史数据。")
        return

    # 2. 手动建立 Date 列 (利用 step_idx 映射)
    total_dates = len(well_data.dates)
    history_df = history_df.copy() # 防警告
    history_df['date'] = history_df['step_idx'].apply(
        lambda x: well_data.dates[x].strftime('%Y-%m-%d') if x < total_dates else 'Initial'
    )
    
    print("=" * 80)
    
    # 3. 按井名 Groupby 极其高效
    for name, w_df in history_df.groupby('well'):
        well_history_len = len(w_df)
        num_windows = calculate_num_windows(well_history_len, nt_width, shift_width)
        
        valid_dates_str = w_df['date'][~w_df['date'].isin(['Initial', 'Init'])]
        
        if len(valid_dates_str) > 1:
            valid_dates = pd.to_datetime(valid_dates_str)
            avg_days = valid_dates.diff().dt.days.mean()
            avg_days_str = f"{avg_days:5.1f} days"
        else:
            avg_days_str = "  N/A days"

        print(f"Well {name:^8} | History: {well_history_len:>3} steps | "
              f"Windows: {num_windows:>3} | Avg Step: {avg_days_str}")
        
        total_windows += num_windows
            
    total_tokens = total_windows * nt_width * tokens_per_step
    
    print("=" * 80)
    print(f"Total Sliding Windows (Samples): {total_windows:,}")
    print(f"Total Effective Tokens for Pre-training: {total_tokens:,}")


def calculate_num_windows(well_history_len, nt_width, shift_width):
    if well_history_len < nt_width:
        return 1 if well_history_len != 0 else 0
    else:
        num_windows = (well_history_len - nt_width) // shift_width + 1
        if (well_history_len - nt_width) % shift_width != 0:
            num_windows += 1
        return num_windows

def calculate_total_tokens(well_data):
    total_windows = 0
    nt_width, shift_width, tokens_per_step = 10, 1, 1024
    
    # 1. 获取全局 df
    history_df = well_data.get_wells_history()
    if history_df is None or history_df.empty:
        print("未找到井动态历史数据。")
        return

    # 2. 手动建立 Date 列 (利用 step_idx 映射)
    total_dates = len(well_data.dates)
    history_df = history_df.copy() # 防警告
    history_df['date'] = history_df['step_idx'].apply(
        lambda x: well_data.dates[x].strftime('%Y-%m-%d') if x < total_dates else 'Initial'
    )
    
    print("=" * 80)
    
    # 3. 按井名 Groupby 极其高效
    for name, w_df in history_df.groupby('well'):
        well_history_len = len(w_df)
        num_windows = calculate_num_windows(well_history_len, nt_width, shift_width)
        
        valid_dates_str = w_df['date'][~w_df['date'].isin(['Initial', 'Init'])]
        
        if len(valid_dates_str) > 1:
            valid_dates = pd.to_datetime(valid_dates_str)
            avg_days = valid_dates.diff().dt.days.mean()
            avg_days_str = f"{avg_days:5.1f} days"
        else:
            avg_days_str = "  N/A days"

        print(f"Well {name:^8} | History: {well_history_len:>3} steps | "
              f"Windows: {num_windows:>3} | Avg Step: {avg_days_str}")
        
        total_windows += num_windows
            
    total_tokens = total_windows * nt_width * tokens_per_step
    
    print("=" * 80)
    print(f"Total Sliding Windows (Samples): {total_windows:,}")
    print(f"Total Effective Tokens for Pre-training: {total_tokens:,}")

if __name__ == "__main__":
    
    args_list = [
        {
            'project_name': 'BYEPD93',
            'project_dir': r'E:\tocug\1c\water_196311-199307',
        },
        {
            'project_name': '1_E100',
            'project_dir': r'E:\tocug\4c\2z\sim',
            'sch_file': r'E:\tocug\4c\2z\sim\sch201012.SCH',
        },
        {
            'project_name': 'X4-6X_E100',
            'project_dir': r'E:\tocug\4c\4-6\x4-6sm20230518' ,
            'sch_file': r'E:\tocug\4c\4-6\x4-6sm20230518\x4-6x.SCH',
            'trj_file': r'E:\tocug\4c\4-6\x4-6sm20230518well.trj'
        }
    ]

    for Args in args_list:

        project_dir, project_name = Args['project_dir'], Args['project_name']
        
        print("正在解析数据并构建 Pandas 数据立方...")
        well_data = EclipseWellData(**Args)
        print(f"解析完成！共 {len(well_data.get_dates())} 个时间步，捕获到 {len(well_data)} 条动态事件。")


        # 【改动】：在拿到 df 后调用 drop_duplicates 获得快照
        last_step_df = well_data.get_steps_info(-1)
        if not last_step_df.empty:
            last_step_snapshot = last_step_df.drop_duplicates(subset=['well'], keep='last')
            print(f'最后一个时间步的井数: {len(last_step_snapshot)}')

        print('well status', well_data.get_states())
        print('well modes', well_data.get_modes())
        print('well keywords', well_data.get_keywords())

        # # 计算所有井的活动时间步能够产生的 token 数量
        # calculate_total_tokens(well_data)
        # quit()

        # well_data.plot_timeline(width=80)
        # quit()

        # ------------------- 可视化呈现 ------------------
        # 初始化 PyVista 渲染视窗
        plotter = pv.Plotter()
        
        # 联合绘制
        from eclipse.model_data import EclipseModelData
        model = EclipseModelData(project_dir, project_name)
        actnum = model.get_model_actnum()
        nz, ny, nx = model.get_3Dframe_dim()
        # actnum[nz//2:, :, :] = False  # 裁剪掉下半部分
        
        # 核心解耦：只获取数据对象，不直接渲染
        mesh = model.get_pv_static_mesh("PORO", certain_actnum=actnum)
        plotter.add_mesh(mesh, cmap='jet', show_edges=False, scalar_bar_args={'title': "PORO - Top Half"})

        # 获取第 50 步的井网 3D 资产
        render_items, label_items = well_data.get_pv_well_tracks(
            model, 
            step_idx=-1, 
            display_radius=5.0,        # 曲线粗细
            extend_track_length=200.0, # 井轨迹延长长度
            show_perforation=True,     # 是否显示射孔球
            show_labels=True, 
            label_scale=14
        )

        # 提取 origin 并转为 numpy array 以方便计算
        origin = model.get_origin()

        print('origin', origin)
        print(model.get_3Dframe_dim())


        # 将资产按预设好的样式装配到 Plotter 中
        for item in render_items:
            # 修复点 1：直接修改 mesh.points 数组，完成物理平移
            pv_mesh = item["mesh"]
            pv_mesh.points = pv_mesh.points - origin
    
            plotter.add_mesh(pv_mesh, **item["kwargs"])
            
        for lbl in label_items:
            # 修复点 2：Numpy 数组相减，实现标签悬浮点的平移
            shifted_points = lbl["points"] - origin
            plotter.add_point_labels(shifted_points, lbl["texts"], **lbl["kwargs"])

        # 最终渲染
        plotter.set_background('white')
        plotter.show()