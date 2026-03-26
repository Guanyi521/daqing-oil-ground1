# data/zarr/build_dataset.py
import zarr
import pickle
import numpy as np
import concurrent.futures as cf
from pathlib import Path
from functools import partial

# 引入 numcodecs 用于 Zarr V3 的压缩配置
from numcodecs import Blosc

# 引入底层的有界流式执行器
from data.distributed import BoundedExecutor

# 引入原生解析器
from data.eclipse.model_data import EclipseModelData
from data.eclipse.well_data import EclipseWellData
from data.eclipse.well_smry import EclipseWellSmry

def build_zarr_for_project(source_dir: str|Path, proj_name: str, output_dir: str|Path, 
                           static_fields: list, dynamic_fields: list):
    """
    独立进程中执行的重量级提取、提纯与压缩任务。
    【架构注意】此函数必须在模块顶层定义，以保证在多进程(ProcessPool)下的安全序列化。
    """
    source = Path(source_dir)
    target = output_dir / Path(source_dir).name
    target.mkdir(parents=True, exist_ok=True)

    # 1. 实例化原生解析器 (触发重度 I/O 和正则解析)
    model = EclipseModelData(source, proj_name)
    well = EclipseWellData(source, proj_name)
    
    # 压盘时必须全量解析产量数据，所以强行设置为 lazy=False (如果你修改了 WellSmry 类的设计)
    well_smry = EclipseWellSmry(source, proj_name, lazy=False)

    # 2. 初始化 Zarr 树结构
    # 适配 Zarr V3 接口：采用 numcodecs 进行 zstd 3级极致压缩
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    
    # 瞬间打开或创建 Zarr 根目录
    root = zarr.open(str(target / f"{proj_name}.zarr"), mode='w')

    # 3. 剥离并存储巨型 NumPy 数组到 Zarr (适配 V3 的 create_array)
    # 几何基底
    root.create_array('coord_1d', data=model.coord_1d, compressors=[compressor])
    root.create_array('zcorn_1d', data=model.zcorn_1d, compressors=[compressor])
    root.create_array('actnum_3d', data=model.get_model_actnum(), compressors=[compressor])
    
    # 静态场
    static_grp = root.create_group('static')
    if static_fields:
        static_arrays = model.get_static_fields(static_fields)
        # 兼容单维或多维返回
        if static_arrays.ndim == 1: 
            static_arrays = static_arrays[np.newaxis, :]
        for idx, f in enumerate(static_fields):
            static_grp.create_array(f, data=static_arrays[idx], compressors=[compressor])

    # 动态场
    dynamic_grp = root.create_group('dynamic')
    for step in model.get_dynamic_steps():
        step_grp = dynamic_grp.create_group(f"Step_{step}")
        try:
            dyn_arrays = model.get_dynamic_fields(step, dynamic_fields)
            if dyn_arrays.ndim == 1: 
                dyn_arrays = dyn_arrays[np.newaxis, :]
            for idx, f in enumerate(dynamic_fields):
                step_grp.create_array(f, data=dyn_arrays[idx], compressors=[compressor])
        except Exception as e:
            print(f"⚠️ [{proj_name}] Step {step} 提取动态场失败: {e}")

    # 4. 🌟 打包轻量级元数据与 Pandas 宽表为 Pickle
    meta_payload = {
        'model': {
            'frame_dim': model.get_3Dframe_dim(),
            'frame_size': model.get_3Dframe_size(),
            'origin': model.get_origin(),
            'nblocks': model.nblocks,
            'radial': model.radial,
            'dynamic_steps': model.get_dynamic_steps(),
            'dynamic_datetimes': model.get_dynamic_datetimes()
        },
        'well_data': {
            'dates': well.dates,
            'static_wells': well.static_wells,
            'trj_data': well.trj_data,
            'df_events': well.df_events,
        },
        'well_smry': {
            'df_field': well_smry.df_field if well_smry else None,
            'df_well': well_smry.df_well if well_smry else None
        }
    }
    
    # 将 Pickle 无缝塞入 .zarr 文件夹内部，随压包整体迁移
    with open(target / 'meta_payload.pkl', 'wb') as f:
        pickle.dump(meta_payload, f)
        
    return f"✅ 成功打包: {proj_name} -> {target.name}"


def zarr_task_generator(projects, output_dir, static_fields, dynamic_fields):
    """
    任务发生器 (Producer)：
    惰性产生任务。内存占用永远是 O(1)，完美化解 10万+ 规模任务的内存溢出危机。
    """
    for p in projects:
        source_dir = p["dir"]
        proj_name = p["name"]
        
        # 🌟 核心技巧：利用 functools.partial 包装任务与参数
        # 这样生成的对象是 Picklable 的，能安全穿越进 ProcessPoolExecutor 的进程边界
        task_closure = partial(
            build_zarr_for_project,
            source_dir=source_dir, 
            proj_name=proj_name, 
            output_dir=output_dir, 
            static_fields=static_fields, 
            dynamic_fields=dynamic_fields
        )
        yield task_closure


if __name__ == "__main__":
    # ==========================================
    # ETL 作业配置区
    # ==========================================
    projects = [
        {
            "dir": r"D:\YihengZhu\jbgs-lma\proj-phase1\eclipse-models\1c\water_flooding", 
            "name": "BYEPD93"
        },
        # 未来你可以从 CSV、数据库或 glob 遍历生成这个列表，支持放入成千上万个对象
    ]
    
    output_dir = Path("./zarr_datasets")
    
    # 定义你要提纯保留的物理场
    STATIC_FIELDS = ["PORO", "PERMX"]
    DYNAMIC_FIELDS = ["PRESSURE", "SWAT"]

    print(f"🚀 准备启动 Zarr 多进程流式压盘，共 {len(projects)} 个项目...")

    # ==========================================
    # 并发调度区
    # ==========================================
    # 写入 Zarr 涉及强烈的 CPU 压缩计算 (zstd)，必须使用进程池 (ProcessPoolExecutor)
    # 根据你的 CPU 核心数，将 max_workers 设置为 4 到 16 之间为宜
    executor = BoundedExecutor(max_workers=4, executor_cls=cf.ProcessPoolExecutor)
    
    # 挂载生产与消费流
    generator = zarr_task_generator(projects, output_dir, STATIC_FIELDS, DYNAMIC_FIELDS)
    
    results = executor.execute_stream(
        task_source=generator, 
        total_est=len(projects), 
        desc="Zarr Dataset ETL"
    )
    
    print("\n🎉 全部数据集打包完成！以下是执行结果日志：")
    for res in results:
        print(f"  - {res}")