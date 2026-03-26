# data/rsim_pipeline.py
from abc import ABC, abstractmethod
import concurrent.futures as cf
from typing import Callable, Union

# 引入底层通用的有界流式执行器
from data.distributed import BoundedExecutor

# =============================================================================
# 领域抽象接口 (油藏数模标准流水线)
# =============================================================================
class COPY(ABC):
    @abstractmethod
    def copy(self):
        """将 template_dir 下的文件拷贝到 instance_dir"""
        raise NotImplementedError

class UPDATE(ABC):
    @abstractmethod
    def update(self):
        """在 instance_dir 下根据控制参数修改配置文件"""
        raise NotImplementedError

class WRITE(ABC):
    @abstractmethod
    def write(self):
        """在 instance_dir 下写入为标准数据或生成网格"""
        raise NotImplementedError

class RUN(ABC):
    @abstractmethod
    def run(self):
        """在 instance_dir 下运行外部模型或命令"""
        raise NotImplementedError

class READ(ABC):
    @abstractmethod
    def read(self):
        """在 instance_dir 下读取并解析模拟结果"""
        raise NotImplementedError


# =============================================================================
# 专属流水线管理器 (原 INSTANCE_MNGR)
# =============================================================================
class RsvSimPipelineMngr:
    """
    油藏数模实例专属管家：
    管理多个 INSTANCE，按领域专属顺序并行执行 copy -> update -> write -> run -> read。
    """
    def __init__(self, max_workers: int = 8, nprocess: int = 4, prog_bar: bool = True):
        self.max_workers = max_workers
        self.nprocess = nprocess
        self.prog_bar = prog_bar
        
        # 维持严格的先后顺序队列
        self._queue_map = {'copy': [], 'update': [], 'write': [], 'run': [], 'read': []}

        # 映射表：(执行器类型, 终端提示语)
        self._async_attrs = {
            'copy':   (cf.ThreadPoolExecutor, '复制配置文件...'),
            'update': (cf.ThreadPoolExecutor, '更新配置文件...'),
            'write':  (cf.ProcessPoolExecutor, '写入为标准数据...'),
            'run':    (cf.ThreadPoolExecutor, '运行模型...'),
            'read':   (cf.ProcessPoolExecutor, '读取为标准结果...')          
        }

    def contain(self, obj: Union[COPY, UPDATE, RUN, READ, WRITE]):
        """
        接收实现了一项或多项流水线接口的子类实例，自动分拣并入队。
        """
        if not isinstance(obj, (COPY, UPDATE, RUN, READ, WRITE)):
            raise TypeError("contain() 只能接收实现了 COPY, UPDATE, RUN, READ, WRITE 接口的实例")
        
        for method_name, method_list in self._queue_map.items():
            method = getattr(obj, method_name, None)
            if method: 
                method_list.append(method)

    def execute(self):
        """
        严格按顺序引爆各个阶段的并发队列
        """
        outputs = {}
        
        for method_name, method_list in self._queue_map.items():
            if not method_list:
                continue
                
            executor_cls, note = self._async_attrs[method_name]
            
            if self.prog_bar:
                print(note)
                
            # 智能分配并发度：多进程操作或外部 RUN 使用 nprocess，纯 I/O 使用 max_workers
            if executor_cls == cf.ProcessPoolExecutor or method_name == 'run':
                workers = self.nprocess
            else:
                workers = self.max_workers
                
            # 实例化底层核心引擎
            executor = BoundedExecutor(max_workers=workers, executor_cls=executor_cls)
            
            total_est = len(method_list) if self.prog_bar else None
            desc = method_name.capitalize()
            
            # 将方法列表直接作为可迭代对象送给执行引擎
            stage_results = executor.execute_stream(
                task_source=method_list, 
                total_est=total_est, 
                desc=desc
            )
            
            outputs[method_name] = stage_results
            
            # 清空已跑完的队列，释放内存
            self._queue_map[method_name] = []
            
        return outputs