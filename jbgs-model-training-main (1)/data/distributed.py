# data/distributed.py
import concurrent.futures as cf
from typing import Callable, Iterable, Type
from tqdm import tqdm
import time
import random

class BoundedExecutor:
    """
    通用流式并发执行器 (生产者-消费者 复合模型)
    接收一个任务生成器或迭代器，保持最大并发数，完成一个消费一个，并立刻补充一个。
    极大降低内存占用，有效防止 10 万+ 级别任务瞬间涌入导致 OOM。
    """
    def __init__(self, max_workers: int, executor_cls: Type = cf.ThreadPoolExecutor):
        self.max_workers = max_workers
        self.executor_cls = executor_cls

    def execute_stream(self, task_source: Iterable[Callable], total_est: int = None, desc: str = "Processing"):
        """
        执行流式任务。
        
        Args:
            task_source: 可以是 list，也可以是惰性求值的 generator (yield 任务)
            total_est: 任务总数预估（用于显示进度条，如果不传则无进度条）
            desc: 进度条前缀说明
        """
        results = []
        it = iter(task_source)
        
        # 只有在传入了预估总数时，才显示进度条
        pbar = tqdm(total=total_est, desc=desc) if total_est is not None else None
        
        with self.executor_cls(max_workers=self.max_workers) as executor:
            inflight = set()
            
            # 1. 预热缓冲区 (最多放入 max_workers * 2 个任务，防止初始内存撑爆)
            max_inflight = max(self.max_workers * 2, 1)
            
            while len(inflight) < max_inflight:
                try:
                    task = next(it)
                    inflight.add(executor.submit(task))
                except StopIteration:
                    break  # 初始任务少于缓冲池容量
                    
            # 2. 动态防闸主循环 (执行器核心：完成一个，补充一个)
            while inflight:
                # 阻塞等待，直到有任意一个任务完成
                done, inflight = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
                
                for fut in done:
                    try:
                        res = fut.result()
                        if res is not None:
                            results.append(res)
                    except Exception as e:
                        print(f"\n[BoundedExecutor] 任务执行出现异常: {e}")
                    finally:
                        if pbar is not None:
                            pbar.update(1)
                    
                    # 消费完一个，立刻从生成器提取一个新任务补充进池子
                    try:
                        next_task = next(it)
                        inflight.add(executor.submit(next_task))
                    except StopIteration:
                        pass  # 任务源已枯竭，靠剩余的 inflight 慢慢跑完即可
                        
        if pbar is not None:
            pbar.close()
            
        return results


# ==========================================
# 简单的本地使用与测试案例
# ==========================================
if __name__ == "__main__":
    
    # 模拟一个消耗 CPU 的简单任务
    def dummy_task(task_id):
        sleep_time = random.uniform(0.1, 0.5)
        time.sleep(sleep_time)
        return f"Task {task_id} done in {sleep_time:.2f}s"
        
    # 🌟 惰性任务发生器：内存占用永远为 O(1)，即便 yield 100 万次也不会爆内存
    def task_generator(num_tasks):
        for i in range(num_tasks):
            # 使用闭包或 lambda 延迟绑定参数
            yield lambda tid=i: dummy_task(tid)

    total_tasks = 20
    print(f"🚀 开始测试 BoundedExecutor (共 {total_tasks} 个流式任务)...")
    
    # 实例化我们的有界执行器 (使用线程池)
    executor = BoundedExecutor(max_workers=4, executor_cls=cf.ThreadPoolExecutor)
    
    # 将生成器送入执行引擎
    start_t = time.time()
    results = executor.execute_stream(
        task_source=task_generator(total_tasks), 
        total_est=total_tasks, 
        desc="Dummy Tasks"
    )
    
    print(f"✅ 测试完成！耗时 {time.time() - start_t:.2f}s，共收集到 {len(results)} 个结果。")
    print(f"抽样展示前 3 个结果: {results[:3]}")