import torch
import gc
import types
from typing import Optional, Any, List
from hammer.logger import logger

class CUDACleaner:
    """
    CUDA资源清理器
    专门用于清理flow中的embedding模型和释放GPU显存
    """
    
    def __init__(self, device_id: int = 0):
        """
        初始化清理器
        
        Args:
            device_id: GPU设备ID
        """
        self.device_id = device_id
        self.visited_objects = set()
    
    def _delete_embedding_models_from_object(self, obj: Any, visited: Optional[set] = None) -> None:
        """
        深度递归遍历对象，找到所有内嵌的embedding模型并彻底删除它们
        
        Args:
            obj: 需要清理的对象
            visited: 防止无限递归的集合
        """
        if visited is None:
            visited = set()
        
        obj_id = id(obj)
        if obj_id in visited or isinstance(obj, (str, int, float, bool, types.FunctionType)):
            return
        visited.add(obj_id)
        
        # 检查是否是embedding模型封装类
        type_name = str(type(obj)).lower()
        if "embedding" in type_name:
            pytorch_model = obj._model
            
            if pytorch_model and hasattr(pytorch_model, 'device') and 'cuda' in str(pytorch_model.device):
                try:
                    # 1. 将模型移动到CPU
                    pytorch_model.to('cpu')
                    
                    # 2. 清理所有梯度
                    pytorch_model.zero_grad()
                    for param in pytorch_model.parameters():
                        if param.grad is not None:
                            param.grad.detach_()
                            param.grad.zero_()
                    
                    # 3. 删除模型引用
                    del obj._model
                    obj._model = None
                    
                    # 4. 强制垃圾回收
                    logger.info(f"✅ 成功删除内嵌模型 {type(pytorch_model).__name__}")

                    del pytorch_model
                    gc.collect()
                    torch.cuda.empty_cache()

                except Exception as e:
                    logger.warning(f"⚠️ 删除内嵌模型时出错: {e}")
                return
        
        # 递归遍历对象属性
        if isinstance(obj, (list, tuple)):
            for item in obj:
                self._delete_embedding_models_from_object(item, visited)
        elif isinstance(obj, dict):
            for value in obj.values():
                self._delete_embedding_models_from_object(value, visited)
        elif hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                self._delete_embedding_models_from_object(attr_value, visited)
    
    def _get_memory_info(self) -> tuple:
        """
        获取当前GPU显存使用情况
        
        Returns:
            (allocated_mb, reserved_mb): 已分配和已保留的显存（MB）
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device_id) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device_id) / 1024**2
            return allocated, reserved
        return 0.0, 0.0
    
    def _force_cleanup(self, iterations: int = 3) -> None:
        """
        强制清理GPU显存
        
        Args:
            iterations: 清理迭代次数
        """
        try:
            torch.cuda.set_device(self.device_id)
            for _ in range(iterations):
                gc.collect()
                torch.cuda.empty_cache()
            logger.debug(f"🧹 已强制清理GPU {self.device_id}的显存")
        except Exception as e:
            logger.warning(f"⚠️ 强制清理CUDA内存时出错: {e}")
    
    def cleanup_flow(self, flow: Any, aggressive: bool = False) -> dict:
        """
        清理flow中的embedding模型
        
        Args:
            flow: flow对象
            aggressive: 是否使用激进清理模式
            
        Returns:
            清理结果信息
        """
        logger.info("🧹 开始清理flow中的embedding模型...")
        
        # 记录清理前的显存使用情况
        before_allocated, before_reserved = self._get_memory_info()
        logger.info(f"清理前 GPU {self.device_id}: Allocated = {before_allocated:.2f} MB | Reserved = {before_reserved:.2f} MB")
        
        # 删除flow中的embedding模型
        if flow is not None:
            self._delete_embedding_models_from_object(flow)
        
        # 清理显存
        if aggressive:
            self._force_cleanup(iterations=5)
        else:
            self._force_cleanup(iterations=3)
        
        # 记录清理后的显存使用情况
        after_allocated, after_reserved = self._get_memory_info()
        freed_allocated = before_allocated - after_allocated
        freed_reserved = before_reserved - after_reserved
        
        result = {
            "before_allocated": before_allocated,
            "before_reserved": before_reserved,
            "after_allocated": after_allocated,
            "after_reserved": after_reserved,
            "freed_allocated": freed_allocated,
            "freed_reserved": freed_reserved,
            "device_id": self.device_id
        }
        
        logger.info(f"清理后 GPU {self.device_id}: Allocated = {after_allocated:.2f} MB | Reserved = {after_reserved:.2f} MB")
        logger.info(f"释放显存: Allocated = {freed_allocated:.2f} MB | Reserved = {freed_reserved:.2f} MB")
        
        return result
    
    def cleanup_and_delete_flow(self, flow: Any, aggressive: bool = False) -> dict:
        """
        清理flow并删除flow对象
        
        Args:
            flow: flow对象
            aggressive: 是否使用激进清理模式
            
        Returns:
            清理结果信息
        """
        try:
            # 先清理flow中的模型
            result = self.cleanup_flow(flow, aggressive)
            
            # 删除flow对象
            del flow
            
            # 再次强制清理
            self._force_cleanup(iterations=3)
            
            logger.info("✅ flow对象已删除并完成清理")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ 删除flow对象时出错: {e}")
            return {"error": str(e)}
    
    def cleanup_multiple_objects(self, objects: List[Any], aggressive: bool = False) -> dict:
        """
        清理多个对象
        
        Args:
            objects: 对象列表
            aggressive: 是否使用激进清理模式
            
        Returns:
            清理结果信息
        """
        logger.info(f"🧹 开始清理 {len(objects)} 个对象...")
        
        before_allocated, before_reserved = self._get_memory_info()
        
        for i, obj in enumerate(objects):
            if obj is not None:
                logger.debug(f"清理对象 {i+1}/{len(objects)}")
                self._delete_embedding_models_from_object(obj)
        
        if aggressive:
            self._force_cleanup(iterations=5)
        else:
            self._force_cleanup(iterations=3)
        
        after_allocated, after_reserved = self._get_memory_info()
        freed_allocated = before_allocated - after_allocated
        freed_reserved = before_reserved - after_reserved
        
        result = {
            "objects_cleaned": len(objects),
            "before_allocated": before_allocated,
            "before_reserved": before_reserved,
            "after_allocated": after_allocated,
            "after_reserved": after_reserved,
            "freed_allocated": freed_allocated,
            "freed_reserved": freed_reserved,
            "device_id": self.device_id
        }
        
        logger.info(f"批量清理完成: 释放显存 Allocated = {freed_allocated:.2f} MB | Reserved = {freed_reserved:.2f} MB")
        return result

# 便捷函数
def cleanup_cuda_resources(flow=None, additional_objects=None, device_id=0, aggressive=False):
    """
    便捷函数，保持与原函数相同的接口
    
    Args:
        flow: flow对象
        additional_objects: 额外的对象列表
        device_id: GPU设备ID
        aggressive: 是否使用激进清理模式
    """
    cleaner = CUDACleaner(device_id)
    
    if flow is not None:
        cleaner.cleanup_flow(flow, aggressive)
    
    if additional_objects:
        cleaner.cleanup_multiple_objects(additional_objects, aggressive)

# 使用示例
if __name__ == "__main__":
    # 创建清理器
    cleaner = CUDACleaner(device_id=0)
    
    # 使用示例
    # flow = your_flow_object
    # result = cleaner.cleanup_and_delete_flow(flow, aggressive=True)
    # print(result)
    pass 