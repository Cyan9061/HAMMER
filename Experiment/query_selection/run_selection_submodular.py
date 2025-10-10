#Experiment/query_selection/run_selection.py
import json
from types import SimpleNamespace
from Submodular import SubmodularSelector

def main():
    """
    主执行函数，完成数据集加载、Coreset选择和结果保存的整个流程。
    """
    # --- 1. 定义文件路径和参数 ---
    input_json_path = 'docs/2wikimultihopqa_backup.json'
    output_json_path = 'docs/2wikimultihopqa_selection_submodular.json'
    num_total_samples = 800  # 从源数据集中加载的样本总数
    num_queries_to_select = 200 # 希望最终选出的Coreset大小
    
    # BAAI/bge-large-en-v1.5 是一个强大的、公开可用的模型，与您代码中指定的模型相对应
    # sentence-transformers库会自动从Hugging Face Hub下载并缓存它
    model_name = 'BAAI/bge-large-en-v1.5'
    
    # lamda=0.5 表示多样性(diversity)和代表性(representativeness)各占50%的权重
    lamda_value = 0.5

    print("--- Coreset选择流程开始 ---")
    print(f"输入文件: {input_json_path}")
    print(f"输出文件: {output_json_path}")
    print(f"将从 {num_total_samples} 条数据中选择 {num_queries_to_select} 条。")
    print(f"使用模型: {model_name}")
    print(f"Lamda (多样性/代表性平衡参数): {lamda_value}")
    print("-" * 20)

    # --- 2. 加载并准备数据集 ---
    print(f"正在加载数据集: {input_json_path}...")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到，请确保 '{input_json_path}' 路径正确。")
        return
    except json.JSONDecodeError:
        print(f"错误: '{input_json_path}' 文件格式不正确，无法解析JSON。")
        return

    # 截取前800条数据
    data_to_process = all_data[:num_total_samples]
    
    # SubmodularSelector类期望输入的数据点可以通过 .question 访问其问题字段。
    # 我们使用 SimpleNamespace 将字典列表转换为对象列表，以匹配代码要求。
    data_objects = [SimpleNamespace(**d) for d in data_to_process]
    print(f"已加载并准备好 {len(data_objects)} 条数据。")
    print("-" * 20)

    # --- 3. 初始化并执行Coreset选择算法 ---
    # 移除原脚本中的pdb.set_trace()以确保流畅运行
    # 如果Submodular.py中有pdb.set_trace()，请手动注释或删除它们
    print("正在初始化SubmodularSelector...")
    selector = SubmodularSelector(model=model_name, lamda=lamda_value)
    
    print(f"开始执行 select_optimized_queries 以选择 {num_queries_to_select} 条数据...")
    # 执行核心算法
    selected_data_points = selector.select_optimized_queries(
        dataset=data_objects,
        num_queries_to_select=num_queries_to_select
    )
    
    if not selected_data_points:
        print("错误: Coreset选择未能返回任何数据点。流程终止。")
        return
        
    print(f"Coreset选择完成，成功选出 {len(selected_data_points)} 条数据。")
    print("-" * 20)

    # --- 4. 整理并保存结果 ---
    # 将选出的对象列表转换回字典列表，以便保存为JSON
    selected_dicts = [vars(p) for p in selected_data_points]
    
    print(f"正在将选出的 {len(selected_dicts)} 条数据保存到 {output_json_path}...")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(selected_dicts, f, indent=4, ensure_ascii=False)
        print("保存成功！")
    except Exception as e:
        print(f"错误: 保存文件时发生错误: {e}")

    print("--- Coreset选择流程结束 ---")

if __name__ == '__main__':
    main()