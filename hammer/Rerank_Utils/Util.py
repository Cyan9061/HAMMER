from hammer.Rerank_Utils.PredefinedDatasets import HF_PRE_DEFIND_DATASET
import pandas as pd
from prettytable import PrettyTable
def load_knobs(filename):
    # Load knobs from a text file
    with open(filename, 'r') as f:
        knobs = [line.strip() for line in f.readlines()]
        return knobs

def get_datasets_info():
    table = PrettyTable(['Retriever', 'Dataset', 'Original ext', 'Compressed','Desc','URL'])
    for retriever, datasets in HF_PRE_DEFIND_DATASET.items():
        for dataset_name, dataset_info in datasets.items():
            
            flattened_entry = {
                'retriever': retriever,
                'dataset': dataset_name,
                'original_ext': dataset_info.get('original_ext'),
                'compressed': dataset_info.get('compressed'),
                'desc': dataset_info.get('desc'),
                'url': dataset_info.get('url')
            }
            table.add_row(flattened_entry.values())
            
    print(table)
from typing import Union, List, Optional, Tuple
import torch

def get_device(
        device: Optional[Union[str, torch.device]],
        no_mps: bool = False,
    ) -> Union[str, torch.device]:
        if not device:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available() and not no_mps:
                device = "mps"
            else:
                device = "cpu"
        return device

def get_dtype(
        dtype: Optional[Union[str, torch.dtype]],
        device: Optional[Union[str, torch.device]],
        verbose: int = 1,
    ) -> torch.dtype:
        if dtype is None:
            print("No dtype set")
        if device == "cpu":
            dtype = torch.float32
        if not isinstance(dtype, torch.dtype):
            if dtype == "fp16" or "float16":
                dtype = torch.float16
            elif dtype == "bf16" or "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        return dtype    