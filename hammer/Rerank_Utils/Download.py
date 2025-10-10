import os
import requests
from hammer.Rerank_Utils.PredefinedDatasets import HF_PRE_DEFIND_DATASET
from tqdm import tqdm

class DownloadManger:
    @staticmethod
    def download(retriever: str, dataset: str, force_download: bool = True) ->str:
        if retriever not in HF_PRE_DEFIND_DATASET:
            raise FileNotFoundError(f"Retriever {retriever} Not Supported yet. Please choose another retriever.\nCheck Dataset.available_dataset()")
        if dataset not in HF_PRE_DEFIND_DATASET[retriever]:
            raise FileNotFoundError(f"Dataset {dataset} Not Supported yet. Please choose another dataset.\nCheck Dataset.available_dataset()")

        filename = HF_PRE_DEFIND_DATASET[retriever][dataset]['filename']
        if '-' in dataset:
            dataset_name, dataset_split = dataset.split('-', 1)
        else:
            dataset_name = dataset
        urls = HF_PRE_DEFIND_DATASET[retriever][dataset]['url']
        path = os.path.join(os.environ['RERANKING_CACHE_DIR'], 'dataset', retriever, dataset_name)
        file_path = os.path.join(path, filename)

        # If force_download is False and file already exists, skip downloading
        if not force_download and os.path.exists(file_path):
            print(f"File {file_path} already exists. Skipping download.")
            return file_path

        os.makedirs(path, exist_ok=True)

        for url in urls:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                with open(file_path, 'wb') as file, tqdm(
                    desc=f"Downloading {retriever} {dataset_name} {filename}",
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    total=total_size,
                ) as bar:
                    # Update progress bar while streaming chunks
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:  # Filter out keep-alive chunks
                            file.write(chunk)
                            bar.update(len(chunk))
                
                return file_path
            else:
                raise Exception(f'Failed to download the file from {url}')