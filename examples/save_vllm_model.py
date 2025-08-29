#  Copyright (c) 2025, TensorCast Team.

import argparse
import os
import shutil
from typing import Optional

os.environ["VLLM_USE_V1"] = "1"
os.environ["TRITON_LIBCUDA_PATH"] = "/usr/local/nvidia/lib64"


class VllmModelDownloader:
    def __init__(self):
        pass

    def download_vllm_model(
        self,
        artifact_name: str,
        torch_dtype: str,
        tensor_parallel_size: int = 1,
        storage_path: str = "./models",
        local_path: Optional[str] = None,
    ):
        import gc
        from tempfile import TemporaryDirectory

        import torch
        from huggingface_hub import snapshot_download
        from vllm import LLM

        # set the artifact storage path
        storage_path = os.getenv("STORAGE_PATH", storage_path)
        target_save_path = os.path.join(
            storage_path, f"{artifact_name}-tp-{tensor_parallel_size}"
        )

        def _run_writer(input_dir, artifact_name):
            # load models from the input directory
            llm_writer = LLM(
                artifact=input_dir,
                download_dir=input_dir,
                dtype=torch_dtype,
                tensor_parallel_size=tensor_parallel_size,
                num_gpu_blocks_override=1,
                enforce_eager=True,
                max_model_len=1,
            )
            llm_writer.llm_engine.collective_rpc(
                "save_sc_model_state", args=(target_save_path,)
            )
            for file in os.listdir(input_dir):
                # Copy the metadata files into the output directory
                if os.path.splitext(file)[1] not in (
                    ".bin",
                    ".pt",
                    ".safetensors",
                ):
                    src_path = os.path.join(input_dir, file)
                    dest_path = os.path.join(target_save_path, file)
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dest_path)
                    else:
                        shutil.copy(src_path, dest_path)
            del llm_writer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        try:
            with TemporaryDirectory() as cache_dir:
                input_dir = local_path
                # download from huggingface
                if local_path is None:
                    input_dir = snapshot_download(
                        artifact_name,
                        cache_dir=cache_dir,
                        allow_patterns=[
                            "*.safetensors",
                            "*.bin",
                            "*.json",
                            "*.txt",
                        ],
                    )
                _run_writer(input_dir, artifact_name)
        except Exception as e:
            print(f"An error occurred while saving the artifact: {e}")
            # remove the output dir
            shutil.rmtree(target_save_path)
            raise RuntimeError(
                f"Failed to save {artifact_name} for vllm backend: {e}"
            ) from e


parser = argparse.ArgumentParser(
    description="Save a artifact from HuggingFace artifact hub."
)
parser.add_argument(
    "--artifact-name",
    type=str,
    required=True,
    help="Artifact name from HuggingFace artifact hub.",
)
parser.add_argument(
    "--local-artifact-path",
    type=str,
    required=False,
    help="Local path to the artifact snapshot.",
)
parser.add_argument(
    "--storage-path",
    type=str,
    default="./models",
    help="Local path to save the artifact.",
)
parser.add_argument(
    "--tensor-parallel-size",
    type=int,
    default=1,
    help="Tensor parallel size.",
)

args = parser.parse_args()

artifact_name = args.artifact_name
local_path = args.local_path
storage_path = args.storage_path
tensor_parallel_size = args.tensor_parallel_size

downloader = VllmModelDownloader()
downloader.download_vllm_model(
    artifact_name,
    "auto",
    tensor_parallel_size=tensor_parallel_size,
    storage_path=storage_path,
    local_path=local_path,
)
