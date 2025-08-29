#  Copyright (c) 2025, TensorCast Team.


import torch
from transformers import AutoModelForCausalLM

# from scstore.transformers_util import save_model

# parser = argparse.ArgumentParser(description="Save a artifact from HuggingFace artifact hub.")
# parser.add_argument(
#     "--artifact-name",
#     type=str,
#     required=True,
#     help="Artifact name from HuggingFace artifact hub.",
# )
# parser.add_argument(
#     "--storage-path",
#     type=str,
#     default="./models",
#     help="Local path to save the artifact.",
# )

# args = parser.parse_args()

# artifact_name = args.artifact_name
# storage_path = args.storage_path

hf_model_name = "Qwen/Qwen3-0.6B"
# Load a artifact from HuggingFace artifact hub.
artifact = AutoModelForCausalLM.from_pretrained(
    hf_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
)

state_dict = artifact.state_dict()
# target_path = os.path.join(storage_path, hf_model_name, "state_dict.pth")
# torch.save(state_dict, target_path)

# Save the artifact to the local path.
# disk_path = os.path.join(storage_path, hf_model_name)
# save_model(artifact, disk_path)
