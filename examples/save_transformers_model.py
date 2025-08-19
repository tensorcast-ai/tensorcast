#  Copyright (c) 2025, StepCast Team.


import torch
from transformers import AutoModelForCausalLM

# from scstore.transformers_util import save_model

# parser = argparse.ArgumentParser(description="Save a model from HuggingFace model hub.")
# parser.add_argument(
#     "--model-name",
#     type=str,
#     required=True,
#     help="Model name from HuggingFace model hub.",
# )
# parser.add_argument(
#     "--storage-path",
#     type=str,
#     default="./models",
#     help="Local path to save the model.",
# )

# args = parser.parse_args()

# model_name = args.model_name
# storage_path = args.storage_path

hf_model_name = "Qwen/Qwen3-0.6B"
# Load a model from HuggingFace model hub.
model = AutoModelForCausalLM.from_pretrained(
    hf_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
)

state_dict = model.state_dict()
# target_path = os.path.join(storage_path, hf_model_name, "state_dict.pth")
# torch.save(state_dict, target_path)

# Save the model to the local path.
# model_path = os.path.join(storage_path, hf_model_name)
# save_model(model, model_path)
