from optimum.onnxruntime.modeling_ort import ORTModelForCausalLM
from transformers import AutoTokenizer

MODEL_PATH = "./models/TinyLlama-1.1B-Chat-v1.0"
SAVE_DIR = "./tinyllama_onnx/"

ort_model = ORTModelForCausalLM.from_pretrained(
    MODEL_PATH,
    export=True,
    provider="CPUExecutionProvider"
)

ort_model.save_pretrained(SAVE_DIR)

print("Saved ONNX model in:", SAVE_DIR)
