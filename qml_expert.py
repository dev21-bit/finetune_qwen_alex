# qml_expert_dual_gpu.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class QMLExpert:
    def __init__(self, base_model_path, adapter_path):
        print("🧠 Inicializando Experto QML con 2 GPUs...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Cargar modelo base
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",  # Distribuye entre GPUs
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Cargar adapter LoRA
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        print("✅ Modelo cargado en ambas GPUs")
    
    def ask(self, question, max_tokens=512):
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("<|im_start|>assistant\n")[1].strip()

# Usar
expert = QMLExpert(
    base_model_path="/home/jesus.amontes/models/Qwen3-32B",
    adapter_path="./qwen3_qml_finetuned/lora_adapter"
)

# Probar
print("🤖 Experto en Quantum Machine Learning")
print("Pregunta sobre QML (escribe 'salir' para terminar)")

while True:
    q = input("\n❓ Tu pregunta: ")
    if q.lower() in ['salir', 'exit', 'quit']:
        break
    print(f"\n🤖 {expert.ask(q)}")