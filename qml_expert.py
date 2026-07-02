# ============================================
# qml_expert.py
# Tu experto en Quantum Machine Learning
# ============================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import json
from pathlib import Path

class QMLExpert:
    """Experto en Quantum Machine Learning basado en Qwen3-32B"""
    
    def __init__(self, base_model="Qwen/Qwen3-32B", adapter_path="./qwen3_qml_finetuned/lora_adapter"):
        self.base_model = base_model
        self.adapter_path = Path(adapter_path)
        
        print("🧠 Inicializando Experto en QML...")
        
        # Cargar configuración
        if self.adapter_path.exists():
            self.load_finetuned_model()
        else:
            print("⚠️ No se encontró adapter. Cargando modelo base...")
            self.load_base_model()
    
    def load_base_model(self):
        """Cargar modelo base sin fine-tuning"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("✅ Modelo base cargado")
    
    def load_finetuned_model(self):
        """Cargar modelo con fine-tuning en QML"""
        print(f"🔧 Cargando adapter desde: {self.adapter_path}")
        
        # Cargar configuración
        peft_config = PeftConfig.from_pretrained(self.adapter_path)
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Cargar modelo base
        self.model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Cargar adapter LoRA
        self.model = PeftModel.from_pretrained(
            self.model,
            self.adapter_path
        )
        
        print("✅ Experto en QML cargado")
    
    def ask(self, question: str, max_new_tokens: int = 512) -> str:
        """Hacer una pregunta al experto en QML"""
        
        # Formatear prompt
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenizar
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generar respuesta
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodificar respuesta
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraer solo la parte del asistente
        if "<|im_start|>assistant\n" in response:
            response = response.split("<|im_start|>assistant\n")[1]
        
        return response.strip()
    
    def interactive_session(self):
        """Sesión interactiva con el experto"""
        print("\n" + "="*60)
        print("🤖 EXPERTO EN QUANTUM MACHINE LEARNING")
        print("="*60)
        print("Pregunta sobre QML, circuitos cuánticos, modelos híbridos, etc.")
        print("Escribe 'salir' o 'exit' para terminar\n")
        
        while True:
            question = input("\n❓ Tu pregunta: ")
            
            if question.lower() in ['salir', 'exit', 'quit']:
                print("👋 ¡Hasta luego!")
                break
            
            print("\n💭 Generando respuesta...")
            answer = self.ask(question)
            print(f"\n🤖 {answer}")

# Ejecutar
if __name__ == "__main__":
    # Crear el experto
    expert = QMLExpert()
    
    # Preguntas de ejemplo
    test_questions = [
        "¿Qué es una Red Neuronal Cuántica (QNN)?",
        "¿Cómo se entrena un modelo híbrido cuántico-clásico?",
        "¿Cuál es la diferencia entre Qiskit y Pennylane?",
        "¿Qué aplicaciones tiene el Quantum Machine Learning en la industria?"
    ]
    
    print("🧪 Probando el experto...")
    for q in test_questions:
        print(f"\n❓ {q}")
        print(f"🤖 {expert.ask(q)}")
        print("-"*50)