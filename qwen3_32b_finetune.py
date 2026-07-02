# ============================================
# qwen3_32b_finetune.py
# Fine-tuning de Qwen3-32B con LoRA para QML
# ============================================

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json
from pathlib import Path
import os

class Qwen3Finetuner:
    """Fine-tuning de Qwen3-32B con LoRA para QML"""
    
    def __init__(self, model_name="Qwen/Qwen3-32B", output_dir="./qwen3_qml_finetuned"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configuración de memoria
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
    def load_model_and_tokenizer(self):
        """Cargar modelo y tokenizer optimizados para GPU"""
        print(f"🔄 Cargando modelo: {self.model_name}")
        
        # Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Cargar modelo con quantización
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"  # Aceleración
        )
        
        # Preparar para entrenamiento con k-bit
        model = prepare_model_for_kbit_training(model)
        
        print(f"✅ Modelo cargado en: {model.device}")
        return model, tokenizer
    
    def setup_lora(self, model):
        """Configurar LoRA para el modelo"""
        print("🔧 Configurando LoRA...")
        
        lora_config = LoraConfig(
            r=64,  # Rango de adaptación
            lora_alpha=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            fan_in_fan_out=False
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def prepare_dataset(self, dataset_path: str):
        """Preparar dataset para fine-tuning"""
        print(f"📊 Cargando dataset: {dataset_path}")
        
        # Cargar dataset desde JSONL
        dataset = load_dataset("json", data_files=dataset_path)
        
        # Formatear para chat template
        def format_instruction(example):
            return {
                "text": f"<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
            }
        
        dataset = dataset.map(format_instruction)
        
        print(f"✅ Dataset preparado: {len(dataset['train'])} ejemplos")
        return dataset['train']
    
    def train(self, dataset_path: str, epochs: int = 3):
        """Ejecutar fine-tuning"""
        print("🚀 Iniciando fine-tuning...")
        
        # Cargar modelo
        model, tokenizer = self.load_model_and_tokenizer()
        model = self.setup_lora(model)
        
        # Preparar dataset
        dataset = self.prepare_dataset(dataset_path)
        
        # Tokenizar
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=2048,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Configuración de entrenamiento
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            warmup_ratio=0.06,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            optim="adamw_8bit",
            logging_steps=10,
            save_steps=200,
            save_total_limit=3,
            bf16=True,
            dataloader_num_workers=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        
        # Entrenar
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )
        
        print("🔥 Comenzando entrenamiento...")
        trainer.train()
        
        # Guardar modelo
        print("💾 Guardando modelo...")
        model.save_pretrained(self.output_dir / "lora_adapter")
        tokenizer.save_pretrained(self.output_dir / "lora_adapter")
        
        print(f"✅ Fine-tuning completado. Modelo guardado en: {self.output_dir}")

# Ejecutar
if __name__ == "__main__":
    # Configurar variables de entorno para GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Ajusta según tus GPUs
    
    # Crear finetuner
    finetuner = Qwen3Finetuner()
    
    # Ejecutar fine-tuning
    finetuner.train(
        dataset_path="./processed_data/training_data.jsonl",
        epochs=3
    )