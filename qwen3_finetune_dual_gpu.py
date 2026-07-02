import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
print("🚀 Iniciando fine-tuning de Qwen3-32B...")
print(f"PyTorch CUDA disponible: {torch.cuda.is_available()}")
print(f"GPUs disponibles: {torch.cuda.device_count()}")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# ==========================================
# RUTAS - CON TU MODELO DESCARGADO
# ==========================================
MODEL_PATH = "/home/jesus.amontes/experimento_lusitania_QNN/finetune_qwen_alex/models/Qwen3-32B-GGUF/Qwen3-32B-GGUF"
DATASET_PATH = "./processed_data/training_data.jsonl"
OUTPUT_DIR = "./qwen3_qml_finetuned"

# ==========================================
# CUANTIZACIÓN 4-BIT (para que quepa en GPU)
# ==========================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ==========================================
# CARGAR MODELO Y TOKENIZER
# ==========================================
print(f"📥 Cargando modelo desde: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print(f"✅ Modelo cargado en: {model.device}")

# ==========================================
# PREPARAR PARA ENTRENAMIENTO
# ==========================================
model = prepare_model_for_kbit_training(model)

# ==========================================
# CONFIGURAR LORA
# ==========================================
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================================
# CARGAR DATASET
# ==========================================
print(f"📊 Cargando dataset desde: {DATASET_PATH}")

# Verificar que el dataset existe
if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: No se encuentra el dataset en {DATASET_PATH}")
    print("Ejecuta primero: python arxiv_qml_scraper.py y python data_processor.py")
    exit(1)

dataset = load_dataset("json", data_files=DATASET_PATH)

def format_instruction(example):
    return {
        "text": f"<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
    }

dataset = dataset.map(format_instruction)
print(f"✅ Dataset cargado: {len(dataset['train'])} ejemplos")

# ==========================================
# CONFIGURAR ENTRENAMIENTO
# ==========================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.06,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    bf16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
    report_to="none",
)

# ==========================================
# ENTRENAR
# ==========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    tokenizer=tokenizer,
)

print("🔥 Iniciando entrenamiento...")
print("="*60)
trainer.train()

# ==========================================
# GUARDAR MODELO
# ==========================================
print("💾 Guardando modelo...")
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
print("✅ Fine-tuning completado")
print(f"📁 Modelo guardado en: {OUTPUT_DIR}")