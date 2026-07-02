# ============================================
# data_processor.py
# Limpieza y transformación a formato de fine-tuning
# ============================================

import json
import re
from pathlib import Path
from typing import List, Dict
import pandas as pd
from arxiv_qml_scraper import QMLArxivScraper
class QMLDataProcessor:
    """Procesa papers para formato de fine-tuning"""
    
    def __init__(self, input_dir="qml_papers"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path("processed_data")
        self.output_dir.mkdir(exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """Limpiar texto de caracteres especiales y formatos"""
        # Eliminar referencias a figuras/tablas
        text = re.sub(r'Figure \d+', '', text)
        text = re.sub(r'Table \d+', '', text)
        text = re.sub(r'Fig\. \d+', '', text)
        
        # Eliminar URLs
        text = re.sub(r'http\S+', '', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def generate_instruction_format(self, paper: Dict) -> Dict:
        """Generar formato de instrucción para fine-tuning"""
        
        # Crear múltiples pares instrucción-respuesta por paper
        prompts = []
        
        # 1. Resumen ejecutivo
        prompts.append({
            "instruction": f"Resume el paper '{paper.get('title', 'Título')}' en español, enfocándote en los aspectos clave de Quantum Machine Learning.",
            "input": f"Autores: {', '.join(paper.get('authors', [''])[:3])}\nCategorías: {', '.join(paper.get('categories', []))}",
            "output": paper.get('abstract', '')[:1000]  # Abstract truncado
        })
        
        # 2. Preguntas específicas de QML
        if "quantum" in paper.get('title', '').lower() or "quantum" in paper.get('abstract', '').lower():
            prompts.append({
                "instruction": "¿Cuál es el enfoque de Quantum Machine Learning utilizado en este paper?",
                "input": f"Título: {paper.get('title', '')}",
                "output": f"Este paper utiliza técnicas de Quantum Machine Learning en el contexto de {paper.get('primary_category', 'investigación cuántica')}. El enfoque principal es {paper.get('abstract', '')[:500]}"
            })
        
        # 3. Aplicación práctica
        prompts.append({
            "instruction": "¿Qué aplicación práctica tiene este paper en el campo de QML?",
            "input": f"Paper: {paper.get('title', '')}",
            "output": f"Las aplicaciones prácticas incluyen {paper.get('abstract', '')[:300]}..."
        })
        
        return {
            "source_id": paper.get('id', 'unknown'),
            "prompts": prompts
        }
    
    def create_training_dataset(self, papers: List[Dict]) -> pd.DataFrame:
        """Crear dataset para fine-tuning"""
        training_data = []
        
        for paper in papers:
            processed = self.generate_instruction_format(paper)
            for prompt in processed['prompts']:
                training_data.append({
                    "instruction": prompt['instruction'],
                    "input": prompt['input'],
                    "output": prompt['output'],
                    "source": paper.get('source', 'arxiv'),
                    "category": paper.get('primary_category', 'unknown')
                })
        
        # Guardar en formatos útiles
        df = pd.DataFrame(training_data)
        
        # Guardar como JSONL para fine-tuning
        jsonl_path = self.output_dir / "training_data.jsonl"
        with open(jsonl_path, 'w') as f:
            for _, row in df.iterrows():
                json.dump(row.to_dict(), f)
                f.write('\n')
        
        # Guardar como CSV
        csv_path = self.output_dir / "training_data.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Dataset creado: {len(df)} ejemplos")
        print(f"📁 Guardado en: {self.output_dir}")
        
        return df

# Ejecutar
if __name__ == "__main__":
    # Cargar papers scrapeados
    scraper = QMLArxivScraper()
    papers = scraper.run_full_scrape(papers_per_query=15)
    
    # Procesar para fine-tuning
    processor = QMLDataProcessor()
    df = processor.create_training_dataset(papers)
    
    print("\n📊 Muestra del dataset:")
    print(df.head())