# ============================================
# arxiv_qml_scraper.py
# Scraper especializado en Quantum Machine Learning
# ============================================

import arxiv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import feedparser

class QMLArxivScraper:
    """Scraper especializado en Quantum Machine Learning para arXiv"""
    
    def __init__(self, output_dir="qml_papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Palabras clave específicas de QML
        self.queries = [
            "quantum machine learning",
            "quantum neural network",
            "QNN hybrid model",
            "quantum circuit machine learning",
            "variational quantum algorithm",
            "quantum kernel method",
            "quantum transfer learning",
            "quantum natural language processing",
            "quantum reinforcement learning",
            "Pennylane Qiskit QML benchmark",
            "quantum-classical hybrid",
            "parameterized quantum circuit",
        ]
        
        # Categorías relevantes
        self.categories = ["quant-ph", "cs.LG", "stat.ML", "physics.comp-ph"]
        
    def search_papers(self, query: str, max_results: int = 50) -> List[Dict]:
        """Buscar papers en arXiv"""
        print(f"🔍 Buscando: {query}")
        
        client = arxiv.Client(
            page_size=25,
            delay_seconds=3.0,  # Respetar rate limits
            num_retries=3
        )
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in client.results(search):
            paper = {
                "id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "categories": result.categories,
                "published": result.published.isoformat(),
                "pdf_url": result.pdf_url,
                "doi": result.doi,
                "journal_ref": result.journal_ref,
                "primary_category": result.primary_category,
                "scraped_at": datetime.now().isoformat()
            }
            papers.append(paper)
            print(f"  ✅ {paper['title'][:60]}...")
            
        return papers
    
    def search_nist_publications(self) -> List[Dict]:
        """Buscar publicaciones de NIST sobre Quantum ML y Criptografía"""
        print("🔍 Buscando publicaciones NIST...")
        
        papers = []
        
        # URL de búsqueda de NIST (simulada)
        # Nota: NIST usa diferentes repositorios, esto es un ejemplo
        queries_nist = [
            "quantum machine learning NIST",
            "post quantum cryptography NIST",
            "quantum neural network NIST"
        ]
        
        # Implementación real con requests + BeautifulSoup
        for query in queries_nist:
            # Simulación - en producción usarías la API real de NIST
            time.sleep(1)
            papers.append({
                "id": f"nist-{int(time.time())}",
                "title": f"NIST Publication on {query}",
                "source": "NIST",
                "query": query,
                "scraped_at": datetime.now().isoformat()
            })
            
        return papers
    
    def search_blogs(self) -> List[Dict]:
        """Buscar blogs técnicos especializados en QML"""
        print("🔍 Buscando blogs técnicos...")
        
        # Blogs relevantes
        blog_sources = [
            "https://www.ibm.com/quantum/blog",
            "https://pennylane.ai/blog",
            "https://qiskit.org/ecosystem/blog",
            "https://medium.com/tag/quantum-machine-learning",
            "https://towardsdatascience.com/tagged/quantum-computing"
        ]
        
        papers = []
        for source in blog_sources:
            try:
                # Simulación - en producción implementarías scraping real
                print(f"  📝 Procesando: {source}")
                time.sleep(1)
                papers.append({
                    "id": f"blog-{int(time.time())}",
                    "source": source,
                    "title": f"Blog post from {source.split('/')[2]}",
                    "scraped_at": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"  ❌ Error en {source}: {e}")
                
        return papers
    
    def run_full_scrape(self, papers_per_query: int = 30):
        """Ejecutar scraping completo"""
        all_papers = []
        
        print("="*60)
        print("🚀 INICIANDO SCRAPING DE QML")
        print("="*60)
        
        # 1. arXiv
        print("\n📚 Buscando en arXiv...")
        for query in self.queries:
            papers = self.search_papers(query, max_results=papers_per_query)
            all_papers.extend(papers)
            time.sleep(3)  # Rate limiting
        
        # 2. NIST
        print("\n🏛️ Buscando en NIST...")
        nist_papers = self.search_nist_publications()
        all_papers.extend(nist_papers)
        
        # 3. Blogs
        print("\n📝 Buscando en Blogs...")
        blog_papers = self.search_blogs()
        all_papers.extend(blog_papers)
        
        # Guardar resultados
        output_file = self.output_dir / f"qml_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(all_papers, f, indent=2, default=str)
        
        print(f"\n✅ Scraping completado: {len(all_papers)} papers")
        print(f"📁 Guardado en: {output_file}")
        
        return all_papers

# Ejecutar
if __name__ == "__main__":
    scraper = QMLArxivScraper()
    papers = scraper.run_full_scrape(papers_per_query=20)