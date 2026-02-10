#!/usr/bin/env python3
"""
IA Multifuncional Avançada - NeuroGenesis AI
Framework modular com capacidades multimodais
Autor: Sistema Auto-Evolutivo
Versão: 2.0.0
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração avançada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neurogenesis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NeuroGenesisConfig:
    """Configuração avançada do sistema"""
    def __init__(self):
        self.device = self._setup_device()
        self.models = {
            'llm': 'microsoft/phi-2',  # Modelo base eficiente
            'code': 'Salesforce/codegen-16B-multi',
            'image': 'stabilityai/stable-diffusion-2-1',
            'audio': 'facebook/wav2vec2-large-960h-lv60-self',
            'video': 'damo-vilab/text-to-video-ms-1.7b',
            'multimodal': 'openai/clip-vit-large-patch14'
        }
        
        # Configurações de memória
        self.memory_path = Path("./neuro_memory/")
        self.knowledge_base = self.memory_path / "knowledge.json"
        self.experience_buffer = self.memory_path / "experience_buffer.npy"
        
        # Auto-aprendizado
        self.learning_rate = 0.001
        self.reinforcement_learning = True
        self.self_play_enabled = True
        self.curriculum_learning = True
        
        # Inicialização
        self._setup_directories()
        
    def _setup_device(self) -> torch.device:
        """Configura o melhor dispositivo disponível"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _setup_directories(self):
        """Cria estrutura de diretórios"""
        self.memory_path.mkdir(exist_ok=True)
        (self.memory_path / "models").mkdir(exist_ok=True)
        (self.memory_path / "temp").mkdir(exist_ok=True)
        (self.memory_path / "logs").mkdir(exist_ok=True)

class SelfLearningModule:
    """Módulo de auto-aprendizagem aprimorada"""
    def __init__(self, config: NeuroGenesisConfig):
        self.config = config
        self.experience_memory = []
        self.knowledge_graph = {}
        self.load_knowledge()
        
    def learn_from_experience(self, state, action, reward, next_state):
        """Aprendizado por reforço profundo"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'timestamp': np.datetime64('now')
        }
        self.experience_memory.append(experience)
        
        # Replay buffer para aprendizado offline
        if len(self.experience_memory) > 1000:
            self._experience_replay()
            
        # Atualização do conhecimento
        self._update_knowledge_graph(state, action, reward)
        
    def _experience_replay(self):
        """Repetição de experiência para aprendizado estável"""
        import random
        batch = random.sample(self.experience_memory, min(64, len(self.experience_memory)))
        
        # Aqui implementaria DQN ou A3C
        for exp in batch:
            # Implementação do algoritmo de aprendizado
            pass
    
    def _update_knowledge_graph(self, state, action, reward):
        """Atualiza grafo de conhecimento"""
        key = f"{hash(str(state))}_{hash(str(action))}"
        if key not in self.knowledge_graph:
            self.knowledge_graph[key] = {
                'count': 0,
                'total_reward': 0,
                'success_rate': 0
            }
        
        self.knowledge_graph[key]['count'] += 1
        self.knowledge_graph[key]['total_reward'] += reward
        self.knowledge_graph[key]['success_rate'] = (
            self.knowledge_graph[key]['total_reward'] / 
            self.knowledge_graph[key]['count']
        )
        
    def save_knowledge(self):
        """Salva conhecimento adquirido"""
        with open(self.config.knowledge_base, 'w') as f:
            json.dump(self.knowledge_graph, f, indent=2)
            
    def load_knowledge(self):
        """Carrega conhecimento existente"""
        if self.config.knowledge_base.exists():
            with open(self.config.knowledge_base, 'r') as f:
                self.knowledge_graph = json.load(f)

class CodeGenerationModule:
    """Módulo de geração de código em múltiplas linguagens"""
    def __init__(self, config: NeuroGenesisConfig):
        self.config = config
        self.supported_languages = [
            'python', 'javascript', 'typescript', 'java', 'c++',
            'c#', 'go', 'rust', 'swift', 'kotlin', 'ruby', 'php',
            'html', 'css', 'sql', 'bash', 'r', 'matlab'
        ]
        
    def generate_code(self, prompt: str, language: str = 'python', 
                     context: Optional[Dict] = None) -> Dict:
        """Gera código em linguagem específica"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        try:
            model_name = self.config.models['code']
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.config.device.type == 'cuda' else torch.float32,
                device_map="auto"
            )
            
            # Template para geração de código
            full_prompt = f"# Language: {language}\n# Description: {prompt}\n\n"
            
            inputs = tokenizer(full_prompt, return_tensors="pt").to(self.config.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=1024,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            code = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Análise do código gerado
            analysis = self._analyze_code(code, language)
            
            return {
                'success': True,
                'code': code,
                'language': language,
                'analysis': analysis,
                'warnings': analysis.get('warnings', [])
            }
            
        except Exception as e:
            logger.error(f"Erro na geração de código: {e}")
            return {
                'success': False,
                'error': str(e),
                'code': ''
            }
    
    def _analyze_code(self, code: str, language: str) -> Dict:
        """Analisa código gerado"""
        import ast
        analysis = {
            'syntax_valid': False,
            'complexity': 0,
            'security_issues': [],
            'optimization_suggestions': []
        }
        
        try:
            if language == 'python':
                tree = ast.parse(code)
                analysis['syntax_valid'] = True
                analysis['complexity'] = self._calculate_complexity(tree)
                
                # Detecção de problemas de segurança básicos
                security_check = self._security_analysis(code)
                analysis['security_issues'] = security_check
                
        except SyntaxError:
            analysis['syntax_valid'] = False
            
        return analysis
    
    def _calculate_complexity(self, tree) -> int:
        """Calcula complexidade ciclomática básica"""
        # Implementação simplificada
        return 1
    
    def _security_analysis(self, code: str) -> List[str]:
        """Análise básica de segurança"""
        issues = []
        dangerous_patterns = [
            'eval(', 'exec(', 'subprocess.call',
            'os.system', 'pickle.loads'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                issues.append(f"Uso potencialmente perigoso: {pattern}")
                
        return issues

class ImageGenerationModule:
    """Módulo de geração de imagens"""
    def __init__(self, config: NeuroGenesisConfig):
        self.config = config
        
    def generate_image(self, prompt: str, style: str = "realistic",
                      negative_prompt: str = "", **kwargs):
        """Gera imagem a partir de prompt textual"""
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            pipe = StableDiffusionPipeline.from_pretrained(
                self.config.models['image'],
                torch_dtype=torch.float16 if self.config.device.type == 'cuda' else torch.float32
            ).to(self.config.device)
            
            # Configurações avançadas
            generator = torch.Generator(device=self.config.device)
            
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generator,
                num_inference_steps=kwargs.get('steps', 50),
                guidance_scale=kwargs.get('guidance_scale', 7.5),
                width=kwargs.get('width', 512),
                height=kwargs.get('height', 512)
            ).images[0]
            
            # Salvar imagem
            output_path = self.config.memory_path / "temp" / f"generated_{hash(prompt)}.png"
            image.save(output_path)
            
            return {
                'success': True,
                'image_path': str(output_path),
                'prompt': prompt,
                'metadata': kwargs
            }
            
        except Exception as e:
            logger.error(f"Erro na geração de imagem: {e}")
            return {
                'success': False,
                'error': str(e)
            }

class AudioVideoModule:
    """Módulo de áudio e vídeo"""
    def __init__(self, config: NeuroGenesisConfig):
        self.config = config
        
    def generate_audio(self, text: str, language: str = "en", 
                      voice_type: str = "neutral"):
        """Gera áudio a partir de texto"""
        try:
            # Usando TTS avançado
            from TTS.api import TTS
            
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                     progress_bar=False).to(self.config.device)
            
            output_path = self.config.memory_path / "temp" / f"audio_{hash(text)}.wav"
            
            tts.tts_to_file(
                text=text,
                speaker_wav="path_to_reference_audio.wav",  # Necessário para XTTS
                language=language,
                file_path=str(output_path)
            )
            
            return {
                'success': True,
                'audio_path': str(output_path),
                'text': text,
                'language': language
            }
            
        except Exception as e:
            logger.error(f"Erro na geração de áudio: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_video(self, prompt: str, duration: int = 5):
        """Gera vídeo a partir de prompt"""
        try:
            from diffusers import DiffusionPipeline
            
            pipe = DiffusionPipeline.from_pretrained(
                self.config.models['video'],
                torch_dtype=torch.float16
            ).to(self.config.device)
            
            video_frames = pipe(
                prompt,
                num_inference_steps=duration * 8,  # Ajuste baseado em FPS
                num_frames=duration * 8
            ).frames
            
            # Processar frames em vídeo
            output_path = self.config.memory_path / "temp" / f"video_{hash(prompt)}.mp4"
            self._save_video(video_frames, str(output_path))
            
            return {
                'success': True,
                'video_path': str(output_path),
                'duration': duration,
                'prompt': prompt
            }
            
        except Exception as e:
            logger.error(f"Erro na geração de vídeo: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _save_video(self, frames, output_path: str):
        """Salva frames como vídeo"""
        import cv2
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, 8.0, (width, height))
        
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_rgb)
            
        video.release()

class MultimodalReasoningModule:
    """Módulo de raciocínio multimodal"""
    def __init__(self, config: NeuroGenesisConfig):
        self.config = config
        
    def analyze_multimodal(self, text_input: str = None, 
                          image_path: str = None,
                          audio_path: str = None):
        """Análise multimodal integrada"""
        from transformers import CLIPProcessor, CLIPModel
        
        model = CLIPModel.from_pretrained(self.config.models['multimodal'])
        processor = CLIPProcessor.from_pretrained(self.config.models['multimodal'])
        
        inputs = {}
        
        if text_input:
            inputs['text'] = text_input
            
        if image_path:
            from PIL import Image
            image = Image.open(image_path)
            inputs['images'] = image
            
        # Processamento multimodal
        processed = processor(**inputs, return_tensors="pt", padding=True)
        
        outputs = model(**processed)
        
        return {
            'text_embeddings': outputs.text_embeddings,
            'image_embeddings': outputs.image_embeddings,
            'logits_per_image': outputs.logits_per_image,
            'logits_per_text': outputs.logits_per_text
        }

class NeuroGenesisAI:
    """Classe principal da IA NeuroGenesis"""
    def __init__(self):
        self.config = NeuroGenesisConfig()
        self.learning_module = SelfLearningModule(self.config)
        self.code_module = CodeGenerationModule(self.config)
        self.image_module = ImageGenerationModule(self.config)
        self.audiovideo_module = AudioVideoModule(self.config)
        self.multimodal_module = MultimodalReasoningModule(self.config)
        
        logger.info(f"NeuroGenesis AI inicializada no dispositivo: {self.config.device}")
        logger.info(f"Memória configurada em: {self.config.memory_path}")
        
    def process_query(self, query: str, modality: str = "auto") -> Dict:
        """Processa consulta e decide melhor módulo"""
        # Análise da consulta para determinar modalidade
        if modality == "auto":
            modality = self._detect_modality(query)
        
        result = {
            'query': query,
            'modality': modality,
            'timestamp': np.datetime64('now'),
            'results': {}
        }
        
        try:
            if modality == "code":
                # Extrair linguagem da consulta
                language = self._extract_language(query)
                code_result = self.code_module.generate_code(query, language)
                result['results']['code'] = code_result
                
            elif modality == "image":
                image_result = self.image_module.generate_image(query)
                result['results']['image'] = image_result
                
            elif modality == "audio":
                audio_result = self.audiovideo_module.generate_audio(query)
                result['results']['audio'] = audio_result
                
            elif modality == "video":
                video_result = self.audiovideo_module.generate_video(query)
                result['results']['video'] = video_result
                
            elif modality == "multimodal":
                multimodal_result = self.multimodal_module.analyze_multimodal(query)
                result['results']['multimodal'] = multimodal_result
                
            else:  # texto geral
                result['results']['text'] = self._process_text(query)
            
            # Aprendizado com resultado
            self.learning_module.learn_from_experience(
                state=query,
                action=modality,
                reward=self._calculate_reward(result),
                next_state=result
            )
            
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Erro no processamento: {e}")
            result['success'] = False
            result['error'] = str(e)
            
        return result
    
    def _detect_modality(self, query: str) -> str:
        """Detecta modalidade baseada na consulta"""
        query_lower = query.lower()
        
        code_keywords = ['código', 'programa', 'função', 'classe', 'implemente']
        image_keywords = ['imagem', 'foto', 'desenho', 'ilustração', 'gerar imagem']
        audio_keywords = ['áudio', 'som', 'falar', 'voz', 'gerar voz']
        video_keywords = ['vídeo', 'animação', 'filme', 'gerar vídeo']
        
        if any(keyword in query_lower for keyword in code_keywords):
            return "code"
        elif any(keyword in query_lower for keyword in image_keywords):
            return "image"
        elif any(keyword in query_lower for keyword in audio_keywords):
            return "audio"
        elif any(keyword in query_lower for keyword in video_keywords):
            return "video"
        else:
            return "text"
    
    def _extract_language(self, query: str) -> str:
        """Extrai linguagem de programação da consulta"""
        languages = self.code_module.supported_languages
        
        for lang in languages:
            if lang in query.lower():
                return lang
                
        return "python"  # Default
    
    def _process_text(self, query: str) -> Dict:
        """Processamento de texto geral"""
        from transformers import pipeline
        
        generator = pipeline('text-generation', 
                           model=self.config.models['llm'],
                           device=self.config.device)
        
        response = generator(query, max_length=500, temperature=0.8)
        
        return {
            'response': response[0]['generated_text'],
            'model': self.config.models['llm']
        }
    
    def _calculate_reward(self, result: Dict) -> float:
        """Calcula recompensa para aprendizado por reforço"""
        reward = 0.0
        
        if result.get('success', False):
            reward += 1.0
            
            # Recompensas específicas por modalidade
            modality = result.get('modality', '')
            
            if modality == 'code':
                code_result = result['results'].get('code', {})
                if code_result.get('analysis', {}).get('syntax_valid', False):
                    reward += 0.5
                    
            elif modality == 'image':
                image_result = result['results'].get('image', {})
                if image_result.get('success', False):
                    reward += 0.5
                    
        return reward
    
    def save_state(self):
        """Salva estado completo do sistema"""
        self.learning_module.save_knowledge()
        
        state = {
            'config': vars(self.config),
            'knowledge_size': len(self.learning_module.knowledge_graph),
            'experience_memory_size': len(self.learning_module.experience_memory)
        }
        
        state_path = self.config.memory_path / "system_state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Estado do sistema salvo em: {state_path}")
    
    def interactive_mode(self):
        """Modo interativo de linha de comando"""
        print("\n" + "="*60)
        print("NEUROGENESIS AI - Sistema Multimodal Avançado")
        print("="*60)
        print("Comandos disponíveis:")
        print("  /code [linguagem] [prompt] - Gerar código")
        print("  /image [prompt] - Gerar imagem")
        print("  /audio [texto] - Gerar áudio")
        print("  /video [prompt] - Gerar vídeo")
        print("  /save - Salvar estado")
        print("  /exit - Sair")
        print("="*60)
        
        while True:
            try:
                user_input = input("\nNeuroGenesis> ").strip()
                
                if user_input.lower() == '/exit':
                    print("Saindo...")
                    self.save_state()
                    break
                    
                elif user_input.lower() == '/save':
                    self.save_state()
                    print("Estado salvo com sucesso!")
                    
                elif user_input.startswith('/code'):
                    parts = user_input.split(' ', 2)
                    if len(parts) >= 3:
                        _, language, prompt = parts
                        result = self.code_module.generate_code(prompt, language)
                        print(f"\nCódigo gerado ({language}):")
                        print("-"*40)
                        print(result.get('code', ''))
                        
                elif user_input.startswith('/image'):
                    prompt = user_input[7:] if len(user_input) > 7 else "Uma paisagem futurista"
                    result = self.image_module.generate_image(prompt)
                    if result['success']:
                        print(f"\nImagem gerada: {result['image_path']}")
                    else:
                        print(f"Erro: {result.get('error', 'Desconhecido')}")
                        
                elif user_input.startswith('/audio'):
                    text = user_input[7:] if len(user_input) > 7 else "Olá, mundo!"
                    result = self.audiovideo_module.generate_audio(text)
                    if result['success']:
                        print(f"\nÁudio gerado: {result['audio_path']}")
                    else:
                        print(f"Erro: {result.get('error', 'Desconhecido')}")
                        
                elif user_input.startswith('/video'):
                    prompt = user_input[7:] if len(user_input) > 7 else "Uma cidade futurista"
                    result = self.audiovideo_module.generate_video(prompt)
                    if result['success']:
                        print(f"\nVídeo gerado: {result['video_path']}")
                    else:
                        print(f"Erro: {result.get('error', 'Desconhecido')}")
                        
                else:
                    result = self.process_query(user_input)
                    if result['success']:
                        print(f"\nResposta ({result['modality']}):")
                        print("-"*40)
                        if result['modality'] == 'text':
                            print(result['results']['text']['response'])
                        else:
                            print(json.dumps(result, indent=2, default=str))
                    else:
                        print(f"Erro: {result.get('error', 'Desconhecido')}")
                        
            except KeyboardInterrupt:
                print("\n\nInterrompido pelo usuário.")
                self.save_state()
                break
            except Exception as e:
                print(f"Erro: {e}")

def install_dependencies():
    """Instala dependências necessárias"""
    import subprocess
    import sys
    
    dependencies = [
        'torch',
        'torchvision',
        'torchaudio',
        'transformers',
        'diffusers',
        'accelerate',
        'opencv-python',
        'pillow',
        'numpy',
        'scipy',
        'TTS',
        'soundfile',
        'librosa',
        'ftfy',
        'regex',
        'tqdm',
        'sentencepiece',
        'protobuf',
        'einops',
        'xformers',  # Otimização para GPU
        'flash-attn'  # Atenção rápida
    ]
    
    print("Instalando dependências...")
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', dep])
            print(f"✓ {dep} instalado")
        except subprocess.CalledProcessError:
            print(f"✗ Falha ao instalar {dep}")
    
    # Instalação especial para CUDA se disponível
    if torch.cuda.is_available():
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'torch', '--index-url', 'https://download.pytorch.org/whl/cu118'])

def main():
    """Função principal"""
    print("Inicializando NeuroGenesis AI...")
    
    # Verificar e instalar dependências
    try:
        import transformers
    except ImportError:
        print("Dependências não encontradas. Instalando...")
        install_dependencies()
    
    # Inicializar IA
    ai = NeuroGenesisAI()
    
    # Modo de operação
    if len(sys.argv) > 1:
        # Modo batch
        query = " ".join(sys.argv[1:])
        result = ai.process_query(query)
        print(json.dumps(result, indent=2, default=str))
    else:
        # Modo interativo
        ai.interactive_mode()

if __name__ == "__main__":
    main()