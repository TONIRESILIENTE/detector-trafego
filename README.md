
# 🚗 Detector de Tráfego Inteligente com YOLOv8

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-nano-brightgreen)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-red)
![Status](https://img.shields.io/badge/status-concluído-success)
![Licença](https://img.shields.io/badge/licença-MIT-yellow)

> **Projeto de portfólio desenvolvido durante transição de carreira para a área de tecnologia, com foco em visão computacional e aprendizado prático.**

---

## 🎯 Objetivo

Construir do zero um sistema capaz de **detectar veículos e pedestres em vídeos de tráfego**, utilizando técnicas de visão computacional e deep learning. O projeto priorizou o entendimento profundo de cada etapa, a experimentação com diferentes cenários e a análise crítica dos resultados — indo muito além de um simples tutorial.

---

## 🧠 Contexto do autor

Este projeto é um marco na minha **transição de carreira** para a área de dados e inteligência artificial. Cada decisão, erro e acerto foi documentado como parte do aprendizado. A ideia central não é apenas mostrar que o código funciona, mas demonstrar a capacidade de **investigar problemas, testar hipóteses e comunicar descobertas técnicas** — habilidades essenciais para um profissional da área.

---

## 🛠️ Tecnologias e ferramentas

| Ferramenta | Função |
|------------|--------|
| **Python 3.13** | Linguagem principal |
| **YOLOv8 (Ultralytics)** | Modelo de detecção de objetos (versão nano) |
| **OpenCV** | Manipulação de vídeos e desenho de anotações |
| **Pandas, Matplotlib** | Análise de dados e geração de gráficos |
| **Google Colab (início)** | Prototipação inicial (depois migrado para ambiente local) |
| **VS Code** | Ambiente de desenvolvimento local |
| **Git & GitHub** | Versionamento e portfólio |

---

## 📁 Estrutura do projeto
detector-trafego/
│
├── main.py # Script principal (processa um vídeo e gera CSV)
├── utils.py # Funções auxiliares (desenho, coleta de dados)
├── analytics.py # Análise estatística de um cenário
├── analytics_comparativo.py # Comparação entre múltiplos cenários
├── drone_tiling.py # Experimento com técnica de tiling para vista aérea
│
├── videos/ # Vídeos de entrada
├── output/ # Vídeos processados, CSVs e gráficos
│
├── requirements.txt # Dependências do projeto
├── .gitignore
└── README.md # Este documento

text

---

## ⚙️ Como reproduzir

1.  Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/detector-trafego.git
    cd detector-trafego
Crie e ative o ambiente virtual:

bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
Instale as dependências:

bash
pip install -r requirements.txt
Execute o script principal (exemplo com vídeo urbano):

bash
python main.py videos/urban.mp4
O vídeo processado e o CSV com as detecções serão salvos na pasta output/.

Gere as análises:

bash
python analytics.py          # para um cenário específico
python analytics_comparativo.py  # para comparar todos os cenários
🔬 Metodologia
Coleta de vídeos: três vídeos gratuitos do Pixabay, representando condições distintas:

urban.mp4 – centro urbano movimentado (pedestres, ônibus).

night.mp4 – tráfego noturno em avenida.

highway.mp4 – rodovia vista de cima (imagem aérea).

Detecção com YOLOv8n: modelo pré-treinado no dataset COCO. Para cada frame, extraímos coordenadas, classe e confiança das detecções.

Coleta de dados: registramos cada detecção em arquivos CSV (frame, classe, confiança, coordenadas) para análise posterior.

Análise quantitativa e qualitativa: geramos estatísticas (contagem, confiança média, distribuição temporal) e gráficos comparativos.

Tentativa de melhoria: ao constatar falha na detecção aérea, implementamos a técnica de tiling (ladrilhamento) e investigamos o uso de um modelo especializado (VisDrone).

📊 Resultados e análises
1. Cenário urbano (urban.mp4)
763 frames processados

Detecções totais: 6.813

Distribuição:

Classe	Detecções	Confiança média
Pedestre	4.499	0.35
Ônibus	1.680	0.46
Carro	566	0.45
Caminhão	66	0.32
Bicicleta	2	0.33
Interpretação: Ambiente extremamente denso, com muitos pedestres e ônibus. A confiança mais baixa para pedestres (0.35) reflete objetos menores e frequentes oclusões. O modelo se comportou de forma consistente com o esperado para uma cena urbana complexa.

2. Cenário noturno (night.mp4)
2.281 frames processados

Detecções totais: 2.380

Distribuição:

Classe	Detecções	Confiança média
Carro	1.153	0.40
Caminhão	921	0.58
Ônibus	299	0.47
Pedestre	5	0.45
Moto	2	0.28
Interpretação: Apesar da baixa iluminação, veículos foram bem detectados. A alta confiança dos caminhões (0.58) sugere que os faróis e o contraste noturno ajudam o modelo. Pedestres quase inexistentes — coerente com o horário.

3. Rodovia aérea (highway.mp4) ⚠️
1.192 frames processados

Detecções totais (modelo padrão): apenas 4 pedestres, nenhum veículo.

Confiança média dos pedestres: 0.31

Problema identificado: O vídeo mostra uma rodovia movimentada vista de cima. O modelo padrão não detectou nenhum carro ou caminhão.

Causa raiz: O YOLOv8n foi treinado no dataset COCO, composto por imagens do nível do chão. Carros vistos de cima têm aparência completamente diferente (retângulos sem rodas, faróis ou para-brisas). O modelo simplesmente não aprendeu esse padrão.

🧪 Tentativa de solução: Tiling (ladrilhamento)
Para mitigar o problema, implementamos a técnica de tiling: cada frame foi dividido em 4 tiles com sobreposição de 20%, redimensionados para 640×640 e processados individualmente. As coordenadas foram mapeadas de volta para o frame original.

Resultado com tiling:

Detecções totais: 171

Caminhão: 12 | Carro: 77 | Pedestre: 82

Análise crítica:
Embora tenham surgido mais detecções, a maioria era falsos positivos — árvores e casas foram confundidas com pedestres e carros. Os veículos em movimento na pista continuaram não sendo detectados. O tiling amplia a imagem, mas não altera a natureza do que o modelo aprendeu. A falha não era de escala, e sim de domínio dos dados de treinamento.

Conclusão: Para detecção eficaz em imagens aéreas, é indispensável um modelo treinado com dados desse domínio (ex.: VisDrone, DOTA) ou realizar fine-tuning com imagens aéreas rotuladas.

🚧 Dificuldades enfrentadas
Dificuldade	Como foi contornada	Aprendizado
Ambientes virtuais conflitantes (venv vs .venv)	Identificação do problema e recriação do ambiente correto	Atenção à gestão de dependências
Mudança de local da pasta (OneDrive) quebrou o venv	Recriação do venv no novo caminho	Entendimento de como paths absolutos afetam ambientes virtuais
Incompatibilidade PyTorch 2.6 com ultralytics	Atualização da biblioteca ultralytics	Versões de bibliotecas são críticas
Falha na detecção aérea	Tentativa de tiling + análise profunda da causa	Limitação de modelos pré-treinados e importância do domínio dos dados
Tentativa frustrada de usar modelo VisDrone (problemas de download e compatibilidade)	Documentação da tentativa como parte do aprendizado	Nem toda solução está prontamente acessível; é preciso saber quando pivotar
💡 Principais aprendizados
Modelos pré-treinados não são universais — o domínio dos dados de treinamento determina o desempenho.

Engenharia de pipelines de inferência — tiling, ajuste de thresholds, pós-processamento — pode compensar limitações, mas dentro de certos limites.

Análise crítica é tão importante quanto a implementação — números sem interpretação não geram valor.

Documentar falhas e tentativas frustradas demonstra honestidade intelectual e capacidade de resolver problemas complexos.

Organização do código e versionamento são fundamentais para qualquer projeto profissional.

🔮 Recomendações para evolução
Utilizar modelo YOLOv8-visdrone (assim que a compatibilidade for resolvida) para detecção aérea.

Implementar rastreamento (ByteTrack) para contar cada veículo uma única vez.

Realizar fine-tuning com imagens aéreas rotuladas manualmente (ex.: trechos do próprio vídeo).

Criar uma interface web simples com Streamlit para upload e visualização de vídeos.

🤝 Agradecimentos
Este projeto foi construído com o apoio de ferramentas gratuitas e comunidades open source. Um agradecimento especial à documentação do Ultralytics e aos mantenedores dos datasets COCO e VisDrone.

📄 Licença
MIT — sinta-se à vontade para usar, modificar e compartilhar.

Desenvolvido como parte de uma jornada de transição de carreira. Feedbacks são muito bem-vindos!