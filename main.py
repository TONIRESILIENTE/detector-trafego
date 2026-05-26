import cv2
import csv
import os
import sys
from ultralytics import YOLO
from utils import CLASSES_INTERESSE, inicializar_contagem, desenhar_e_coletar

# ================= CONFIGURAÇÕES =================
if len(sys.argv) > 1:
    VIDEO_ENTRADA = sys.argv[1]
else:
    VIDEO_ENTRADA = os.path.join('videos', 'traffic.mp4')  # padrão
nome_base = os.path.splitext(os.path.basename(VIDEO_ENTRADA))[0]
VIDEO_SAIDA = os.path.join('output', f'{nome_base}_out.mp4')
MODELO = 'yolov8n.pt'

# ================= CARREGAR MODELO =================
print("Carregando modelo YOLOv8...")
model = YOLO(MODELO)

# ================= ABRIR VÍDEO =================
cap = cv2.VideoCapture(VIDEO_ENTRADA)

if not cap.isOpened():
    print(f"Erro ao abrir o vídeo: {VIDEO_ENTRADA}")
    exit()

# Propriedades do vídeo
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Criar diretório de saída se não existir
os.makedirs('output', exist_ok=True)

# Configurar gravador de vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_SAIDA, fourcc, fps, (largura, altura))

# ================= LOOP DE PROCESSAMENTO =================
contagem = inicializar_contagem()
frame_count = 0
dados_deteccoes = []   # acumulará todos os dados

print("Processando...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes_ids = results.boxes.cls.cpu().numpy().astype(int)

        # Uma única chamada já desenha e coleta os dados
        frame, deteccoes_frame = desenhar_e_coletar(
            frame, boxes, confs, classes_ids, CLASSES_INTERESSE, contagem
        )

        # Adiciona o número do frame a cada detecção e acumula
        for d in deteccoes_frame:
            d['frame'] = frame_count
        dados_deteccoes.extend(deteccoes_frame)

    # Gravar frame processado
    out.write(frame)
    frame_count += 1

# ================= SALVAR CSV =================
csv_path = os.path.join('output', f'{nome_base}_deteccoes.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['frame', 'classe', 'confianca',
                                           'x1', 'y1', 'x2', 'y2'])
    writer.writeheader()
    writer.writerows(dados_deteccoes)
print(f"Dados das detecções salvos em {csv_path}")

# ================= FINALIZAR =================
cap.release()
out.release()
print(f"Processamento concluído! {frame_count} frames analisados.")
print("\nContagem total de objetos detectados:")
for classe, qtd in sorted(contagem.items()):
    print(f"  {classe}: {qtd}")
