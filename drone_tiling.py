# drone_tiling.py
import cv2
import csv
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from utils import CLASSES_INTERESSE, inicializar_contagem

# ================= CONFIGURAÇÕES =================
VIDEO_ENTRADA = os.path.join('videos', 'highway.mp4')
VIDEO_SAIDA = os.path.join('output', 'highway_tiling_out.mp4')
CSV_SAIDA = os.path.join('output', 'highway_tiling_deteccoes.csv')
MODELO = 'yolov8n.pt'

# Parâmetros do tiling
TILES_LINHAS = 2       # divide a altura em 2
TILES_COLUNAS = 2      # divide a largura em 2
OVERLAP = 0.2          # 20% de sobreposição entre tiles
CONF_THRESHOLD = 0.25  # limiar mínimo de confiança

# ================= CARREGAR MODELO =================
print("Carregando YOLOv8n...")
model = YOLO(MODELO)

# ================= ABRIR VÍDEO =================
cap = cv2.VideoCapture(VIDEO_ENTRADA)
if not cap.isOpened():
    print(f"Erro ao abrir o vídeo: {VIDEO_ENTRADA}")
    exit()

largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

os.makedirs('output', exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_SAIDA, fourcc, fps, (largura, altura))

# ================= FUNÇÃO DE TILING =================


def gerar_tiles(frame, linhas, colunas, overlap):
    """Divide o frame em tiles com sobreposição e retorna lista de (tile, x_offset, y_offset)."""
    h, w = frame.shape[:2]
    tile_h = int(h / linhas * (1 + overlap))
    tile_w = int(w / colunas * (1 + overlap))
    step_y = int(h / linhas)
    step_x = int(w / colunas)

    tiles = []
    for i in range(linhas):
        for j in range(colunas):
            y_start = max(0, i * step_y - int(tile_h * overlap / 2))
            y_end = min(h, y_start + tile_h)
            x_start = max(0, j * step_x - int(tile_w * overlap / 2))
            x_end = min(w, x_start + tile_w)

            tile = frame[y_start:y_end, x_start:x_end]
            tile_resized = cv2.resize(tile, (640, 640))
            tiles.append((tile_resized, x_start, y_start,
                         x_end - x_start, y_end - y_start))
    return tiles


def ajustar_coordenadas(box, tile_orig_x, tile_orig_y, tile_orig_w, tile_orig_h):
    """Converte coordenadas do tile (640x640) de volta para o frame original."""
    x1, y1, x2, y2 = box
    escala_x = tile_orig_w / 640
    escala_y = tile_orig_h / 640
    x1_orig = int(x1 * escala_x + tile_orig_x)
    y1_orig = int(y1 * escala_y + tile_orig_y)
    x2_orig = int(x2 * escala_x + tile_orig_x)
    y2_orig = int(y2 * escala_y + tile_orig_y)
    return x1_orig, y1_orig, x2_orig, y2_orig


# ================= LOOP =================
contagem = inicializar_contagem()
frame_count = 0
dados_deteccoes = []

print("Processando com tiling...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_com_deteccoes = frame.copy()
    tiles = gerar_tiles(frame, TILES_LINHAS, TILES_COLUNAS, OVERLAP)

    for tile_img, orig_x, orig_y, tile_w, tile_h in tiles:
        results = model(tile_img, verbose=False)[0]

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes_ids = results.boxes.cls.cpu().numpy().astype(int)

            for i, box in enumerate(boxes):
                classe_id = classes_ids[i]
                confianca = float(confs[i])
                if classe_id in CLASSES_INTERESSE and confianca >= CONF_THRESHOLD:
                    nome = CLASSES_INTERESSE[classe_id]
                    contagem[nome] += 1

                    # Ajustar coordenadas para o frame original
                    x1, y1, x2, y2 = ajustar_coordenadas(
                        box, orig_x, orig_y, tile_w, tile_h)

                    cv2.rectangle(frame_com_deteccoes, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_com_deteccoes, f"{nome} {confianca:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    dados_deteccoes.append({
                        'frame': frame_count,
                        'classe': nome,
                        'confianca': confianca,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                    })

    out.write(frame_com_deteccoes)
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Frames processados: {frame_count}")

# ================= SALVAR CSV =================
with open(CSV_SAIDA, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(
        f, fieldnames=['frame', 'classe', 'confianca', 'x1', 'y1', 'x2', 'y2'])
    writer.writeheader()
    writer.writerows(dados_deteccoes)

cap.release()
out.release()
print(f"Processamento concluído! {frame_count} frames.")
print("Contagem de objetos detectados:")
for classe, qtd in sorted(contagem.items()):
    print(f"  {classe}: {qtd}")
print(f"Dados salvos em {CSV_SAIDA}")
