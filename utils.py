import cv2
from collections import defaultdict

CLASSES_INTERESSE = {
    0: 'pedestre',
    1: 'bicicleta',
    2: 'carro',
    3: 'moto',
    5: 'onibus',
    7: 'caminhao'
}


def inicializar_contagem():
    return defaultdict(int)


def desenhar_e_coletar(frame, boxes, confs, classes_ids, nomes_classes, contagem):
    """
    Desenha retângulos, atualiza contagem e retorna lista de detecções
    para análise posterior.
    """
    deteccoes_frame = []
    for i, box in enumerate(boxes):
        classe_id = classes_ids[i]
        if classe_id in nomes_classes:
            nome = nomes_classes[classe_id]
            contagem[nome] += 1
            confianca = float(confs[i])

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{nome} {confianca:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            deteccoes_frame.append({
                'classe': nome,
                'confianca': confianca,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })
    return frame, deteccoes_frame
