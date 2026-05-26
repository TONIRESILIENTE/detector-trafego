import pandas as pd
import matplotlib.pyplot as plt
import os

# Carregar dados
csv_path = os.path.join('output', 'deteccoes.csv')
df = pd.read_csv(csv_path)

print("=== ESTATÍSTICAS GERAIS ===")
print(f"Total de detecções: {len(df)}")
print(f"Total de frames com detecções: {df['frame'].nunique()}")
print("\nContagem por classe:")
print(df['classe'].value_counts())

print("\nConfiança média por classe:")
print(df.groupby('classe')['confianca'].mean())

print("\nConfiança mínima / máxima por classe:")
print(df.groupby('classe')['confianca'].agg(['min', 'max']))

# Gráfico 1: distribuição de confiança por classe (boxplot)
plt.figure(figsize=(10, 5))
df.boxplot(column='confianca', by='classe')
plt.title('Distribuição de Confiança por Classe')
plt.suptitle('')
plt.xlabel('Classe')
plt.ylabel('Confiança')
plt.grid(False)
plt.savefig(os.path.join('output', 'confianca_por_classe.png'))
plt.show()

# Gráfico 2: detecções ao longo do tempo (série temporal)
detections_by_frame = df.groupby(
    ['frame', 'classe']).size().unstack(fill_value=0)
detections_by_frame.plot(figsize=(12, 6))
plt.title('Detecções por Frame e Classe')
plt.xlabel('Frame')
plt.ylabel('Número de detecções')
plt.grid(True)
plt.savefig(os.path.join('output', 'deteccoes_tempo.png'))
plt.show()

# Gráfico 3: total por classe (barras)
plt.figure(figsize=(8, 5))
df['classe'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Total de Detecções por Classe')
plt.xlabel('Classe')
plt.ylabel('Contagem')
plt.tight_layout()
plt.savefig(os.path.join('output', 'total_por_classe.png'))
plt.show()
