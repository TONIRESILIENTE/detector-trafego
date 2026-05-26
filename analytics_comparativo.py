import pandas as pd
import matplotlib.pyplot as plt
import os

# Lista de vídeos analisados
cenarios = ['urban', 'highway', 'night']
dfs = {}

for cenario in cenarios:
    csv_path = os.path.join('output', f'{cenario}_deteccoes.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['cenario'] = cenario
        dfs[cenario] = df
    else:
        print(f"Aviso: {csv_path} não encontrado.")

if not dfs:
    print("Nenhum CSV encontrado.")
    exit()

# Concatenar todos os dados
dados_completos = pd.concat(dfs.values(), ignore_index=True)

# =============== ESTATÍSTICAS ===============
print("=== TOTAIS POR CENÁRIO E CLASSE ===\n")
totais = dados_completos.groupby(
    ['cenario', 'classe']).size().unstack(fill_value=0)
print(totais)

print("\n=== CONFIANÇA MÉDIA POR CENÁRIO E CLASSE ===\n")
conf_media = dados_completos.groupby(['cenario', 'classe'])[
    'confianca'].mean().unstack()
print(conf_media)

# =============== GRÁFICO 1: Barras comparativas de total de detecções ===============
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for i, cenario in enumerate(cenarios):
    ax = axes[i]
    if cenario in totais.index:
        totais.loc[cenario].plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title(f'Total de detecções - {cenario}')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Quantidade')
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('output/comparativo_total.png')
plt.show()

# =============== GRÁFICO 2: Boxplots de confiança por cenário e classe ===============
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for i, cenario in enumerate(cenarios):
    ax = axes[i]
    if cenario in dfs:
        df_cen = dfs[cenario]
        df_cen.boxplot(column='confianca', by='classe', ax=ax, grid=False)
        ax.set_title(f'Confiança - {cenario}')
        ax.set_xlabel('Classe')
        ax.set_ylabel('Confiança')
        ax.set_ylim(0, 1)
        plt.sca(ax)
        plt.xticks(rotation=45)
plt.suptitle('')
plt.tight_layout()
plt.savefig('output/comparativo_confianca.png')
plt.show()

# =============== GRÁFICO 3: Linha temporal empilhada (opcional, pode ser pesado) ===============
# Para o caso highway quase não tem detecções, então vamos pular.
# Mas podemos gerar para urban e night separadamente se quisermos.
