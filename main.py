import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as path_effects

# =============================================================================
# CONFIGURAÇÕES E PARÂMETROS DO MODELO SEIRD
# =============================================================================

# Configurações da Grade
GRID_SIZE = 50  # 50x50 = 2500 agentes
DAYS = 200      # Duração da simulação

# Parâmetros Epidemiológicos
INFECTION_RATE = 0.3    # Probabilidade de infecção por vizinho infectado
INCUBATION_PERIOD = 5.0 # Dias (T_inc)
RECOVERY_RATE = 0.1     # Taxa de recuperação (1/T_inf)
MORTALITY_RATE = 0.0225 # Taxa de letalidade (~2.25%)

# Mapeamento de Estados
SUSCEPTIBLE = 0
EXPOSED = 1
INFECTED = 2
RECOVERED = 3
DEAD = 4

# Configuração Visual
# Cores: Branco (S), Amarelo (E), Vermelho (I), Verde (R), Preto (D)
COLORS = ['#FFFFFF', '#F1C40F', '#E74C3C', '#2ECC71', '#2C3E50']
CMAP = ListedColormap(COLORS)
LABELS = ['S', 'E', 'I', 'R', 'D']

# =============================================================================
# INICIALIZAÇÃO DO SISTEMA
# =============================================================================

grid = np.full((GRID_SIZE, GRID_SIZE), SUSCEPTIBLE)

# Paciente Zero (Centro da grade)
grid[GRID_SIZE//2, GRID_SIZE//2] = INFECTED

# Histórico para análise estatística
stats = {label: [] for label in LABELS}

def get_neighbors_infected(r, c, current_grid):
    """Vizinhança de Moore (8 vizinhos)"""
    count = 0
    for i in range(max(0, r-1), min(GRID_SIZE, r+2)):
        for j in range(max(0, c-1), min(GRID_SIZE, c+2)):
            if (i != r or j != c) and current_grid[i, j] == INFECTED:
                count += 1
    return count

def update(frame, img, ax):
    global grid
    new_grid = grid.copy()
    counts = {s: 0 for s in range(5)}
    
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            state = grid[r, c]
            counts[state] += 1
            
            if state == SUSCEPTIBLE:
                n_inf = get_neighbors_infected(r, c, grid)
                # Probabilidade de transição S -> E
                prob = 1 - (1 - INFECTION_RATE)**n_inf
                if np.random.rand() < prob:
                    new_grid[r, c] = EXPOSED
                    
            elif state == EXPOSED:
                # Transição E -> I baseada no período de incubação
                if np.random.rand() < (1.0 / INCUBATION_PERIOD):
                    new_grid[r, c] = INFECTED
                    
            elif state == INFECTADO or state == INFECTED: # Compatibilidade
                # Transição I -> R ou I -> D
                if np.random.rand() < RECOVERY_RATE:
                    if np.random.rand() < MORTALITY_RATE:
                        new_grid[r, c] = DEAD
                    else:
                        new_grid[r, c] = RECOVERED

    grid = new_grid
    img.set_data(grid)
    
    # Atualiza estatísticas
    for i, label in enumerate(LABELS):
        stats[label].append(counts[i])
    
    # Título com contador em tempo real
    title_str = (f"Dia: {frame} | S: {counts[0]} | E: {counts[1]} | "
                 f"I: {counts[2]} | R: {counts[3]} | D: {counts[4]}")
    ax.set_title(title_str, fontsize=10, fontweight='bold')
    
    return [img]

# =============================================================================
# EXECUÇÃO E VISUALIZAÇÃO
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 8))
img = ax.imshow(grid, cmap=CMAP, vmin=0, vmax=4, interpolation='nearest')

# Barra de cores rotulada
cbar = plt.colorbar(img, ticks=[0, 1, 2, 3, 4])
cbar.ax.set_yticklabels(['Suscetível', 'Exposto', 'Infectado', 'Recuperado', 'Óbito'])

ani = animation.FuncAnimation(fig, update, fargs=(img, ax), frames=DAYS, interval=50, repeat=False)

plt.tight_layout()
plt.show()

# Gráfico de Evolução Final
plt.figure(figsize=(12, 6))
colors_plot = ['#95a5a6', '#f1c40f', '#e74c3c', '#2ecc71', '#2c3e50']
for i, label in enumerate(LABELS):
    plt.plot(stats[label], label=label, color=colors_plot[i], lw=2.5)

plt.title('Dinâmica Temporal do Modelo SEIRD (Autômatos Celulares)', fontsize=14)
plt.xlabel('Dias', fontsize=12)
plt.ylabel('Número de Agentes', fontsize=12)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.show()
