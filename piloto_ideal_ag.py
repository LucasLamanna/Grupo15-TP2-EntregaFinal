import random
from deap import base, creator, tools, algorithms # 
import numpy 
import matplotlib.pyplot as plt 
# --- Importamos desde nuestro archivo de configuración del problema ---
from config_piloto import LONGITUD_CROMOSOMA, evaluar_aptitud_piloto, imprimir_perfil_piloto, evaluar_aptitud_piloto_nueva

# --- 1. Definición de Tipos  ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax)

# --- 2. Inicialización y Registro en la Toolbox ---
toolbox = base.Toolbox()

# Generador de Atributos (cada bit del cromosoma):
toolbox.register("attr_bool", random.randint, 0, 1)

# Inicializador de Individuos (cromosomas):

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, LONGITUD_CROMOSOMA)

# Inicializador de Población:

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- 3. Registro de los Operadores Genéticos ---

# A. Función de Evaluación (Fitness Function):
toolbox.register("evaluate", evaluar_aptitud_piloto_nueva)

# B. Operador de Cruce (Crossover):
toolbox.register("mate", tools.cxTwoPoint) 

# C. Operador de Mutación:
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# D. Operador de Selección:
# Método Torneo, ir modificando el tournsize según la configuración de cada corrida
toolbox.register("select", tools.selTournament, tournsize=40) 

# Ir cambiando según la configuración de cada corrida
# Selección por ruleta:
#toolbox.register("select", tools.selRoulette)

# --- 4. Configuración de Estadísticas y Hall of Fame ---
hof = tools.HallOfFame(3)

# B. Estadísticas:
# Configuración de parametros de librería
stats = tools.Statistics(lambda ind: ind.fitness.values)

# Registramos las estadísticas específicas que queremos calcular:
stats.register("avg", numpy.mean) 
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)

# --- 5. Definición de Parámetros del Algoritmo y Ejecución ---
def ejecutar_ag():

    TAM_POBLACION = 100  
    PROBABILIDAD_CRUCE = 0.7 
    PROBABILIDAD_MUTACION = 0.3 
    NUM_GENERACIONES = 100 

    print(f"Iniciando evolución con {NUM_GENERACIONES} generaciones y población de {TAM_POBLACION} individuos...")
    print(f"Probabilidad de Cruce: {PROBABILIDAD_CRUCE}, Probabilidad de Mutación: {PROBABILIDAD_MUTACION}")

    # Creación de la población inicial
    pop = toolbox.population(n=TAM_POBLACION)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
    cxpb=PROBABILIDAD_CRUCE,
    mutpb=PROBABILIDAD_MUTACION,
    ngen=NUM_GENERACIONES,
    stats=stats,
    halloffame=hof,
    verbose=True
    )

    return pop, logbook, hof

if __name__ == "__main__":
    poblacion_final, libro_estadisticas, salon_fama = ejecutar_ag()

    for piloto_hof in salon_fama:
        print("\n --- SUGERENCIA PILOTO CANDIDATO ---")
        imprimir_perfil_piloto(piloto_hof) 
        print(f"Aptitud del perfil sugerido: {piloto_hof.fitness.values[0]:.2f}")

    # Generamos gráfico de la función aptitud
    gen = libro_estadisticas.select("gen")
    avg_fitness = libro_estadisticas.select("avg")
    max_fitness = libro_estadisticas.select("max")

    plt.figure(figsize=(10, 6))
    plt.plot(gen, avg_fitness, label="Aptitud Promedio", color='red')
    plt.xlabel("Generación")
    plt.ylabel("Aptitud")
    plt.legend(loc="lower right")
    plt.title("Evolución de la Aptitud a lo largo de las Generaciones")
    plt.grid(True)
    plt.savefig("grafico_piloto.png")

    print("\nEvolución completada.")