# --- Definición de Atributos y sus Valores ---
ATRIBUTOS_CONFIG = {
    "A1_Experiencia": {
        "00": "Novato", 
        "01": "Joven Promesa", 
        "10": "Establecido", 
        "11": "Veterano"
    },
    "A2_EstiloConduccion": {
        "00": "Agresivo Controlado",
        "01": "Consistente y Calculador",
        "10": "Técnico y Metódico",
        "11": "Adaptable Camaleónico"
    },
    "A3_VelocidadPura": {
        "00": "Regular",
        "01": "Buena",
        "10": "Muy Buena",
        "11": "Excepcional"
    },
    "A4_ConsistenciaCarrera": {
        "00": "Inconsistente",
        "01": "Algo Consistente",
        "10": "Muy Consistente",
        "11": "Extremadamente Consistente"
    },
    "A5_FeedbackTecnico": {
        "00": "Limitada",
        "01": "Adecuada",
        "10": "Fuerte",
        "11": "Excepcional"
    },
    "A6_MentalidadEquipo": {
        "00": "Primariamente Individualista",
        "01": "Equilibrado",
        "10": "Jugador de Equipo Nato",
        "11": "Totalmente Alineado con el Equipo"
    },
    "A7_EncajeMarca": {
        "00": "Bajo Encaje",
        "01": "Encaje Aceptable",
        "10": "Buen Encaje",
        "11": "Encaje Perfecto"
    },
    "A8_ExigenciaSalarial": {
        "00": "Salario Muy Bajo",
        "01": "Salario Bajo",
        "10": "Salario Medio",
        "11": "Salario Alto"
    }
}
# Definimos el orden y el tamaño en bits de cada atributo en el cromosoma
ORDEN_ATRIBUTOS = [
    ("A1_Experiencia", 0, 2),
    ("A2_EstiloConduccion", 2, 2),
    ("A3_VelocidadPura", 4, 2),
    ("A4_ConsistenciaCarrera", 6, 2),
    ("A5_FeedbackTecnico", 8, 2),
    ("A6_MentalidadEquipo", 10, 2),
    ("A7_EncajeMarca", 12, 2),
    ("A8_ExigenciaSalarial", 14, 2)
]

LONGITUD_CROMOSOMA = 16

def decodificar_cromosoma(cromosoma_bits):
    if len(cromosoma_bits) != LONGITUD_CROMOSOMA:
        raise ValueError(f"El cromosoma debe tener {LONGITUD_CROMOSOMA} bits.")

    perfil_decodificado = {}
    for nombre_attr, inicio_bit, num_bits in ORDEN_ATRIBUTOS:
        
        bits_atributo = "".join(map(str, cromosoma_bits[inicio_bit : inicio_bit + num_bits]))
        
        perfil_decodificado[nombre_attr] = ATRIBUTOS_CONFIG[nombre_attr][bits_atributo]

    return perfil_decodificado

# --- Función de Aptitud ---
def evaluar_aptitud_piloto(cromosoma_bits):

    perfil = decodificar_cromosoma(cromosoma_bits)
    
    puntaje_base = 0
    bonificaciones_individuales = 0
    bonificaciones_sinergia = 0
    penalizaciones = 0 

    # --- A. Bonificaciones por Atributos Individuales Positivos (BIs) ---
    if perfil["A1_Experiencia"] == "Joven Promesa": bonificaciones_individuales += 2
    elif perfil["A1_Experiencia"] == "Establecido": bonificaciones_individuales += 2
    elif perfil["A1_Experiencia"] == "Veterano": bonificaciones_individuales += 1

    if perfil["A2_EstiloConduccion"] == "Agresivo Controlado": bonificaciones_individuales += 4 
    elif perfil["A2_EstiloConduccion"] == "Adaptable Camaleónico": bonificaciones_individuales += 2

    if perfil["A3_VelocidadPura"] == "Excepcional": bonificaciones_individuales += 4
    elif perfil["A3_VelocidadPura"] == "Muy Buena": bonificaciones_individuales += 2

    if perfil["A4_ConsistenciaCarrera"] == "Extremadamente Consistente": bonificaciones_individuales += 4
    elif perfil["A4_ConsistenciaCarrera"] == "Muy Consistente": bonificaciones_individuales += 2

    if perfil["A5_FeedbackTecnico"] == "Excepcional": bonificaciones_individuales += 4
    elif perfil["A5_FeedbackTecnico"] == "Fuerte": bonificaciones_individuales += 2

    if perfil["A6_MentalidadEquipo"] == "Jugador de Equipo Nato": bonificaciones_individuales += 2
    elif perfil["A6_MentalidadEquipo"] == "Totalmente Alineado con el Equipo": bonificaciones_individuales += 3
    elif perfil["A6_MentalidadEquipo"] == "Equilibrado": bonificaciones_individuales += 1

    if perfil["A7_EncajeMarca"] == "Encaje Perfecto": bonificaciones_individuales += 3
    elif perfil["A7_EncajeMarca"] == "Buen Encaje": bonificaciones_individuales += 2

    if perfil["A8_ExigenciaSalarial"] == "Salario Muy Bajo": bonificaciones_individuales += 2 
    elif perfil["A8_ExigenciaSalarial"] == "Salario Bajo": bonificaciones_individuales += 1 

    # --- B. Bonificaciones por Sinergias (BDs) ---
    # BD1
    if (perfil["A3_VelocidadPura"] in ["Muy Buena", "Excepcional"] and 
        perfil["A4_ConsistenciaCarrera"] in ["Muy Consistente", "Extremadamente Consistente"]):
        bonificaciones_sinergia += 10
    # BD2
    if (perfil["A6_MentalidadEquipo"] in ["Jugador de Equipo Nato", "Totalmente Alineado con el Equipo"] and
        perfil["A5_FeedbackTecnico"] in ["Fuerte", "Excepcional"]):
        bonificaciones_sinergia += 8
    # BD3
    if ((perfil["A1_Experiencia"] in ["Novato"] or                            
         perfil["A7_EncajeMarca"] in ["Buen Encaje", "Encaje Perfecto"]) and
        (perfil["A8_ExigenciaSalarial"] in ["Salario Muy Bajo", "Salario Bajo"])):
        bonificaciones_sinergia += 6 
    # BD4
    if (perfil["A5_FeedbackTecnico"] == "Fuerte" and
        perfil["A6_MentalidadEquipo"] == "Jugador de Equipo Nato"):
        bonificaciones_sinergia += 9
    # BD5
    if (perfil["A1_Experiencia"] == "Joven Promesa" and
        (perfil["A7_EncajeMarca"] in ["Buen Encaje", "Encaje Perfecto"]) and
        perfil["A8_ExigenciaSalarial"] == "Salario Bajo"):
        bonificaciones_sinergia += 7
    # BD6
    if (perfil["A2_EstiloConduccion"] == "Adaptable Camaleónico" and
        perfil["A4_ConsistenciaCarrera"] == "Muy Consistente"):
        bonificaciones_sinergia += 6
    # BD7 
    if (perfil["A1_Experiencia"] in ["Establecido", "Veterano"] and
        perfil["A5_FeedbackTecnico"] == "Excepcional" and
        perfil["A4_ConsistenciaCarrera"] in ["Muy Consistente", "Extremadamente Consistente"]):
        bonificaciones_sinergia += 10
    # BD8 
    if (perfil["A1_Experiencia"] == "Veterano" and
        perfil["A7_EncajeMarca"] in ["Buen Encaje"] and 
        perfil["A6_MentalidadEquipo"] in ["Jugador de Equipo Nato", "Totalmente Alineado con el Equipo"]):
        bonificaciones_sinergia += 8
    # BD9 
    if ((perfil["A3_VelocidadPura"] == "Excepcional" or perfil["A5_FeedbackTecnico"] == "Excepcional") and
        perfil["A8_ExigenciaSalarial"] == "Salario Medio"):
        bonificaciones_sinergia += 4
        
    # --- C. Reglas de Incompatibilidad (Penalizaciones - INCs) ---
    # INC1
    if (perfil["A1_Experiencia"] == "Veterano" and 
        perfil["A8_ExigenciaSalarial"] in ["Salario Muy Bajo", "Salario Bajo"]):
        penalizaciones += 9 
    # INC2
    if (perfil["A3_VelocidadPura"] == "Excepcional" and 
        perfil["A6_MentalidadEquipo"] == "Totalmente Alineado con el Equipo"):
        penalizaciones += 5
    # INC3
    if (perfil["A1_Experiencia"] == "Novato" and 
        perfil["A5_FeedbackTecnico"] == "Excepcional"):
        penalizaciones += 4
    # INC4
    if (perfil["A6_MentalidadEquipo"] in ["Jugador de Equipo Nato", "Totalmente Alineado con el Equipo"] and 
        perfil["A2_EstiloConduccion"] == "Agresivo Controlado"):
        penalizaciones += 3
    # INC5
    condiciones_elite_inc5 = 0
    if perfil["A3_VelocidadPura"] == "Excepcional": condiciones_elite_inc5 += 1
    if perfil["A4_ConsistenciaCarrera"] == "Extremadamente Consistente": condiciones_elite_inc5 += 1
    if perfil["A5_FeedbackTecnico"] == "Excepcional": condiciones_elite_inc5 += 1
    if perfil["A7_EncajeMarca"] == "Encaje Perfecto": condiciones_elite_inc5 += 1
    if condiciones_elite_inc5 >= 2 and perfil["A8_ExigenciaSalarial"] in ["Salario Muy Bajo", "Salario Bajo"]:
        penalizaciones += 13 
    # INC6
    if (perfil["A2_EstiloConduccion"] == "Adaptable Camaleónico" and 
        perfil["A5_FeedbackTecnico"] == "Limitada"):
        penalizaciones += 4
    # INC7
    if (perfil["A8_ExigenciaSalarial"] == "Salario Alto" and 
        perfil["A7_EncajeMarca"] == "Bajo Encaje"):
        penalizaciones += 5
    # INC8
    if (perfil["A1_Experiencia"] in ["Establecido", "Veterano"] and 
        perfil["A4_ConsistenciaCarrera"] == "Inconsistente"):
        penalizaciones += 4
    # INC9
    if (perfil["A2_EstiloConduccion"] == "Consistente y Calculador" and 
        perfil["A3_VelocidadPura"] == "Excepcional"):
        penalizaciones += 3
    # INC10
    if (perfil["A1_Experiencia"] in ["Novato", "Joven Promesa"] and
        perfil["A5_FeedbackTecnico"] in ["Fuerte", "Excepcional"]):
        penalizaciones += 2
    # INC11
    if (perfil["A3_VelocidadPura"] == "Excepcional" and
        perfil["A4_ConsistenciaCarrera"] == "Extremadamente Consistente"):
        penalizaciones += 3
    # INC12
    if (perfil["A1_Experiencia"] == "Joven Promesa" and
        perfil["A2_EstiloConduccion"] == "Adaptable Camaleónico" and
        perfil["A7_EncajeMarca"] in ["Buen Encaje", "Encaje Perfecto"] and
        (perfil["A3_VelocidadPura"] in ["Muy Buena", "Excepcional"] or 
         perfil["A4_ConsistenciaCarrera"] in ["Muy Consistente", "Extremadamente Consistente"])):
        penalizaciones += 6
    # INC13
    if (perfil["A6_MentalidadEquipo"] in ["Jugador de Equipo Nato", "Totalmente Alineado con el Equipo"] and
        perfil["A3_VelocidadPura"] in ["Muy Buena", "Excepcional"]):
        penalizaciones += 4
    # INC14
    if (perfil["A1_Experiencia"] in ["Novato", "Joven Promesa"] and
        perfil["A3_VelocidadPura"] in ["Muy Buena", "Excepcional"] and
        perfil["A4_ConsistenciaCarrera"] in ["Muy Consistente", "Extremadamente Consistente"] and
        perfil["A8_ExigenciaSalarial"] in ["Salario Muy Bajo", "Salario Bajo"]):
        penalizaciones += 8 
    # INC15
    if (perfil["A1_Experiencia"] == "Veterano" and
        perfil["A3_VelocidadPura"] == "Excepcional" and
        perfil["A8_ExigenciaSalarial"] == "Salario Alto" and
        perfil["A6_MentalidadEquipo"] in ["Equilibrado", "Primariamente Individualista"]):
        penalizaciones += 9
    # INC16
    if (perfil["A1_Experiencia"] in ["Establecido", "Veterano"] and
        perfil["A3_VelocidadPura"] in ["Muy Buena", "Excepcional"] and
        perfil["A4_ConsistenciaCarrera"] in ["Muy Consistente", "Extremadamente Consistente"]):
        penalizaciones += 6
    # INC17
    if (perfil["A1_Experiencia"] in ["Establecido", "Veterano"] and
        perfil["A5_FeedbackTecnico"] in ["Fuerte", "Excepcional"] and
        perfil["A8_ExigenciaSalarial"] in ["Salario Medio", "Salario Alto"]):
        penalizaciones += 7
    # INC18
    if (perfil["A1_Experiencia"] == "Veterano" and
        perfil["A2_EstiloConduccion"] == "Agresivo Controlado"):
        penalizaciones += 5
    # INC19 (Ajustada)
    if (perfil["A2_EstiloConduccion"] == "Agresivo Controlado" and
        perfil["A3_VelocidadPura"] in ["Muy Buena", "Excepcional"] and 
        perfil["A4_ConsistenciaCarrera"] != "Extremadamente Consistente"):
        penalizaciones += 4
    #INC20
    if (perfil["A4_ConsistenciaCarrera"] in ["Extremadamente Consistente","Muy Consistente"] and
        perfil["A8_ExigenciaSalarial"] in ["Salario Muy Bajo", "Salario Bajo"]):
        penalizaciones += 9
    #INC21
    if (perfil["A1_Experiencia"] == "Veterano" and
        perfil["A7_EncajeMarca"] == "Encaje Perfecto"):
        penalizaciones += 5
        
    aptitud_final = (puntaje_base + 
                     bonificaciones_individuales + 
                     bonificaciones_sinergia - 
                     penalizaciones)
    
    return (max(0, aptitud_final),)


def imprimir_perfil_piloto(cromosoma_bits):
    perfil = decodificar_cromosoma(cromosoma_bits)
    print("--- Perfil de Piloto Ideal ---")
    for atributo, valor in perfil.items():
        print(f"{atributo.replace('_', ' ')}: {valor}")
    
    print("-----------------------------")

def evaluar_aptitud_piloto_nueva(cromosoma_bits):
    perfil = decodificar_cromosoma(cromosoma_bits)

    puntaje_base = 0
    bonificaciones_individuales = 0
    bonificaciones_sinergia = 0
    penalizaciones = 0

    # --- Bonificaciones Individuales ---
    #BI1
    if perfil["A1_Experiencia"] == "Joven Promesa": bonificaciones_individuales += 2
    elif perfil["A1_Experiencia"] == "Establecido": bonificaciones_individuales += 1
    elif perfil["A1_Experiencia"] == "Veterano": bonificaciones_individuales += 2
    #BI2
    if perfil["A2_EstiloConduccion"] == "Agresivo Controlado": bonificaciones_individuales += 2
    elif perfil["A2_EstiloConduccion"] == "Adaptable Camaleónico": bonificaciones_individuales += 1
    if perfil["A2_EstiloConduccion"] == "Consistente y Calculador": bonificaciones_individuales += 1
    elif (perfil["A2_EstiloConduccion"] == "Técnico y Metódico" and
        perfil["A5_FeedbackTecnico"] in ["Adecuada", "Fuerte"]): bonificaciones_individuales += 1 
    #BI3
    if perfil["A3_VelocidadPura"] == "Excepcional": bonificaciones_individuales += 3
    elif perfil["A3_VelocidadPura"] == "Muy Buena": bonificaciones_individuales += 2
    #BI4
    if perfil["A4_ConsistenciaCarrera"] == "Extremadamente Consistente": bonificaciones_individuales += 4
    elif perfil["A4_ConsistenciaCarrera"] == "Muy Consistente": bonificaciones_individuales += 2
    if (perfil["A4_ConsistenciaCarrera"] == "Algo Consistente" and
        perfil["A1_Experiencia"] in ["Novato", "Joven Promesa"]): bonificaciones_individuales += 1  # Coherente con curva de aprendizaje, no habria por que sumarle este punto por poca consistencia para un establecido o veterano
    #BI5
    if perfil["A5_FeedbackTecnico"] == "Excepcional": bonificaciones_individuales += 4
    elif perfil["A5_FeedbackTecnico"] == "Fuerte": bonificaciones_individuales += 2
    if perfil["A5_FeedbackTecnico"] == "Adecuada" and perfil["A1_Experiencia"] != "Veterano": bonificaciones_individuales += 1 # no es logico un veterano con mal feedback
    #BI6
    if perfil["A6_MentalidadEquipo"] == "Jugador de Equipo Nato": bonificaciones_individuales += 2
    elif perfil["A6_MentalidadEquipo"] == "Totalmente Alineado con el Equipo": bonificaciones_individuales += 3
    elif (perfil["A6_MentalidadEquipo"] == "Equilibrado" and
        perfil["A2_EstiloConduccion"] != "Agresivo Controlado"): bonificaciones_individuales += 1 # no busca conflicto
    #BI7
    if perfil["A7_EncajeMarca"] == "Encaje Perfecto": bonificaciones_individuales += 3
    elif perfil["A7_EncajeMarca"] == "Buen Encaje": bonificaciones_individuales += 2
    #BI8
    if perfil["A8_ExigenciaSalarial"] == "Salario Muy Bajo": bonificaciones_individuales += 2
    elif perfil["A8_ExigenciaSalarial"] == "Salario Bajo": bonificaciones_individuales += 1

    # --- Bonificaciones por Sinergia ---
    #BD1
    BD1_aplicado = False
    if (perfil["A3_VelocidadPura"] in ["Muy Buena", "Excepcional"] and 
        perfil["A4_ConsistenciaCarrera"] in ["Muy Consistente", "Extremadamente Consistente"]):
        bonificaciones_sinergia += 10
        BD1_aplicado = True
    #BD2
    if (perfil["A5_FeedbackTecnico"] in ["Fuerte", "Excepcional"] and
        perfil["A6_MentalidadEquipo"] in ["Jugador de Equipo Nato", "Totalmente Alineado con el Equipo"]):
        bonificaciones_sinergia += 10
    #BD3
    if (perfil["A1_Experiencia"] in ["Novato", "Joven Promesa"] and
        perfil["A7_EncajeMarca"] in ["Buen Encaje", "Encaje Perfecto"] and
        perfil["A8_ExigenciaSalarial"] in ["Salario Muy Bajo", "Salario Bajo"]):
        bonificaciones_sinergia += 6
    #BD4
    if (perfil["A2_EstiloConduccion"] == "Adaptable Camaleónico" and
        perfil["A4_ConsistenciaCarrera"] == "Muy Consistente" and
        perfil["A5_FeedbackTecnico"] in ["Fuerte", "Excepcional"]):
        bonificaciones_sinergia += 5
    #BD5
    if (perfil["A1_Experiencia"] in ["Establecido", "Veterano"] and
        perfil["A5_FeedbackTecnico"] == "Excepcional" and
        perfil["A4_ConsistenciaCarrera"] in ["Muy Consistente", "Extremadamente Consistente"] and
        perfil["A6_MentalidadEquipo"] in ["Jugador de Equipo Nato", "Totalmente Alineado con el Equipo"]):
        bonificaciones_sinergia += 6
    #BD6
    if (perfil["A1_Experiencia"] == "Veterano" and
        perfil["A7_EncajeMarca"] == "Buen Encaje" and
        perfil["A6_MentalidadEquipo"] in ["Jugador de Equipo Nato", "Totalmente Alineado con el Equipo"]):
        bonificaciones_sinergia += 8
    #BD7
    if ((perfil["A3_VelocidadPura"] == "Excepcional" and perfil["A5_FeedbackTecnico"] == "Excepcional") and
        perfil["A8_ExigenciaSalarial"] == "Salario Medio"):
        bonificaciones_sinergia += 4
    #BD8
    if (perfil["A1_Experiencia"] == "Establecido" and
        perfil["A5_FeedbackTecnico"] == "Fuerte" and
        perfil["A6_MentalidadEquipo"] == "Jugador de Equipo Nato" and
        perfil["A8_ExigenciaSalarial"] == "Salario Medio"):
        bonificaciones_sinergia += 6
    #BD9
    if (perfil["A1_Experiencia"] == "Novato" and
        perfil["A5_FeedbackTecnico"] == "Adecuada" and
        perfil["A6_MentalidadEquipo"] == "Equilibrado" and
        perfil["A8_ExigenciaSalarial"] == "Salario Muy Bajo"):
        bonificaciones_sinergia += 5
    #BD10
    if (perfil["A1_Experiencia"] in ["Joven Promesa", "Establecido"] and
        perfil["A5_FeedbackTecnico"] == "Adecuada" and
        perfil["A6_MentalidadEquipo"] == "Jugador de Equipo Nato" and
        perfil["A8_ExigenciaSalarial"] == "Salario Bajo"):
        bonificaciones_sinergia += 6
    #BD11
    if (perfil["A1_Experiencia"] == "Joven Promesa" and
        perfil["A5_FeedbackTecnico"] == "Adecuada" and
        perfil["A4_ConsistenciaCarrera"] == "Algo Consistente" and
        perfil["A6_MentalidadEquipo"] in ["Jugador de Equipo Nato", "Totalmente Alineado con el Equipo"] and
        perfil["A8_ExigenciaSalarial"] == "Salario Bajo"):
        bonificaciones_sinergia += 5
    #BD12
    if (perfil["A2_EstiloConduccion"] == "Técnico y Metódico" and
        perfil["A5_FeedbackTecnico"] in ["Fuerte", "Adecuada"] and
        perfil["A7_EncajeMarca"] == "Encaje Aceptable"):
        bonificaciones_sinergia += 4
    #BD13
    if (perfil["A6_MentalidadEquipo"] == "Equilibrado" and
        perfil["A5_FeedbackTecnico"] in ["Adecuada", "Fuerte"] and
        perfil["A8_ExigenciaSalarial"] == "Salario Medio" and
        perfil["A1_Experiencia"] in ["Establecido", "Joven Promesa"]):
        bonificaciones_sinergia += 5
    #BD14
    if (perfil["A1_Experiencia"] == "Novato" and
        perfil["A6_MentalidadEquipo"] in ["Equilibrado", "Jugador de Equipo Nato"] and
        perfil["A8_ExigenciaSalarial"] == "Salario Muy Bajo" and
        perfil["A4_ConsistenciaCarrera"] == "Algo Consistente"):
        bonificaciones_sinergia += 4
    #BD15
    if (perfil["A1_Experiencia"] == "Joven Promesa" and
        perfil["A3_VelocidadPura"] in ["Buena", "Muy Buena"] and
        perfil["A5_FeedbackTecnico"] in ["Adecuada", "Fuerte"] and
        perfil["A8_ExigenciaSalarial"] in ["Salario Bajo", "Salario Medio"]):
        bonificaciones_sinergia += 5
    # BD16
    if (perfil["A3_VelocidadPura"] == "Muy Buena" and
        perfil["A5_FeedbackTecnico"] in ["Fuerte", "Excepcional"] and
        perfil["A8_ExigenciaSalarial"] in ["Salario Muy Bajo", "Salario Bajo"]):
        bonificaciones_sinergia += 5

    # --- Penalizaciones agrupadas escalonadas ---
    #INC_AVANZADA1
    penalizador_elite = 0
    if perfil["A3_VelocidadPura"] == "Excepcional": penalizador_elite += 1
    if perfil["A4_ConsistenciaCarrera"] == "Extremadamente Consistente": penalizador_elite += 1
    if perfil["A5_FeedbackTecnico"] == "Excepcional": penalizador_elite += 1
    if perfil["A7_EncajeMarca"] == "Encaje Perfecto": penalizador_elite += 1
    if perfil["A8_ExigenciaSalarial"] in ["Salario Muy Bajo", "Salario Bajo"] and penalizador_elite >= 2:
        penalizaciones += 10 + (penalizador_elite - 2) * 2
    #INC_AVANZADA2 - Penalización por "perfil incoherente"
    penalizador_incoherente = 0
    if perfil["A1_Experiencia"] in ["Establecido", "Veterano"]: penalizador_incoherente += 1
    if perfil["A5_FeedbackTecnico"] in ["Fuerte", "Excepcional"]: penalizador_incoherente += 1
    if perfil["A8_ExigenciaSalarial"] in ["Salario Muy Bajo", "Salario Bajo"]: penalizador_incoherente += 1
    if penalizador_incoherente >= 2:
        penalizaciones += 9 + (penalizador_incoherente - 2) * 2
    #INC3 - Penalización por “perfil Dios”
    if (BD1_aplicado and
        perfil["A3_VelocidadPura"] == "Excepcional" and
        perfil["A4_ConsistenciaCarrera"] == "Extremadamente Consistente"):
        penalizaciones += 6
    #INC4 - veterano perfecto es muy díficil
    if (perfil["A1_Experiencia"] == "Veterano" and
        perfil["A3_VelocidadPura"] in ["Muy Buena", "Excepcional"] and
        perfil["A4_ConsistenciaCarrera"] in ["Muy Consistente", "Extremadamente Consistente"] and
        perfil["A5_FeedbackTecnico"] == "Excepcional" and
        perfil["A6_MentalidadEquipo"] in ["Jugador de Equipo Nato", "Totalmente Alineado con el Equipo"]):
        penalizaciones += 6
    #INC_5
    if (perfil["A2_EstiloConduccion"] == "Adaptable Camaleónico" and
        perfil["A1_Experiencia"] == "Veterano"):
        penalizaciones += 2
   # INC6 – penaliza pilotos con poca experiencia que se adoatan bien
    if perfil["A2_EstiloConduccion"] == "Adaptable Camaleónico" and perfil["A1_Experiencia"] in ["Novato", "Joven Promesa"]:
        penalizaciones += 3
    # INC7 – penaliza salario medio con poco merito
    if (perfil["A8_ExigenciaSalarial"] == "Salario Medio" and
        perfil["A1_Experiencia"] in ["Novato", "Joven Promesa"] and
        perfil["A5_FeedbackTecnico"] in ["Adecuada", "Limitada"]):
        penalizaciones += 4
    # INC8 – penaliza leve por mucho de salario medio
    if perfil["A8_ExigenciaSalarial"] == "Salario Medio":
        penalizaciones += 1
    # INC9 – feedback técnico exagerado en joven promesa
    if perfil["A1_Experiencia"] in ["Novato", "Joven Promesa"] and perfil["A5_FeedbackTecnico"] == "Excepcional":
        penalizaciones += 4
    # INC10 – penaliza a joven promesa perfecto y barato
    if (perfil["A1_Experiencia"] == "Joven Promesa" and
        perfil["A3_VelocidadPura"] in ["Muy Buena", "Excepcional"] and
        perfil["A4_ConsistenciaCarrera"] in ["Muy Consistente", "Extremadamente Consistente"] and
        perfil["A8_ExigenciaSalarial"] in ["Salario Muy Bajo", "Salario Bajo"]):
        penalizaciones += 6
    #INC11
    if (perfil["A1_Experiencia"] == "Establecido" and
        perfil["A2_EstiloConduccion"] == "Adaptable Camaleónico" and
        perfil["A5_FeedbackTecnico"] in ["Limitada", "Adecuada"]):
        penalizaciones += 3
    #INC12
    if (perfil["A8_ExigenciaSalarial"] == "Salario Medio" and
        perfil["A1_Experiencia"] == "Establecido" and
        (perfil["A4_ConsistenciaCarrera"] == "Extremadamente Consistente" or 
        perfil["A3_VelocidadPura"] == "Excepcional") and
        perfil["A5_FeedbackTecnico"] in ["Adecuada", "Fuerte"]):
        penalizaciones += 4
    #INC13 – excepcional sin base tecnica o experiencia solida
    if (perfil["A3_VelocidadPura"] == "Excepcional" and
        perfil["A1_Experiencia"] in ["Novato", "Joven Promesa"] and
        perfil["A5_FeedbackTecnico"] in ["Limitada", "Adecuada"]):
        penalizaciones += 4
    # INC14 – Excepcional sin alineación de equipo (riesgo de individualismo)
    if (perfil["A3_VelocidadPura"] == "Excepcional" and
        perfil["A6_MentalidadEquipo"] in ["Primariamente Individualista", "Equilibrado"]):
        penalizaciones += 3
 
    aptitud_final = max(0, puntaje_base + bonificaciones_individuales + bonificaciones_sinergia - penalizaciones)
    return (aptitud_final,)
