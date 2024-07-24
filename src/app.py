import streamlit as st
import numpy as np
from xgboost import XGBClassifier
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from pickle import dump, load



data_dir = 'src'
try:
    data_path = 'suicide_attacks_filtered.csv'  # el archivo esta en la carpeta src
    data = pd.read_csv(data_path, parse_dates=['date'])
except FileNotFoundError:
    st.error(f"El archivo {data_path} no se encuentra. Asegúrate de que el archivo está en la misma carpeta que este script.")
    st.stop()

# Configuramos la barra side
st.sidebar.image('logo2.png', width=50, use_column_width=True)
st.sidebar.title('Navegación')
options = st.sidebar.radio('Selecciona una página:', 
                          ['Inicio', 'Ataques suicidas en el tiempo', 'Impacto de ataques suicidas', 'Mapa de ataques suicidas', 'Modelo de predicción'])

# Página de Inicio
if options == 'Inicio':
    st.markdown("<h1 style='text-align: center;'>Análisis de Ataques Suicidas Terroristas</h1>", unsafe_allow_html=True)

    
    st.markdown("""
    ### Introducción al Global Terrorism Database (GTD)

    El **Global Terrorism Database (GTD)** es una iniciativa de código abierto mantenida por el National Consortium for the Study of Terrorism and Responses to Terrorism (START). Este dataset incluye información detallada sobre más de **180,000 incidentes terroristas** a nivel global. Los datos cubren una amplia gama de variables clave, como la fecha y ubicación del incidente, armas utilizadas, naturaleza del objetivo, número de víctimas y responsables identificados cuando es posible.

    El GTD no solo documenta eventos terroristas internacionales, sino también incidentes domésticos, proporcionando una visión completa de la violencia terrorista a nivel mundial. Los datos están basados en informes de diversas fuentes mediáticas confiables, asegurando la calidad y veracidad de la información recopilada.

    **Definición de un ataque terrorista en el GTD:**
    Un ataque terrorista se define como el uso amenazante o real de fuerza ilegal y violencia por parte de un actor no estatal para lograr un objetivo político, económico, religioso o social mediante el miedo, la coerción o la intimidación. Los incidentes en el GTD cumplen con los siguientes atributos:
    
    1. **Intencionalidad**: El incidente debe ser el resultado de un cálculo consciente por parte del perpetrador.
    2. **Violencia o amenaza inmediata de violencia**: Incluye daño a la propiedad y violencia contra personas.
    3. **Actores subnacionales**: Los perpetradores deben ser actores no estatales; el GTD no incluye actos de terrorismo estatal.
    """)
    
    st.markdown("""
    ### Planteamiento del problema y objetivo de la investigación

    Los ataques suicidas terroristas representan una amenaza grave y multifacética debido a su alta letalidad, impacto psicológico y complejidad táctica. Comprender sus características es crucial para desarrollar estrategias efectivas de prevención y respuesta. Esto incluye una combinación de inteligencia, seguridad física, programas de desradicalización y cooperación internacional para mitigar y, en última instancia, prevenir estos ataques devastadores. 

    **Características clave de los ataques terroristas suicidas:**

    1. **Motivación y objetivos:**
        - **Sacrificio personal**: Los perpetradores están dispuestos a morir en el ataque, lo que elimina la necesidad de escape y permite ejecutar misiones más arriesgadas y complejas.
        - **Impacto psicológico**: Estos ataques buscan causar terror y desmoralización en la población objetivo debido a su naturaleza letal y aparentemente irracional.
        - **Simbología y propaganda**: A menudo son utilizados por grupos terroristas para simbolizar el compromiso extremo con su causa y para reclutar y radicalizar a nuevos miembros.

    2. **Técnicas y tácticas:**
        - **Carga explosiva personal**: Los atacantes suelen llevar explosivos en su cuerpo o en objetos cercanos (como mochilas o cinturones explosivos).
        - **Vehículos bomba**: En algunos casos, los atacantes suicidas usan vehículos cargados con explosivos para maximizar el daño.
        - **Ataques coordinados**: A veces, múltiples atacantes suicidas actúan de manera coordinada para aumentar el impacto y la complejidad de la respuesta de seguridad.

    3. **Selección de objetivos:**
        - **Lugares públicos**: Los atacantes suelen dirigirse a lugares con alta concentración de personas, como mercados, estaciones de transporte, conciertos y eventos deportivos.
        - **Objetivos simbólicos**: También pueden elegir lugares con alto valor simbólico, como embajadas, edificios gubernamentales, instalaciones militares y sitios religiosos.
        - **Infraestructuras críticas**: En ocasiones, los ataques se dirigen a infraestructuras críticas como aeropuertos, puertos y plantas de energía.

    4. **Planificación y ejecución:**
        - **Reclutamiento y entrenamiento**: Los atacantes suicidas a menudo pasan por un proceso de radicalización y entrenamiento intensivo, tanto psicológico como técnico.
        - **Logística y apoyo**: Requieren una red de apoyo para proporcionar explosivos, transporte, inteligencia sobre el objetivo y, en algunos casos, refugio antes del ataque.
        - **Innovación táctica**: Los métodos y técnicas evolucionan para adaptarse a las medidas de seguridad, utilizando tecnologías y tácticas que maximicen el éxito del ataque.

    5. **Perfil de los atacantes:**
        - **Diversidad demográfica**: Los atacantes pueden provenir de diversas edades, géneros y orígenes socioeconómicos, aunque comúnmente son jóvenes.
        - **Motivaciones ideológicas**: Las motivaciones suelen estar profundamente arraigadas en creencias ideológicas, religiosas o políticas extremas.
        - **Estado mental**: Los perpetradores pueden estar sujetos a adoctrinamiento intenso, manipulación emocional y presión social, que los lleva a ver el ataque como un acto de martirio o sacrificio heroico.

    6. **Impacto y consecuencias:**
        - **Alta letalidad**: Estos ataques suelen resultar en un alto número de víctimas mortales y heridos debido a la proximidad del atacante a los objetivos.
        - **Desestabilización social y política**: Pueden desencadenar respuestas desproporcionadas de las fuerzas de seguridad y generar tensión política y social.
        - **Cobertura mediática**: La naturaleza extrema y dramática de estos ataques atrae una amplia cobertura mediática, amplificando su impacto psicológico y propagandístico.

    7. **Prevención y contramedidas:**
        - **Inteligencia y monitoreo**: La identificación temprana de redes de radicalización y la vigilancia de potenciales atacantes son cruciales.
        - **Seguridad física**: Mejora de la seguridad en lugares públicos y objetivos potenciales mediante controles de acceso y dispositivos de detección de explosivos.
        - **Programas de desradicalización**: Iniciativas para prevenir la radicalización y reintegrar a individuos que han sido adoctrinados.

    **Importancia del estudio:**

    1. **Protección de vidas humanas:**
        - **Minimización de pérdidas**: Prevenir ataques suicidas salva vidas y reduce el número de víctimas.
        - **Prevención de daños colaterales**: Además de las víctimas directas, los ataques pueden causar daños colaterales significativos.

    2. **Estabilidad y seguridad nacional:**
        - **Prevención de la desestabilización**: Los ataques suicidas pueden desestabilizar regiones enteras, creando caos y desconfianza en las instituciones de seguridad.
        - **Fortalecimiento de la confianza pública**: La capacidad de prevenir ataques refuerza la confianza del público en el gobierno y las fuerzas de seguridad.

    3. **Impacto psicológico y social:**
        - **Reducción del miedo y la ansiedad**: Prevenir estos ataques ayuda a mantener la moral pública y la tranquilidad.
        - **Cohesión social**: Evitar ataques terroristas ayuda a prevenir divisiones sociales y étnicas.

    4. **Impacto económico:**
        - **Protección de infraestructura**: La prevención protege activos esenciales como transportes públicos y centros comerciales.
        - **Reducción de costos**: La prevención ahorra costos asociados con emergencias, reparaciones y pérdidas económicas.

    5. **Prevención de radicalización:**
        - **Interrupción de cadenas de reclutamiento**: Desmantelar redes terroristas y programas de desradicalización.
        - **Educación y concientización**: Programas educativos que disuadan a potenciales terroristas suicidas.

    6. **Cooperación internacional:**
        - **Fortalecimiento de la colaboración**: Requiere cooperación internacional, compartiendo inteligencia y estrategias entre países.
        - **Cumplimiento de obligaciones globales**: Participar en la prevención de ataques terroristas ayuda a cumplir responsabilidades en seguridad global y derechos humanos.
    """)

# Página 1: Ataques suicidas en el tiempo
elif options == 'Ataques suicidas en el tiempo':
    st.markdown("<h1 style='text-align: center;'>Ataques Suicidas a lo Largo del Tiempo</h1>", unsafe_allow_html=True)
    imagen_ataquespy = st.image('Suicide attacks per year.png') 

    st.markdown("""
    ### Ataques suicidas por año:
    Este gráfico muestra el número de ataques suicidas registrados anualmente desde 1981 hasta 2017. Aquí hay algunos puntos clave a tener en cuenta:

    - **Período del Dataset**: Los datos abarcan desde 1981 hasta 2017.
    - **Observación 1**: A partir de 1981, los registros de ataques suicidas comienzan a aparecer en el dataset.
    - **Observación 2**: Se observa un notable incremento de ataques a partir del año 2000, posiblemente vinculado con los eventos del 11 de septiembre de 2001 (9/11) y el consecuente aumento de actividades terroristas a nivel global.
    - **Observación 3**: Hay un crecimiento sostenido en el número de ataques suicidas, alcanzando un pico de 985 ataques en el año 2016.
    """)


    #Agregamos aqui el grafico con la prediccion hecha con modelo Prophet
    st.markdown("<h1 style='text-align: center;'>Predicción de Ataques Suicidas</h1>", unsafe_allow_html=True)
    imagen = st.image('PREDICCION.png')   
    
    st.markdown("""
    ### Predicción de ataques suicidas:
    Este gráfico presenta una predicción de futuros ataques suicidas basada en el modelo Prophet. Aquí hay algunas observaciones clave:

    - **Modelo Utilizado**: La predicción se ha realizado utilizando el modelo Prophet, conocido por su capacidad para manejar series temporales con componentes estacionales y de tendencia.
    - **Observación 1**: La predicción sugiere una tendencia continua de incremento en la actividad de ataques suicidas en el futuro cercano.
    - **Observación 2**: Los intervalos de confianza reflejan la incertidumbre inherente a las predicciones. Esto indica que los números pueden variar significativamente dependiendo de múltiples factores, como cambios en las tácticas terroristas o mejoras en las medidas de seguridad y prevención.
    - **Observación 3**: Es crucial considerar que las predicciones están sujetas a cambios bruscos si los grupos implementan métodos aún más mortíferos o, por el contrario, si las medidas de seguridad y prevención se vuelven más efectivas.
    """)

#cargo el csv de forecast pra la tabla
    forecast_path = 'forecast_prophet.csv'
    forecast_df = pd.read_csv(forecast_path)

#mostramos solo el ano
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds']).dt.year + 1
    forecast_df = forecast_df.rename(columns={
        'ds': 'Año',
        'yhat': 'Predicción',
        'yhat_lower': 'Límite Inferior',
        'yhat_upper': 'Límite Superior'
    })
    
#mostramos la tabla
    st.markdown("""
    ### Predicción de ataques suicidas para los próximos años
    A continuación se muestra la predicción del número de ataques suicidas para los próximos años utilizando el modelo Prophet.
    """)

    # Mostrar el DataFrame como tabla
    st.write(forecast_df)
    
    st.markdown("""
    ### Leyenda:

    - **Año**: El año correspondiente a la predicción.
    - **Predicción**: El número estimado de ataques suicidas para el año dado.
    - **Límite Inferior**: El límite inferior del intervalo de confianza para la predicción.
    - **Límite Superior**: El límite superior del intervalo de confianza para la predicción.
    """)

# Página 2: Impacto de ataques y ataques más mortíferos por grupo
elif options == 'Impacto de ataques suicidas':
    st.markdown("<h1 style='text-align: center;'>Impacto de los Ataques Suicidas</h1>", unsafe_allow_html=True)

    # Impacto 
    st.subheader("Impacto total por grupo terrorista")

    # Calcular impacto total (muertes + heridos) por grupo terrorista
    impact_df = data.groupby('gname')[['nkill', 'nwound']].sum()
    impact_df['total_impact'] = impact_df['nkill'] + impact_df['nwound']

    # Obtener los 5 grupos terroristas más letales
    top_5_groups = impact_df.sort_values(by='total_impact', ascending=False).head(5)

    # Graficar el impacto de los 5 grupos más letales
    fig_impact, ax = plt.subplots(figsize=(10, 6))
    top_5_groups[['nkill', 'nwound']].rename(columns={'nkill': 'Número de Muertos', 'nwound': 'Número de Heridos'}).plot(
        kind='bar', 
        stacked=True, 
        ax=ax, 
        color=['lightblue', 'grey']
    )
    plt.xlabel('Grupo Terrorista')
    plt.ylabel('Número de Personas')
    plt.title('Muertes y Heridos por los 5 Grupos más Letales')
    plt.grid(True)
    st.pyplot(fig_impact)
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Ataques Más Mortíferos por Grupo Terrorista
    st.subheader("Top 5 Ataques más mortíferos por grupo terrorista")
    
    # Cargar los datos desde el archivo CSV
    deadly_attacks_path = 'top_5_deadly_attacks.csv'
    top_5_deadly_attacks = pd.read_csv(deadly_attacks_path)

    # Graficar los datos
    fig_deadly_attacks, ax = plt.subplots(figsize=(12, 10))
    bar_width = 0.4

    # Posiciones para las barras
    r1 = range(len(top_5_deadly_attacks))
    r2 = [x + bar_width for x in r1]

    # Crear las barras
    ax.bar(r1, top_5_deadly_attacks['nkill'], color='lightblue', width=bar_width, edgecolor='grey', label='Número de Muertos')
    ax.bar(r2, top_5_deadly_attacks['nwound'], color='grey', width=bar_width, edgecolor='grey', label='Número de Heridos')

    # Añadir etiquetas
    ax.set_xlabel('Grupo Terrorista', fontweight='bold')
    ax.set_xticks([r + bar_width / 2 for r in range(len(top_5_deadly_attacks))])
    ax.set_xticklabels(top_5_deadly_attacks['gname'], rotation=45, ha='right')
    ax.set_ylabel('Número de Personas', fontweight='bold')
    ax.set_title('Top 5 Ataques Más Mortíferos por Grupo Terrorista', fontweight='bold')

    # Añadir leyenda
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig_deadly_attacks)



# Página 3: Mapa de ataques suicidas
elif options == 'Mapa de ataques suicidas':
    st.title('Mapa de ataques suicidas')

# latitud y longitud tienen NaN, por lo cual vamos a dropear esos NaN
    data = data.dropna(subset=['latitude', 'longitude'])

# Vamos a contar los ataques para cada lat y long
    location_counts = data.groupby(['latitude', 'longitude']).size().reset_index(name='count')
    # el mapa se crea centrado en una ubicación inicial
    map_center = [location_counts['latitude'].mean(), location_counts['longitude'].mean()]
    mapa = folium.Map(location=map_center, zoom_start=2)

     # Add marcadores para cada location con size y color basado en la q de ataques
    for _, row in location_counts.iterrows():
        # Ajusta el tamaño y el color según la cantidad de ataques
        radius = min(row['count'] / 10, 20)  # Tamaño máximo del marcador
        color = 'green' if row['count'] < 10 else 'yellow' if row['count'] < 50 else 'red'

        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=radius,
            popup=f'Location: ({row["latitude"]}, {row["longitude"]})<br>Number of Attacks: {row["count"]}',
            color=color,
            fill=True,
            fill_color=color
        ).add_to(mapa)

#https://realpython.com/python-folium-web-maps-from-data
# Mostrar el mapa en Streamlit

    folium_static(mapa)

    st.markdown("""

    ### Cómo leer el mapa de ataques suicidas

    En este mapa, cada marcador representa una ubicación donde se han registrado ataques suicidas. Los detalles que debes tener en cuenta son:

    - **Colores de los marcadores**:
      - **Verde**: Indica que la ubicación ha tenido un número relativamente bajo de ataques suicidas (menos de 10). Estas zonas tienen una menor densidad de ataques.
      - **Amarillo**: Representa ubicaciones con un número moderado de ataques suicidas (entre 10 y 50). Estas áreas tienen una cantidad intermedia de ataques.
      - **Rojo**: Señala ubicaciones con un número alto de ataques suicidas (50 o más). Estas zonas tienen una alta densidad de ataques.
  
                
    - **Tamaño de los Marcadores**:
      - El tamaño del marcador aumenta con el número de ataques. Esto ayuda a visualizar no solo la ubicación, sino también la magnitud de los ataques en el área.
      - Los marcadores más grandes indican una mayor cantidad de ataques, mientras que los más pequeños indican menos ataques.

    Para obtener más información sobre una ubicación específica, puedes hacer clic en el marcador correspondiente para ver un popup con detalles sobre la ubicación y la cantidad de ataques reportados.

""")    

# Página 4: Modelo de predicción    
elif options == 'Modelo de predicción':
    st.markdown("<h1 style='text-align: center;'>Modelo de predicción</h1>", unsafe_allow_html=True)

#cargmaos el modelo
    model_path = "bestxgb2.pkl"
    xgb_model = load(open(model_path, "rb"))
    
#cargmaos el mapeo de variables es lenguaje humano
    mappings_path = "mappings2.pkl"
    country_map, attacktype_map, targtype_map, gname_map, ideology_map = load(open(mappings_path, "rb"))

#se hace la inversion de los mapeos para get los valores # a partir de las selecciones
    inv_country_map = {v: k for k, v in country_map.items()}
    inv_attacktype_map = {v: k for k, v in attacktype_map.items()}
    inv_targtype_map = {v: k for k, v in targtype_map.items()}
    inv_gname_map = {v: k for k, v in gname_map.items()}
    inv_ideology_map = {v: k for k, v in ideology_map.items()}

#configuramos las listas desplegables en Streamlit
    st.title('Predicción de ataques suicidas con XGBoost')

    selected_country = st.selectbox('Seleccione el país', list(country_map.values()))
    selected_attacktype = st.selectbox('Seleccione el tipo de ataque', list(attacktype_map.values()))
    selected_targtype = st.selectbox('Seleccione el tipo de objetivo', list(targtype_map.values()))
    selected_gname = st.selectbox('Seleccione el grupo terrorista', list(gname_map.values()))
    selected_ideology = st.selectbox('¿Tiene motivación ideológica el ataque?', list(ideology_map.values()))
    total_victims = st.number_input('Número total de víctimas', min_value=0, step=1)

#convertimos las selecciones a sus valores factorizados - que lio 
    input_data = {
        'country_txt_num': inv_country_map[selected_country],
        'attacktype1_txt_num': inv_attacktype_map[selected_attacktype],
        'targtype1_txt_num': inv_targtype_map[selected_targtype],
        'gname_num': inv_gname_map[selected_gname],
        'INT_IDEO_num': inv_ideology_map[selected_ideology],
        'totales': total_victims
    }

# Convertir el diccionario a un DataFrame
    input_df = pd.DataFrame([input_data])

# Realizar la predicción
    prediction = xgb_model.predict(input_df)

  #convertimo la predicción a si o no
    prediction_text = "Sí" if prediction[0] == 1 else "No"

    # mostramos lo predecido 
    st.markdown(
        f"<h2 style='text-align: center; font-size: 48px; color: #ff4b4b;'>¿Predecimos un ataque suicida? {prediction_text}</h2>",
        unsafe_allow_html=True
    )
#agregamos algo asi como un exto de warning por el resultado    
    st.markdown(
        """
        **Nota:** Esta predicción está basada en datos históricos y un modelo estadístico. No debe ser considerada como una certeza absoluta.
        Los resultados son indicativos y deben ser utilizados con precaución. Se recomienda realizar un análisis más profundo y considerar 
        múltiples factores antes de tomar cualquier decisión basada en esta predicción.
        """
    )