from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from openai import OpenAI
from PIL import Image
from datetime import datetime
from langsmith import traceable,Client
import folium
from streamlit_folium import st_folium
import json

import streamlit as st
import os
import matplotlib.pyplot as plt 
import io
import base64



now = datetime.now()

# LangChain
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

# OPENAI
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

host = st.secrets["postgres"]["host"]
port = st.secrets["postgres"]["port"]
user = st.secrets["postgres"]["user"]
password = st.secrets["postgres"]["password"]
database = st.secrets["postgres"]["database"]


#client = Client()

def init_database()-> SQLDatabase:
    #db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    db_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)


def get_sql_chain(db):
    template = """
    Tu es un data analyst travaillant pour une entreprise.
    Tu échanges avec un utilisateur qui te pose des questions sur la base de données spatial (postgis) de l'entreprise.

    À partir du schéma des tables ci-dessous, écris une requête SQL qui permettrait de répondre à la question de l'utilisateur.
    Tiens également compte de l'historique de la conversation pour formuler ta réponse.

    Si la question concerne la temporalité, la date actuelle est : {current_date}.

    ⚠️ IMPORTANT — RÈGLES POUR SUPABASE :
    - N'utilise JAMAIS ST_DistanceSphere().
    - Pour calculer des distances réelles en mètres, utilise : 
    ST_Distance(geom::geography, geom::geography)
    - Pour calculer un rayon autour d'un point, utilise aussi ST_Distance(...::geography).
    - Toujours caster les géométries en ::geography avant ST_Distance.
    - Toujours renvoyer une REQUÊTE SQL VALIDE SUPABASE.

    <SCHEMA>{schema}</SCHEMA>

    Historique de la conversation : {chat_history}

    Rédige uniquement la requête SQL — sans aucun texte explicatif, sans commentaire et sans backticks.

    Exemple :
    Question : Trouver les bureaux dans un rayon de 100 km autour de Paris ?
    Requête SQL : 
    SELECT o.name, c.name AS city
    FROM offices o
    JOIN cities c ON o.city_id = c.id
    WHERE ST_DistanceSphere(o.geom, ST_GeomFromText('POINT(2.3522 48.8566)', 4326)) < 100000;


    Question : Trouver les clients à moins de 50 km du bureau de Lyon.
    Requête SQL : 
    SELECT cl.name, cl.revenue
    FROM clients cl
    JOIN offices o ON cl.office_id = o.id
    WHERE o.name = 'Lyon Center'
    AND ST_DistanceSphere(cl.geom, o.geom) < 50000;

    Question : Calculer la distance entre Paris et Marseille.
    Requête SQL : 
    SELECT ST_DistanceSphere(
        (SELECT geom FROM cities WHERE name = 'Paris'),
        (SELECT geom FROM cities WHERE name = 'Marseille')
    ) / 1000 AS distance_km;

    À ton tour :

    Question : {question}
    Requête SQL :
    """
    prompt = ChatPromptTemplate.from_template(template)

    #llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    llm= ChatOpenAI(model= "gpt-4o-mini")
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()

    )
    
def get_rep(user_query: str, chat_history: list):
    prompt = """
    Tu es un spécialiste dans le sujet de la base de donnée qui est à ta disposition. Analyse la demande utilisateur et réponds UNIQUEMENT par :
    - "sql" si la question nécessite une requête SQL sur la base, c'est une base spatiale postgis.
    - "image" si l'utilisateur veut une image, carte, schéma, visualisation
    - "map" si l'utilisateur veut une carte, visualisation géographique
    Historique : {chat_history}
    Question : {question}
    Réponse :"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template(prompt)
    chain = prompt | llm | StrOutputParser()
    reponsee = chain.invoke({"question": user_query, "chat_history": chat_history})
    return reponsee.strip().lower()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def generate_graph_from_prompt(prompt, db): #c'est bon normalement
    besoins =get_sql_chain(db)
    full_prompt = f"""
    Génère uniquement du code Python utilisant matplotlib, SANS texte autour. 
    LE CODE DOIT ÊTRE IMMÉDIATEMENT EXÉCUTABLE.
    Toute réponse doit être du code Python brut uniquement. Aucun texte, aucun Markdown, aucune balise ``` autorisée.
    IMPORTANT :
    - Tu dois impérativement utiliser SQLAlchemy pour exécuter la requête SQL retournée par {besoins}.
    - Interdiction ABSOLUE d utiliser sqlite3.
    - La base de données est PostgreSQL, déjà configurée et accessible via la variable `db` passée dans l environnement.
    - Pour exécuter la requête : utilise db._engine (un engine SQLAlchemy valide).
    Utilise ce modèle :
    import pandas as pd
    df = pd.read_sql(query, db._engine)
    Ensuite génère le graphique avec matplotlib.
    Le graphique doit répondre à :
    {prompt}
    Règles :
    - Aucun texte hors code
    - Aucune balise Markdown
    - Aucune donnée inventée : tout provient de la base de données
    - Code immédiatement exécutable
    Utilise uniquement le schéma réel suivant (ne jamais inventer de colonnes ou tables) :
    {db.get_table_info()}
    """
    answer=client.responses.create(
        model="gpt-4o-mini", 
        input=full_prompt
    )
    code = answer.output_text
    local_vars={}
    exec(code,{"plt":plt,"io":io, "db":db}, local_vars)
    buf=io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    img_base64_str = "data:image/png;base64," + img_base64
    return img_base64_str

def genere_titre(prompt,db): #c'est bon c'est validé
    besoins =get_sql_chain(db)
    pprompt = f"""
    T'es un spécialiste dans le sujet de la base de données qu'on t'a fournis 
    et t'as besoins d'écrire un titre simple et concis pour un graphique basé sur le contenue de la demande suivante :
    {prompt}
    Le titre doit être court, clair et pertinent par rapport à la demande et doive refléter le contenu du graphique basé sur: 
    {besoins}
    """
    aanswer=client.responses.create(
        model="gpt-4o-mini", 
        input=pprompt
    )
    titre = aanswer.output_text  
    print("et pour le titre ?")
    return titre



def get_response(user_query : str, db: SQLDatabase, chat_history: list):
    route = get_rep(user_query, chat_history)
    
    if route == "map":
        return {"type": "map", "query": user_query}
    
    if route == "image":
       url = generate_graph_from_prompt(user_query,db)
       return url

    sql_chain = get_sql_chain(db)

    template = """
    Tu es un data analyst travaillant pour une entreprise.  
    Tu échanges avec un utilisateur qui te pose des questions sur la base de données spatial (postgis) de l'entreprise.

    Si la question concerne la temporalité, la date actuelle est : {current_date}.

    En te basant sur :
    - le schéma des tables ci-dessous,  
    - la question de l'utilisateur,  
    - la requête SQL générée,  
    - et le résultat de cette requête,  

    rédige une **réponse claire et naturelle** en français, adaptée à l'utilisateur. Donne aussi la requete sql en fin de réponse.

    <SCHEMA>{schema}</SCHEMA>

    Historique de la conversation : {chat_history}  
    Requête SQL : <SQL>{query}</SQL>  
    Question de l'utilisateur : {question}  
    Résultat SQL : {response}"""

    prompt = ChatPromptTemplate.from_template(template)

    #llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    llm= ChatOpenAI(model= "gpt-4o-mini")

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars : db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
        "current_date": now,

    })

def get_geojson_chain(db):
    def get_schema(_):
        return db.get_table_info()
    
    template = """
Tu es un data analyst travaillant pour une entreprise.
Génère une requête SQL qui renvoie des données au format GeoJSON.

IMPORTANT : 
- Rédige UNIQUEMENT la requête SQL, SANS backticks
- La requête doit retourner UNE SEULE colonne nommée 'geojson'
- Utilise json_build_object et json_agg pour construire le GeoJSON

Schéma : <SCHEMA>{schema}</SCHEMA>

Modèle à suivre EXACTEMENT :
SELECT json_build_object(
    'type', 'FeatureCollection',
    'features', json_agg(
        json_build_object(
            'type', 'Feature',
            'geometry', ST_AsGeoJSON(geom)::json,
            'properties', json_build_object('name', name)
        )
    )
) AS geojson
FROM nom_table;

Question : {question}
Requête SQL :
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )


def display_schema(db: SQLDatabase):
    def get_schema(_):
        return db.get_table_info()
    template = """
    Voici le schéma des tables de la base de données :
    <SCHEMA>{schema}</SCHEMA>
    Rédige une courte et concise présentation de cette base de données en français. Pas besoin d'exemples ou de détails techniques.
    Présente la de façon claire, structurée et ergonomique.
    Par exemple, noms des tables et colonnes avec une courte description en langage naturel.
    """
    prompt= ChatPromptTemplate.from_template(template)
    llm= ChatOpenAI(model= "gpt-4o-mini")
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )
def clean_sql_query(query: str) -> str:
    """Nettoie la requête SQL en retirant les backticks et espaces superflus"""
    query = query.replace("```sql", "").replace("```", "").strip()
    return query


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Bonjour! Je suis un assistant SQL. Demande moi ce que tu veux sur ta base de données")
    ]
if "schema_display" not in st.session_state:
    st.session_state.schema_display = None
if "maps_data" not in st.session_state:
    st.session_state.maps_data = {}



st.set_page_config(page_title="Discute avec ta base de données", page_icon="💬")
st.title("Discute avec ta base de données")

with st.sidebar:
    st.subheader("Paramètres")
    st.write("C'est une simple application de discussion utilisant SQL. Connectez vous à la base de données pour commencer la discussion")
    if st.button("Connection"):
        with st.spinner("Connection à la base de données..."):
            db = init_database()
            st.session_state.db=db
            st.success("Connecté à la base de données!")
            st.session_state.schema_display = display_schema(st.session_state.db).invoke({})
    if st.session_state.schema_display:
        st.markdown(st.session_state.schema_display)


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            if isinstance(message.content,str)and message.content.startswith("data:image/png;base64,"):
                image_data = message.content.split(",")[1]
                image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                st.image(image, caption="")
            # Si c'est une carte stockée
            # Si c'est une carte stockée (content peut être dict ou [dict])
            elif (isinstance(message.content, dict) and message.content.get("type") == "stored_map") or (
                isinstance(message.content, list) and len(message.content) > 0 and isinstance(message.content[0], dict) and message.content[0].get("type") == "stored_map"
            ):
                content = message.content[0] if isinstance(message.content, list) else message.content
                map_id = message.content.get("map_id")
                if map_id in st.session_state.maps_data:
                    map_data = st.session_state.maps_data[map_id]
                    
                    # Recréer la carte avec les données stockées
                    m = folium.Map(location=[46.603354, 1.888334], zoom_start=6)
                    folium.GeoJson(
                        map_data["geojson"],
                        name="geojson",
                        tooltip=folium.GeoJsonTooltip(
                            fields=['name'] if 'features' in map_data["geojson"] and len(map_data["geojson"].get('features', [])) > 0 else [],
                            aliases=['Nom:'],
                            localize=True
                        )
                    ).add_to(m)
                    
                    st_folium(m, width=700, height=500, key=f"map_{map_id}")
                    st.markdown(f"### {map_data['titre']}")
                else:
                    st.markdown(f"### {message.content.get('titre', 'Carte')}")
                    st.info("(Carte non disponible)")
            else:
                st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Ecrivez un message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        response = get_response(user_query,st.session_state.db, st.session_state.chat_history)
        
        # Gestion des images
        if isinstance(response, str) and response.startswith("data:image/png;base64,"):
            st.image(response, caption="")
        
        # Gestion des cartes
        elif isinstance(response, dict) and response.get("type") == "map":
            try:
                titre = genere_titre(user_query, st.session_state.db)
                
                # Générer et nettoyer la requête SQL
                geojson_chain = get_geojson_chain(st.session_state.db)
                json_sql_query = geojson_chain.invoke({"question": user_query})
                json_sql_query = clean_sql_query(json_sql_query)
                
                # Debug : afficher la requête
                with st.expander("🔍 Voir la requête SQL générée"):
                    st.code(json_sql_query, language="sql")
                
                # Exécuter la requête
                result = st.session_state.db.run(json_sql_query)
                
                # Debug : afficher le résultat brut
                with st.expander("🔍 Voir le résultat brut de la base"):
                    st.write(f"**Type:** `{type(result)}`")
                    st.write(f"**Contenu:** `{repr(result)}`")
                
                # Parser le résultat
                try:
                    import ast
                    
                    # Cas 1 : C'est une string représentant une structure Python
                    if isinstance(result, str):
                        result = result.strip()
                        
                        # Si ça ressemble à une structure Python (commence par [( ou [{)
                        if result.startswith("[") or result.startswith("("):
                            python_obj = ast.literal_eval(result)
                            
                            # Extraire le GeoJSON de la structure
                            if isinstance(python_obj, list) and len(python_obj) > 0:
                                if isinstance(python_obj[0], tuple):
                                    geojson_data = python_obj[0][0]
                                else:
                                    geojson_data = python_obj[0]
                            else:
                                geojson_data = python_obj
                        else:
                            # Sinon c'est du JSON pur
                            geojson_data = json.loads(result)
                    
                    # Cas 2 : C'est déjà une liste Python
                    elif isinstance(result, list):
                        if len(result) > 0:
                            if isinstance(result[0], tuple):
                                geojson_data = result[0][0]
                            else:
                                geojson_data = result[0]
                        else:
                            raise ValueError("Aucune donnée retournée")
                    
                    # Cas 3 : C'est déjà un dict
                    elif isinstance(result, dict):
                        geojson_data = result
                    
                    else:
                        raise ValueError(f"Type non supporté: {type(result)}")
                    
                    # Vérifier que c'est bien un dict
                    if not isinstance(geojson_data, dict):
                        st.error(f"Le résultat n'est pas un dictionnaire: {type(geojson_data)}")
                        raise ValueError(f"Format invalide: {type(geojson_data)}")
                        
                except Exception as parse_error:
                    st.error(f"Erreur lors du parsing: {str(parse_error)}")
                    raise
                
                # Vérifier la structure du GeoJSON
                with st.expander("🔍 Voir le GeoJSON parsé"):
                    st.json(geojson_data)
                
                # Créer la carte
                m = folium.Map(location=[46.603354, 1.888334], zoom_start=6)
                
                # Ajouter le GeoJSON
                folium.GeoJson(
                    geojson_data,
                    name="geojson",
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name'] if 'features' in geojson_data and len(geojson_data.get('features', [])) > 0 else [],
                        aliases=['Nom:'],
                        localize=True
                    )
                ).add_to(m)
                
                # Afficher la carte
                st_folium(m, width=700, height=500)
                st.markdown(f"### {titre}")

                # IMPORTANT : Stocker les données de la carte pour pouvoir la réafficher
                map_id = len(st.session_state.chat_history)
                st.session_state.maps_data[map_id] = {
                    "geojson": geojson_data,
                    "titre": titre
                }
                
                # Créer un objet de réponse spécial pour les cartes
                response = {"type": "stored_map", "map_id": map_id, "titre": titre}
                
            except json.JSONDecodeError as e:
                st.error(f"❌ Erreur de parsing JSON à la position {e.pos}")
                st.write(f"**Message d'erreur:** {str(e)}")
                if 'result' in locals():
                    st.write(f"**Résultat brut:** `{repr(result)[:500]}`")
                response = "Erreur : Le résultat de la base n'est pas un JSON valide"
                
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")
                import traceback
                with st.expander("Voir le détail de l'erreur"):
                    st.code(traceback.format_exc())
                response = f"Erreur lors de la génération de la carte"
        
        # Réponse textuelle normale
        else:
            st.markdown(response)
    if isinstance(response, dict):
        st.session_state.chat_history.append(AIMessage(content=[response]))
    else:
        st.session_state.chat_history.append(AIMessage(content=response))

