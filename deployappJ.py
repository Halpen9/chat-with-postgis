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
    - Pour calculer un rayon autour d’un point, utilise aussi ST_Distance(...::geography).
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
    - "sql" si la question nécessite une requête SQL sur la base
    - "image" si l'utilisateur veut une image, carte, schéma, visualisation
    - "chat" pour toute réponse en langage naturel
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
    answer=client.responses.create( model="gpt-4o-mini", input=full_prompt)
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

def generate_map_from_prompt(prompt, db): #je ne sais pas si c'est bon je n'ai pas encore teste
    besoins =get_sql_chain(db)
    map_prompt = f"""
    Génère uniquement du code Python utilisant matplotlib (AUCUN texte autour, AUCUN Markdown, AUCUNE balise).
    LE CODE DOIT ÊTRE IMMÉDIATEMENT EXÉCUTABLE.
    Contraintes obligatoires :
    - Tu dois IMPERATIVEMENT utiliser SQLAlchemy pour exécuter la requête SQL retournée par {besoins}.
    - La base de données est PostgreSQL et st accessible grâce la variable db passée dans l'environnement.
    - Pour exécuter la requête, utilise l'engine SQLAlchemy disponible : db._engine.
    Utilise exactement ce modèle pour charger les données :
     import pandas as pd
     df = pd.read_sql(query, db._engine)
    AUCUNE DONNEE INVENTEE : tout provient de la base de données.
    Code immédiatement exécutable (importations nécessaires incluses).
    - Tu dois analyser le schéma réel fourni par {db.get_table_info()} et construire une requête SQL valide en fonction des tables qui existent — ne jamais inventer de noms de table ou de colonne.
    - Objectif : générer une carte (map) avec matplotlib selon les instructions contenues dans {prompt}.
    Spécifications fonctionnelles (le code doit implémenter ces vérifications et comportements) :
    Analyser le texte renvoyé par {db.get_table_info()} pour identifier les tables et colonnes disponibles. 
    Déterminer automatiquement quelles colonnes utiliser pour les coordonnées géographiques en cherchant parmi les noms courants (par exemple : latitude, lat, y, longitude, lon, lng, x) ou une colonne géométrique nommée geom/geometry.
    Si aucune colonne de coordonnées n'est trouvée, le script doit lever une erreur Python claire (par exemple ValueError) indiquant que la table ne contient pas de coordonnées et expliquer brièvement quelles colonnes attendues (liste de noms) faire apparaître dans la base pour que la génération soit possible.
    Choisir une colonne numérique pour la coloration (couleur par valeur) si disponible ; sinon tracer simplement les points. Les noms recherchés pour la valeur peuvent inclure : value, count, measure, pop, density, etc. Si aucune colonne numérique n'existe, continuer en traçant des points simples.
    Construire une requête SQL sûre (SELECT explicite) en utilisant uniquement les tables/colonnes du schéma fourni. La requête finale doit être stockée dans la variable query avant l'appel à pd.read_sql.
    Charger les données avec pd.read_sql(query, db._engine).
    Générer la carte avec matplotlib :
    Tracer les points longitude (x) / latitude (y) correctement orientés.
    Si une colonne numérique est disponible, utiliser un scatter plot avec colorbar (échelle de couleurs) représentant cette valeur.
    Ajouter axes, titre minimal (sous forme de variable dans le code — si le {prompt} indique un titre, l'utiliser), et une légende/échelle de couleurs quand pertinent.
    Gérer les cas où il y a très peu de points (par ex. < 2) en adaptant les tailles/limites d'axes.
    Tout le code doit être autonome : inclure import nécessaires (pandas, matplotlib.pyplot, éventuellement numpy), mais ne pas utiliser de bibliothèques géospatiales externes qui pourraient ne pas être installées (sauf si explicitement demandé dans {prompt}).
    Respecter strictement : Aucun texte hors code, Aucune balise Markdown, Aucune donnée inventée.
    Format attendu (concret) : un script Python complet qui :
    Parse / lit la variable texte {db.get_table_info()} (fourni par l'environnement) pour décider des noms de table/colonnes à sélectionner.
    Construit query en conséquence.
    Exécute df = pd.read_sql(query, db._engine).
    Produit la figure matplotlib décrite.
    Remplace les placeholders suivants avant exécution :
    {besoins} → instructions fonctionnelles SQL (si ton système les fournit).
    {prompt} → description textuelle de la carte attendue (couleurs, filtres, titre).
    {db.get_table_info()} → description textuelle du schéma réel (table(s) et colonnes)
    """
    answer=client.responses.create(
        model="gpt-4o-mini", 
        input=map_prompt
    )
    code = answer.output_text
    #print(code) 
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

    if route == "image":
       url = generate_graph_from_prompt(user_query,db)
       return url

    if route == "chat":
        llm = ChatOpenAI(model="gpt-4o-mini")
        return llm.invoke(user_query).content
    sql_chain = get_sql_chain(db)

    template = """
    Tu es un data analyst travaillant pour une entreprise.  
    Tu échanges avec un utilisateur qui te pose des questions sur la base de données spatial (postgis) de l'entreprise.

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


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Bonjour! Je suis un assistant SQL. Demande moi ce que tu veux sur ta base de données")
    ]
if "schema_display" not in st.session_state:
    st.session_state.schema_display = None



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
                st.image(image, caption=genere_titre(image,db))
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
        if response.startswith("data:image/png;base64,"):
            st.image(response, caption=genere_titre(response,db))
        else :
            st.markdown(response)
    st.session_state.chat_history.append(AIMessage(content=response))

