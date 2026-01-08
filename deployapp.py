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
import folium
from streamlit_folium import st_folium
import json

import streamlit as st
import os
import matplotlib.pyplot as plt 
import io
import base64
from sqlalchemy import inspect
from sqlalchemy.exc import NoInspectionAvailable
try:
    from geoalchemy2.types import Geometry
    GEOALCHEMY_AVAILABLE = True
except Exception:
    Geometry = None
    GEOALCHEMY_AVAILABLE = False

load_dotenv()

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

def _choose_tooltip_fields(geojson, prefer=None, max_fields=2):
    """
    Retourne une liste de champs utilisables pour GeoJsonTooltip.
    Gestion sûre si geojson['features'] est None ou vide.
    """
    if not geojson:
        return []
    features = geojson.get("features") or []
    if not isinstance(features, list) or len(features) == 0:
        return []
    props = features[0].get("properties", {}) or {}
    if not props:
        return []
    prefer = prefer or ["name", "label", "type_episode", "commune", "pk_residential_episode", "id"]
    for p in prefer:
        if p in props:
            return [p]
    keys = [k for k in props.keys() if k.lower() not in ("geom", "geometry")]
    return keys[:max_fields]

def schema_with_geo_via_geoalchemy(db, engine=None, schema: str = "public") -> str:
    """
    Version minimale et concise : récupère le schéma via SQLAlchemy.inspect si possible,
    puis ajoute les entrées de geometry_columns en gérant simplement le cas où
    db.run(...) renvoie une chaîne (string) représentant la liste.
    On ne multiplie pas les vérifications — on traite juste les cas courants.
    """
    import ast

    engine = engine or getattr(db, "_engine", None)
    out_lines = []

    # 1) Table/colonnes via inspect si possible, sinon fallback sur db.get_table_info()
    try:
        if engine is not None:
            insp = inspect(engine)
            try:
                tables = insp.get_table_names(schema=schema)
            except Exception:
                tables = insp.get_table_names()
            for table in tables:
                out_lines.append(f"Table {table}:")
                try:
                    cols = insp.get_columns(table, schema=schema)
                except Exception:
                    cols = insp.get_columns(table)
                for c in cols:
                    out_lines.append(f"  - {c.get('name')}: {c.get('type')}")
                out_lines.append("")
        else:
            base = db.get_table_info() or ""
            if base:
                out_lines.append(base)
    except Exception:
        # si inspect plante, on retombe sur db.get_table_info() (minimal)
        try:
            base = db.get_table_info() or ""
            if base:
                out_lines.append(base)
        except Exception:
            pass

    # 2) geometry_columns : cas simple (string ou list/tuple)
    try:
        geom_meta_raw = db.run(
            """
            SELECT f_table_schema AS schema,
                   f_table_name  AS table,
                   f_geometry_column AS column,
                   type,
                   srid
            FROM public.geometry_columns;
            """
        )

        # Normaliser en liste d'enregistrements
        if isinstance(geom_meta_raw, str):
            # si c'est une string représentant une structure Python, essayer de la parser
            try:
                parsed = ast.literal_eval(geom_meta_raw)
                if isinstance(parsed, (list, tuple)):
                    geom_rows = list(parsed)
                else:
                    geom_rows = [parsed]
            except Exception:
                geom_rows = [geom_meta_raw]
        elif isinstance(geom_meta_raw, (list, tuple)):
            geom_rows = list(geom_meta_raw)
        else:
            geom_rows = [geom_meta_raw]

        if geom_rows:
            out_lines.append("Geometry columns (information from geometry_columns):")
            for r in geom_rows:
                # cas dict (idéal)
                if isinstance(r, dict):
                    table = str(r.get("table", "")).strip()
                    col = str(r.get("column", "")).strip()
                    typ = str(r.get("type", r.get("udt", ""))).strip()
                    srid = r.get("srid", None)
                    if srid is not None:
                        out_lines.append(f"  - {table}.{col}: {typ} SRID={srid}")
                    else:
                        out_lines.append(f"  - {table}.{col}: {typ}")
                    continue

                # cas tuple/list attendu: ('schema','table','column','TYPE',srid)
                if isinstance(r, (list, tuple)) and len(r) >= 3:
                    parts = [str(x).strip().strip("'\"") for x in r]
                    # si on a au moins schema, table, column
                    if len(parts) >= 5:
                        out_lines.append(f"  - {parts[1]}.{parts[2]}: {parts[3]} SRID={parts[4]}")
                    elif len(parts) >= 3:
                        out_lines.append(f"  - {parts[0]}.{parts[1]}: {parts[2]}")
                    else:
                        out_lines.append("  - " + " ".join(parts))
                    continue

                # fallback simple : afficher sur une seule ligne
                s = str(r)
                s = " ".join(s.split())
                s = s.strip(" '\"")
                out_lines.append("  - " + s)

    except Exception:
        # si la requête geometry_columns échoue, on ignore silencieusement (fonction minimale)
        pass

    return "\n".join(out_lines).strip()


def get_sql_chain(db):
    try:
        engine = getattr(db, "_engine", None)
        schema_text = schema_with_geo_via_geoalchemy(db, engine=engine, schema="public")
    except Exception as e:
        # fallback simple en cas d'erreur
        schema_text = db.get_table_info() or f"(échec récupération schéma: {e})"
    template = """
    Tu es un data analyst travaillant pour une entreprise.
    Tu échanges avec un utilisateur qui te pose des questions sur la base de données spatial (postgis) de l'entreprise.

    À partir du schéma des tables ci-dessous, écris une requête SQL qui permettrait de répondre à la question de l'utilisateur.
    Tiens également compte de l'historique de la conversation pour formuler ta réponse.

    Si la question concerne la temporalité, la date actuelle est : {current_date}.

    La base de données est une base de données sur les trajectoires de vies, celle-ci est décomposé en épisode de vie.
    Il y a 4 types d'épisode : familial, professionnel, loisir et résidentiel.
    Un épisode a donc un début et une fin, et les épisodes s'enchainent. 
    C'est a dire que la fin d'un épisode est aussi le début du suivant. 
    ex : 3 épisodes qui se suivent ont en commun la date de fin du premier est la date de début du second et la date de fin du deuxième est la date de début du troisième etc... 

    ⚠️ IMPORTANT — RÈGLES POUR SUPABASE :
    - N'utilise JAMAIS ST_DistanceSphere().
    - Pour calculer des distances réelles en mètres, utilise : 
    ST_Distance(geom::geography, geom::geography)
    - Pour calculer un rayon autour d'un point, utilise aussi ST_Distance(...::geography).
    - Toujours caster les géométries en ::geography avant ST_Distance.
    - Toujours renvoyer une REQUÊTE SQL VALIDE SUPABASE.
    
    IMPORTANT — CAST SÛR DES CHAMPS DE TEXTE A TRANSFORMER EN INTEGER :

    Si vous devez convertir une colonne texte en entier, n'utilisez jamais directement CAST(col AS INTEGER).
    Utilisez systématiquement cette forme sûre qui supprime les caractères non numériques et gère les valeurs non convertibles : CAST(NULLIF(regexp_replace(col, '\D', '', 'g'), '') AS INTEGER)
    Exemple : remplacez CAST(date_fin AS INTEGER) par CAST(NULLIF(regexp_replace(date_fin, '\D', '', 'g'), '') AS INTEGER)
    Ou utilisez la forme équivalente avec ::int : NULLIF(regexp_replace(date_fin, '\D', '', 'g'), '')::int
    Si vous devez vérifier explicitement la validité, vous pouvez utiliser : CASE WHEN regexp_replace(col,'\D','','g') ~ '^\d+$' THEN regexp_replace(col,'\D','','g')::int ELSE NULL END Ne fournissez que la requête SQL (pas d'explication) et appliquez toujours ce pattern pour les conversions en entier. 

    — Exemple concret : transformation attendue Entrée générée par défaut (problème) : SELECT ..., (CAST(le.date_fin AS INTEGER) - CAST(le.date_debut AS INTEGER)) AS duree FROM leisure_episode le;
        Version sûre (ce que vous voulez) : SELECT ..., ( CAST(NULLIF(regexp_replace(le.date_fin, '\D', '', 'g'), '') AS INTEGER) - CAST(NULLIF(regexp_replace(le.date_debut, '\D', '', 'g'), '') AS INTEGER) ) AS duree FROM leisure_episode le;

    <SCHEMA>{schema}</SCHEMA>
    Utilise bien les noms des colonnes utilisés dans le schéma, et non celles des exemples.
    Historique de la conversation : {chat_history}

    Rédige uniquement la requête SQL — sans aucun texte explicatif, sans commentaire et sans backticks.

    Exemple :
    Question : Quels sont les lieux de résidence successifs de la personne 5067, classés dans l'ordre chronologique ?
    Requête SQL : 
    SELECT r.type_episode, l.commune, r.date_debut, r.date_fin
    FROM residential_episode r
    JOIN localisation l ON r.fk_ref_loc = l.pk_ref_loc
    WHERE r.fk_personne_id = 5067
    ORDER BY r.date_debut;


    Question : Quels épisodes professionnels ont eu lieu à moins de 50 km du lieu de naissance des personnes ? (limite de 20)
    Requête SQL : 
    SELECT 
    pe.pk_professionnal_episode,
    pe.type_episode,
    pe.date_debut,
    pe.date_fin,
    p.pk_personne_id,
    l_ep.commune AS lieu_episode,
    l_birth.commune AS lieu_naissance,
    ST_Distance(
        l_ep.geom::geography,
        l_birth.geom::geography
    ) / 1000 AS distance_km
    FROM professionnal_episode pe
    JOIN personne p
    ON pe.fk_personne_id = p.pk_personne_id
    JOIN localisation l_ep
    ON pe.fk_ref_loc = l_ep.pk_ref_loc
    JOIN localisation l_birth
    ON p.fk_ref_loc = l_birth.pk_ref_loc
    WHERE ST_Distance(
        l_ep.geom::geography,
        l_birth.geom::geography
    ) < 50000
    ORDER BY distance_km
    LIMIT 20;


    Question : Quelles personnes ont vécu une trajectoire résidentielle diversifiée (plus de 3 lieux différents) et ont connu au moins un événement familial et un événement professionnel dans des communes différentes ? (limite de 20)
    Requête SQL : 
    WITH residential_count AS (
        SELECT 
            fk_personne_id,
            COUNT(DISTINCT fk_ref_loc) AS nb_lieux
        FROM residential_episode
        GROUP BY fk_personne_id
    ),
    familial_places AS (
        SELECT 
            fk_personne_id,
            fk_ref_loc
        FROM familial_event
    ),
    professionnal_places AS (
        SELECT 
            fk_personne_id,
            fk_ref_loc
        FROM professionnal_event
    )
    SELECT DISTINCT
        rc.fk_personne_id,
        rc.nb_lieux
    FROM residential_count rc
    JOIN familial_places fe
        ON rc.fk_personne_id = fe.fk_personne_id
    JOIN professionnal_places pe
        ON rc.fk_personne_id = pe.fk_personne_id
    WHERE rc.nb_lieux > 3
        AND fe.fk_ref_loc <> pe.fk_ref_loc
    LIMIT 20;

    (alternative : Quelles personnes ont vécu une trajectoire résidentielle longue (plus de 3 lieux distincts) et ont connu au moins un épisode familial et un épisode professionnel, dont les événements associés ont eu lieu dans des communes différentes ?
    Requête SQL :
    WITH residential_count AS (
        SELECT 
            fk_personne_id,
            COUNT(DISTINCT fk_ref_loc) AS nb_lieux
        FROM residential_episode
        WHERE fk_ref_loc IS NOT NULL
        GROUP BY fk_personne_id
    ),

    familial_places AS (
        SELECT DISTINCT
            fe.fk_personne_id,
            fe.fk_ref_loc
        FROM familial_episode fep
        JOIN familial_event fe
        ON fe.fk_familial_episode = fep.pk_familial_episode
        WHERE fe.fk_ref_loc IS NOT NULL
    ),

    professionnal_places AS (
        SELECT DISTINCT
            pe.fk_personne_id,
            pe.fk_ref_loc
        FROM professionnal_episode pep
        JOIN professionnal_event pe
        ON pe.fk_professionnal_episode = pep.pk_professionnal_episode
        WHERE pe.fk_ref_loc IS NOT NULL
    )

    SELECT DISTINCT
        rc.fk_personne_id,
        rc.nb_lieux
    FROM residential_count rc
    JOIN familial_places fp
    ON rc.fk_personne_id = fp.fk_personne_id
    JOIN professionnal_places pp
    ON rc.fk_personne_id = pp.fk_personne_id
    WHERE rc.nb_lieux > 3
    AND fp.fk_ref_loc <> pp.fk_ref_loc
    ORDER BY rc.nb_lieux DESC
    LIMIT 20;)

    À ton tour :

    Question : {question}
    Requête SQL :
    """
    prompt = ChatPromptTemplate.from_template(template)

    #llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    llm= ChatOpenAI(model= "gpt-4o-mini")
    
    return (
        RunnablePassthrough.assign(schema=lambda _: schema_text)
        | prompt
        | llm
        | StrOutputParser()

    )
    
def get_rep(user_query: str, chat_history: list):
    prompt = """
    Tu es un spécialiste dans le sujet de la base de donnée qui est à ta disposition. Analyse la demande utilisateur et réponds UNIQUEMENT par :
    - "sql" si la question nécessite une requête SQL sur la base, c'est une base spatiale postgis.
    - "image" si l'utilisateur veut une image, schéma, visualisation, graphique
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
    IMPORTANT — CAST SÛR DES CHAMPS DE TEXTE A TRANSFORMER EN INTEGER :

    Si vous devez convertir une colonne texte en entier, n'utilisez jamais directement CAST(col AS INTEGER).
    Utilisez systématiquement cette forme sûre qui supprime les caractères non numériques et gère les valeurs non convertibles : CAST(NULLIF(regexp_replace(col, '\D', '', 'g'), '') AS INTEGER)
    Exemple : remplacez CAST(date_fin AS INTEGER) par CAST(NULLIF(regexp_replace(date_fin, '\D', '', 'g'), '') AS INTEGER)
    Ou utilisez la forme équivalente avec ::int : NULLIF(regexp_replace(date_fin, '\D', '', 'g'), '')::int
    Si vous devez vérifier explicitement la validité, vous pouvez utiliser : CASE WHEN regexp_replace(col,'\D','','g') ~ '^\d+$' THEN regexp_replace(col,'\D','','g')::int ELSE NULL END Ne fournissez que la requête SQL (pas d'explication) et appliquez toujours ce pattern pour les conversions en entier. 

    — Exemple concret : transformation attendue Entrée générée par défaut (problème) : SELECT ..., (CAST(le.date_fin AS INTEGER) - CAST(le.date_debut AS INTEGER)) AS duree FROM leisure_episode le;
        Version sûre (ce que vous voulez) : SELECT ..., ( CAST(NULLIF(regexp_replace(le.date_fin, '\D', '', 'g'), '') AS INTEGER) - CAST(NULLIF(regexp_replace(le.date_debut, '\D', '', 'g'), '') AS INTEGER) ) AS duree FROM leisure_episode le;
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
    {schema_with_geo_via_geoalchemy(db)}
    """
    answer=client.responses.create(
        model="gpt-4o-mini", 
        input=full_prompt
    )
    code = answer.output_text
    local_vars={}
    print(code)
    exec(code,{"plt":plt,"io":io, "db":db}, local_vars)
    buf=io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    img_base64_str = "data:image/png;base64," + img_base64
    return img_base64_str

def genere_titre(prompt,db,chat_history): #c'est bon c'est validé
    besoins =get_sql_chain(db)
    pprompt = f"""
    T'es un spécialiste dans le sujet de la base de données qu'on t'a fournis 
    et t'as besoins d'écrire un titre simple et concis pour un graphique ou une carte basé sur le contenue de la demande suivante :
    {prompt}
    (Tu peux t'aider de l'historique de la conversation pour le contexte: {chat_history})
    Le titre doit être court, clair et pertinent par rapport à la demande et doit refléter le contenu du graphique ou de la carte basé sur: 
    {besoins}
    """
    aanswer=client.responses.create(
        model="gpt-4o-mini", 
        input=pprompt
    )
    titre = aanswer.output_text  
    return titre



def get_response(user_query : str, db: SQLDatabase, chat_history: list):
    route = get_rep(user_query, chat_history)
    
    if route == "map":
        return {"type": "map", "query": user_query}
    
    if route == "image":
       url = generate_graph_from_prompt(user_query,db)
       return url

    sql_chain = get_sql_chain(db)
    try:
        engine = getattr(db, "_engine", None)
        schema_text = schema_with_geo_via_geoalchemy(db, engine=engine, schema="public")
    except Exception as e:
        # fallback simple en cas d'erreur
        schema_text = db.get_table_info() or f"(échec récupération schéma: {e})"

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
            schema=lambda _: schema_text,
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
    try:
        engine = getattr(db, "_engine", None)
        schema_text = schema_with_geo_via_geoalchemy(db, engine=engine, schema="public")
    except Exception as e:
        # fallback simple en cas d'erreur
        schema_text = db.get_table_info() or f"(échec récupération schéma: {e})"
    
    template = """
Tu es un data analyst travaillant pour une entreprise.
Génère une requête SQL qui renvoie des données au format GeoJSON en utilisant EXACTEMENT les colonnes du schéma fourni.

La base de données est une base de données sur les trajectoires de vies, celle-ci est décomposé en épisode de vie.
Il y a 4 types d'épisode : familial, professionnel, loisir et résidentiel.
Un épisode a donc un début et une fin, et les épisodes s'enchainent. 
C'est a dire que la fin d'un épisode est aussi le début du suivant. 
ex : 3 épisodes qui se suivent ont en commun la date de fin du premier est la date de début du second et la date de fin du deuxième est la date de début du troisième etc... 

RÈGLES CRITIQUES :
1. Rédige UNIQUEMENT la requête SQL, SANS backticks
2. La requête doit retourner UNE SEULE colonne nommée 'geojson'
3. Utilise json_build_object et json_agg pour construire le GeoJSON
4. IMPORTANT : Utilise UNIQUEMENT les colonnes qui existent dans le schéma ci-dessous
5. N'invente JAMAIS de colonnes (comme 'name' ou 'population') qui ne sont pas dans le schéma
6. Pour la géométrie, utilise la colonne qui contient 'geom' ou similaire
7. Pour les properties, inclus TOUTES les colonnes non-géométriques de la table

IMPORTANT — CAST SÛR DES CHAMPS DE TEXTE A TRANSFORMER EN INTEGER :

    Si vous devez convertir une colonne texte en entier, n'utilisez jamais directement CAST(col AS INTEGER).
    Utilisez systématiquement cette forme sûre qui supprime les caractères non numériques et gère les valeurs non convertibles : CAST(NULLIF(regexp_replace(col, '\D', '', 'g'), '') AS INTEGER)
    Exemple : remplacez CAST(date_fin AS INTEGER) par CAST(NULLIF(regexp_replace(date_fin, '\D', '', 'g'), '') AS INTEGER)
    Ou utilisez la forme équivalente avec ::int : NULLIF(regexp_replace(date_fin, '\D', '', 'g'), '')::int
    Si vous devez vérifier explicitement la validité, vous pouvez utiliser : CASE WHEN regexp_replace(col,'\D','','g') ~ '^\d+$' THEN regexp_replace(col,'\D','','g')::int ELSE NULL END Ne fournissez que la requête SQL (pas d'explication) et appliquez toujours ce pattern pour les conversions en entier. 

    — Exemple concret : transformation attendue Entrée générée par défaut (problème) : SELECT ..., (CAST(le.date_fin AS INTEGER) - CAST(le.date_debut AS INTEGER)) AS duree FROM leisure_episode le;
        Version sûre (ce que vous voulez) : SELECT ..., ( CAST(NULLIF(regexp_replace(le.date_fin, '\D', '', 'g'), '') AS INTEGER) - CAST(NULLIF(regexp_replace(le.date_debut, '\D', '', 'g'), '') AS INTEGER) ) AS duree FROM leisure_episode le;

Schéma de la base de données :
<SCHEMA>{schema}</SCHEMA>

Historique de la conversation : {chat_history}

ÉTAPES À SUIVRE :
1. Identifie la table concernée par la question
2. Repère la colonne de géométrie (généralement 'geom', 'geometry', 'location', etc.)
3. Identifie TOUTES les autres colonnes de cette table (ce seront les properties)
4. Si la requête SQL passée filtre certaines lignes, applique le même filtre

Requête SQL de base (pour filtrage si nécessaire) : {requete_sql}

TEMPLATE DE RÉPONSE (à adapter avec les VRAIES colonnes) :
SELECT json_build_object(
    'type', 'FeatureCollection',
    'features', json_agg(
        json_build_object(
            'type', 'Feature',
            'geometry', ST_AsGeoJSON([nom_colonne_géométrie])::json,
            'properties', json_build_object(
                '[colonne1]', [colonne1],
                '[colonne2]', [colonne2],
                '[colonne3]', [colonne3]
                -- Liste TOUTES les colonnes non-géométriques ici
            )
        )
    )
) AS geojson
FROM [nom_table];


À TON TOUR - Question de l'utilisateur : {question}

RAPPEL FINAL : Utilise UNIQUEMENT les colonnes qui existent réellement dans le schéma fourni !
Requête SQL :
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    return (
        RunnablePassthrough.assign(schema=lambda _: schema_text, requete_sql=lambda _: get_sql_chain(db).invoke({"question": "{question}", "chat_history": "{chat_history}","current_date": now}))
        | prompt
        | llm
        | StrOutputParser()
    )


def display_schema(db: SQLDatabase):
    try:
        engine = getattr(db, "_engine", None)
        schema_text = schema_with_geo_via_geoalchemy(db, engine=engine, schema="public")
    except Exception as e:
        # fallback simple en cas d'erreur
        schema_text = db.get_table_info() or f"(échec récupération schéma: {e})"
    template = """
    Voici le schéma des tables de la base de données :
    <SCHEMA>{schema}</SCHEMA>
    Rédige une courte et concise présentation de cette base de données en français. Pas besoin d'exemples ou de détails techniques.
    Présente la de façon claire, structurée et ergonomique.
    Par exemple, noms des tables et colonnes avec une courte description en langage naturel.
    C'est une base de données spatiale PostGIS, prends bien en comptes les colonnes géographiques.
    """
    prompt= ChatPromptTemplate.from_template(template)
    llm= ChatOpenAI(model= "gpt-4o-mini")
    return (
        RunnablePassthrough.assign(schema=lambda _: schema_text)
        | prompt
        | llm
        | StrOutputParser()
    )
def clean_sql_query(query: str) -> str:
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
            st.markdown(schema_with_geo_via_geoalchemy(st.session_state.db))
            print(schema_with_geo_via_geoalchemy(st.session_state.db))
            st.success("Connecté à la base de données!")
            st.session_state.schema_display = display_schema(st.session_state.db).invoke({})
    if st.session_state.schema_display:
        st.markdown(st.session_state.schema_display)


for message in st.session_state.chat_history:
    # Normaliser le contenu : si c'est [dict], prendre le dict à l'indice 0
    raw = message.content
    content = raw
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
        content = raw[0]
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            if isinstance(content,str)and content.startswith("data:image/png;base64,"):
                image_data = content.split(",")[1]
                image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                st.image(image, caption="")
            # Si c'est une carte stockée
            # Si c'est une carte stockée (content peut être dict ou [dict])
            elif (isinstance(content, dict) and content.get("type") == "stored_map") or (
                isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict) and content[0].get("type") == "stored_map"
            ):
                content = content[0] if isinstance(content, list) else content
                map_id = content.get("map_id")
                if map_id in st.session_state.maps_data:
                    map_data = st.session_state.maps_data[map_id]
                    
                    # Recréer la carte avec les données stockées
                    m = folium.Map(location=[46.603354, 1.888334], zoom_start=6)
                    tooltip_fields = _choose_tooltip_fields(map_data["geojson"])
                    if tooltip_fields:
                        tooltip = folium.GeoJsonTooltip(fields=tooltip_fields, aliases=[f"{f}:" for f in tooltip_fields], localize=True)
                        gj = folium.GeoJson(map_data["geojson"], name="geojson")
                        gj.add_to(m)
                        gj.add_child(tooltip)
                    else:
                        folium.GeoJson(map_data["geojson"], name="geojson").add_to(m)
                    
                    st_folium(m, width=700, height=500, key=f"map_{map_id}")
                    st.markdown(f"### {map_data['titre']}")
                else:
                    st.markdown(f"### {content.get('titre', 'Carte')}")
                    st.info("(Carte non disponible)")
            else:
                st.markdown(content)
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
                titre = genere_titre(user_query, st.session_state.db, st.session_state.chat_history)
                
                # Générer et nettoyer la requête SQL
                geojson_chain = get_geojson_chain(st.session_state.db)
                json_sql_query = geojson_chain.invoke({"question": user_query, "chat_history": st.session_state.chat_history})
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
                tooltip_fields = _choose_tooltip_fields(geojson_data)
                if tooltip_fields:
                    tooltip = folium.GeoJsonTooltip(fields=tooltip_fields, aliases=[f"{f}:" for f in tooltip_fields], localize=True)
                    gj = folium.GeoJson(geojson_data, name="geojson")
                    gj.add_to(m)
                    gj.add_child(tooltip)
                else:
                    folium.GeoJson(geojson_data, name="geojson").add_to(m)
                
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

