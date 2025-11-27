from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langsmith import traceable,Client
from langchain_groq import ChatGroq
import streamlit as st
import os



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


client = Client()

def init_database()-> SQLDatabase:
    #db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    db_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

@traceable(name="SQL Query Generator")
def get_sql_chain(db):
    template = """
Tu es un data analyst travaillant pour une entreprise.
Tu échanges avec un utilisateur qui te pose des questions sur la base de données spatial (postgis) de l'entreprise.

À partir du schéma des tables ci-dessous, écris une requête SQL qui permettrait de répondre à la question de l'utilisateur.
Tiens également compte de l'historique de la conversation pour formuler ta réponse.

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
    
@traceable(name="SQL Response Generator")
def get_response(user_query : str, db: SQLDatabase, chat_history: list):
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
        AIMessage(content="Bonjour! Je suis un assistant SQL. Demande moi ce que tu veux sur ta base de donnée")
    ]



st.set_page_config(page_title="Discute avec ta base de donnée", page_icon=":speech_balloon:")
st.title("Discute avec ta base de donnée")

with st.sidebar:
    st.subheader("Paramètres")
    st.write("C'est une simple application de discussion utilisant SQL. Connectez vous à la base de donnée pour commencer la discussion")
    if st.button("Connection"):
        with st.spinner("Connection à la base de donnée"):
            db = init_database()
            st.session_state.db=db
            st.success("Connecté à la base de donnée!")
            st.markdown(display_schema(st.session_state.db).invoke({}))
    


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Humain"):
            st.markdown(message.content)

user_query = st.chat_input("Ecrivez un message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        response = get_response(user_query,st.session_state.db, st.session_state.chat_history)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))

