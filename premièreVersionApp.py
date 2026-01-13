from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import streamlit as st
import os
load_dotenv()
print("✅ Variables chargées:")
print("Host:", os.getenv("MYSQL_HOST"))
print("User:", os.getenv("MYSQL_USER"))
print("Groq API Key:", os.getenv("GROQ_API_KEY")[:10] if os.getenv("GROQ_API_KEY") else "None")
host = os.getenv("MYSQL_HOST")
port = os.getenv("MYSQL_PORT")
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASSWORD")
database = os.getenv("MYSQL_DATABASE")

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
  db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
  print("🧠 DEBUG — db_uri:", db_uri)
  return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
  template = """
    Tu es un data analyst travaillant pour une entreprise.
Tu échanges avec un utilisateur qui te pose des questions sur la base de données de l'entreprise.

À partir du schéma des tables ci-dessous, écris une requête SQL qui permettrait de répondre à la question de l'utilisateur.
Tiens également compte de l'historique de la conversation pour formuler ta réponse.

<SCHEMA>{schema}</SCHEMA>

Historique de la conversation : {chat_history}

Rédige uniquement la requête SQL — sans aucun texte explicatif, sans commentaire et sans backticks.

Exemple :
Question : Quels sont les 5 clients qui ont payé le plus d'argent au total ?
Requête SQL : 
SELECT 
    c.customerName, 
    c.country,
    SUM(p.amount) as total_paid
FROM customers c
JOIN payments p ON c.customerNumber = p.customerNumber
GROUP BY c.customerNumber, c.customerName, c.country
ORDER BY total_paid DESC
LIMIT 5;

Question : Montre-moi tous les bureaux avec leur ville et leur pays.
Requête SQL : 
SELECT city, country
FROM offices 
ORDER BY country;

À ton tour :

Question : {question}
Requête SQL :
    """
    
  prompt = ChatPromptTemplate.from_template(template)
  
  # llm = ChatOpenAI(model="gpt-4o-mini")
  llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )
    
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  sql_chain = get_sql_chain(db)
  
  template = """
    Tu es un data analyst travaillant pour une entreprise.  
Tu échanges avec un utilisateur qui te pose des questions sur la base de données de l'entreprise.

En te basant sur :
- le schéma des tables ci-dessous,  
- la question de l'utilisateur,  
- la requête SQL générée,  
- et le résultat de cette requête,  

rédige une **réponse claire et naturelle** en français, adaptée à l'utilisateur.

<SCHEMA>{schema}</SCHEMA>

Historique de la conversation : {chat_history}  
Requête SQL : <SQL>{query}</SQL>  
Question de l'utilisateur : {question}  
Résultat SQL : {response}"""
  
  prompt = ChatPromptTemplate.from_template(template)
  
  # llm = ChatOpenAI(model="gpt-4o-mini")
  llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
  
  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | llm
    | StrOutputParser()
  )
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })
    
  
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [ 
      AIMessage(content="Bonjour! Je suis un assistant SQL. Demande moi ce que tu veux sur ta base de donnée."),
    ]



st.set_page_config(page_title="Discute avec MySQL", page_icon=":speech_balloon:")

st.title("Discute avec MySQL")

with st.sidebar:
    st.subheader("Paramètres") 
    st.write("Ceci est une simple application de discussion qui utilise MySQL. Connecte toi à ta base de donnée et commence la discussion.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Mot de passe", type="password", key="Password")
    st.text_input("Base de donnée", value="classicmodels" ,key="Database")
    if st.button("Connection"):
        with st.spinner("Connection à la base de donnée..."):
            db = init_database(
            st.session_state["User"],
            st.session_state["Password"],
            st.session_state["Host"],
            st.session_state["Port"],
            st.session_state["Database"]
        )
            st.session_state.db = db
            st.success("Connecté à la base de donnée!")
    
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Ecris un message..")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))