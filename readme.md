# Application de chat PostGIS (local `Localapp.py` et déploiement `deployapp.py`)

Cette documentation détaille l’usage local et le déploiement Streamlit Cloud d’une application Streamlit qui interroge une base PostgreSQL/PostGIS en français. L’app génère des requêtes SQL (GPT-4o-mini), des graphiques (Matplotlib) et des cartes (Folium/GeoJSON) en tenant compte de l’historique de conversation. Vous pouvez tester directement l’application déployée ici : https://chat-with-postgis-udd2bspgtlyphkkjoqwzlb.streamlit.app/

## Sommaire
1. Fonctionnalités clés
2. Architecture et flux
3. Prérequis communs
4. Configuration locale (Localapp.py)
5. Déploiement Streamlit Cloud (deployapp.py)
6. Détails techniques importants
7. Dépannage rapide
8. Commandes utiles
9. Licence

## 1) Fonctionnalités clés
- Requêtes SQL en français via GPT-4o-mini (contexte de conversation pris en compte).
- Analyse spatiale PostGIS : détection des colonnes géométriques, export GeoJSON, cartes Folium avec tooltips.
- Visualisations automatiques : code Matplotlib généré côté serveur, rendu en image base64.
- Schéma enrichi : inspection SQLAlchemy + lecture de `geometry_columns` (type, SRID), fallback `db.get_table_info()`.
- Router intelligent : classification des demandes en `sql`, `image`, ou `map`.

## 2) Architecture et flux
```
Question utilisateur
  → get_rep() (router : sql / image / map)
    ├─ sql   → get_sql_chain() → db.run → get_response() → réponse textuelle
    ├─ image → generate_graph_from_prompt() → Matplotlib → base64 → affichage
    └─ map   → get_geojson_chain() → db.run → GeoJSON → Folium → carte interactive
```
Principales fonctions (dans `deployapp.py`) :
- `init_database()` : connexion PostgreSQL.
- `schema_with_geo_via_geoalchemy()` : schéma enrichi (tables/colonnes + geometry_columns).
- `get_sql_chain()` : génération de requêtes SQL par LLM.
- `get_response()` : orchestre schéma + requête SQL + exécution + réponse LLM.
- `get_geojson_chain()` : requêtes GeoJSON prêtes pour Folium.
- `generate_graph_from_prompt()` : code Matplotlib généré et exécuté.
- `genere_titre()` : titres contextuels (graphique ou carte) basés sur la demande et l’historique.

## 3) Prérequis communs
- Python 3.8+
- PostgreSQL 12+ avec extension PostGIS (`CREATE EXTENSION postgis;`)
- Accès API OpenAI (GPT-4o-mini)
- (Optionnel) LangSmith pour le tracing
- Paquets (voir `requirements.txt`) : Streamlit, LangChain (core/community/openai/groq), psycopg2-binary, geoalchemy2, folium, streamlit-folium, matplotlib, Pillow, python-dotenv, openai, sqlalchemy.

## 4) Configuration locale (Localapp.py)
1. Créez un `.env` à la racine :
```
OPENAI_API_KEY=xxxxxxxx
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=motdepasse
POSTGRES_DATABASE=ma_base
# Optionnel LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=cle_langsmith
LANGCHAIN_PROJECT=mon_projet
```
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```
3. Lancez en local :
```bash
streamlit run Localapp.py
```
4. Flux local :
   - `.env` chargé (python-dotenv) → variables pour DB et OpenAI.
   - Sidebar : bouton **Connection** → instancie la DB, récupère le schéma.
   - Saisie utilisateur → router (sql/image/map) → rendu (texte, image, carte).

## 5) Déploiement Streamlit Cloud (deployapp.py)
1. Créez `.streamlit/secrets.toml` (non commité) :
```toml
# LangChain
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_API_KEY = "votre_cle_langsmith"
LANGCHAIN_PROJECT = "nom_du_projet"

# OpenAI
OPENAI_API_KEY = "votre_cle_openai"

# PostgreSQL
[postgres]
host = "votre_host"
port = "5432"
user = "votre_utilisateur"
password = "votre_mot_de_passe"
database = "nom_base_de_donnees"
```
2. Sur Streamlit Cloud, définissez `deployapp.py` comme entrypoint.
3. Après déploiement : ouvrez l’URL, cliquez sur **Connection** pour initialiser la DB et récupérer le schéma, puis posez vos questions.

## 6) Détails techniques importants
- Distances / PostGIS : cast `::geography` + `ST_Distance` imposés dans les prompts (évite `ST_DistanceSphere`).
- Conversion sûre texte → entier :
  ```sql
  CAST(NULLIF(regexp_replace(colonne, '\\D', '', 'g'), '') AS INTEGER)
  ```
  pour éviter les échecs sur des champs texte non numériques.
- Schéma enrichi : inspection SQLAlchemy + table `geometry_columns`, fallback `db.get_table_info()` si besoin.
- Debug : expanders Streamlit affichent la requête SQL générée et le résultat brut (diagnostic GeoJSON/SQL).
- Session : cartes et graphiques stockés dans `st.session_state` pour réaffichage sans régénération.

## 7) Dépannage rapide
- Connexion DB : vérifier host/port/user/password/database dans `.env` ou `secrets.toml`; tester `psql`.
- PostGIS : `SELECT PostGIS_Version();` et `SELECT * FROM geometry_columns;`.
- OpenAI : clé valide, quotas disponibles, accès au modèle `gpt-4o-mini`.
- GeoJSON vide ou invalide : vérifier la colonne géométrique, le SRID (souvent 4326), consulter l’expander debug.

## 8) Commandes utiles
- Local : `streamlit run Localapp.py`
- Cloud : URL Streamlit Cloud (entrypoint `deployapp.py`)

## 9) Licence

Ce projet a été développé dans le cadre d'un stage académique à des fins éducatives et de démonstration. 


---
Documentation mise à jour : Janvier 2026
