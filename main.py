from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import ast


app = FastAPI()

@app.get('/')
def bienvenida():
    return {'API de consultas a una base de datos de Steam, /docs en el link para acceder a las funciones de consulta.'}


@app.get('/PlayTimeGenre/ {genero}')

def PlayTimeGenre(genero: str):
    try:
        
        function1 = pd.read_csv('./Datasets/function1.csv.gz', compression='gzip')
        genre_df = function1[function1['genres'].apply(lambda x: genero in x)]
        grouped_df = genre_df.groupby('year')['playtime_forever'].sum()
        max_playtime_year = grouped_df.idxmax()
        result = {"Año de lanzamiento con más horas jugadas para Género X": str(max_playtime_year)}
        return result
    
    except Exception as e:
        return {"Error":str(e)}


@app.get('/UserForGenre/ {genero}')

def UserForGenre(genre):
    try:
            consulta_final = pd.read_csv('02_nombres_max.csv.gz', index_col=['index'])
            user_max = consulta_final.loc[genre].nombre
            diccionario = ast.literal_eval(consulta_final.loc[genre].year)
            diccionario['Horas_Jugadas'] = diccionario.pop('playtime_forever')

            # Construir la cadena con saltos de línea para las horas jugadas
            horas_jugadas_str = "\n".join(f"{key}: {value}" for key, value in diccionario.items())

            result = f"Usuario con más horas jugadas para Género {genre}: {user_max},\n Horas jugadas:\n{horas_jugadas_str}"
            return result

        
    except Exception as e:
        return {"Error":str(e)}



@app.get('/UsersRecommend/ {año}')

def UsersRecommend(year: int):
    try:
        function3 = pd.read_csv('./Datasets/function3.csv')
        filtered_df = function3[(function3['recommend'] == 1) & (function3['sentiment_analysis'].isin([1, 2])) & (function3['year'] == year)]
        game_recommendations = filtered_df.groupby('app_name')['recommend'].count().sort_values(ascending=False)
        top_3_games = game_recommendations.head(3).to_dict()
        result = {"year": year, "top_3_games": [{"puesto {}".format(i + 1): {game}} for i, (game, _) in enumerate(top_3_games.items())]}
        return result

    except Exception as e:
        return {"Error": str(e)}



@app.get('/UsersNotRecommend/ {año}')

def UsersNotRecommend(year: int):
    try:
        function3 = pd.read_csv('./Datasets/function3.csv')
        filtered_df = function3[(function3['recommend'] == 0) & (function3['sentiment_analysis'].isin([0, -1])) & (function3['year'] == year)]
        game_not_recommendations = filtered_df.groupby('app_name')['recommend'].count().sort_values(ascending=False)
        top_3_not_recommended = game_not_recommendations.head(3).to_dict()
        result = {"year": year, "top_3_not_recommended": [{"puesto {}".format(i + 1): {game}} for i, (game, _) in enumerate(top_3_not_recommended.items())]}
        return result

    except Exception as e:
        return {"Error": str(e)}
    
    
    
@app.get('/sentiment_analysis/ {año}')
def sentiment_analysis(year: int):
    try:
        function3 = pd.read_csv('./Datasets/function3.csv.gz', compression='gzip')
        filtered_df = function3[function3['year'] == year]
        sentiment_counts = filtered_df.groupby('sentiment_analysis').size().reset_index(name='count')
        result = {
            'Negative': int(sentiment_counts.loc[sentiment_counts['sentiment_analysis'] == 0, 'count'].values[0]) if 0 in sentiment_counts['sentiment_analysis'].values else 0,
            'Neutral': int(sentiment_counts.loc[sentiment_counts['sentiment_analysis'] == 1, 'count'].values[0]) if 1 in sentiment_counts['sentiment_analysis'].values else 0,
            'Positive': int(sentiment_counts.loc[sentiment_counts['sentiment_analysis'] == 2, 'count'].values[0]) if 2 in sentiment_counts['sentiment_analysis'].values else 0
        }

        return result

    except Exception as e:
        return {"Error": str(e)}


@app.get('/recomendacion_juego/ {app_name}')
def recomendacion_juego(nombre_juego: str, num_recomendaciones=5):
    try:
        # Cargar el conjunto de datos
        df = pd.read_csv('./Datasets/function3.csv.gz', compression='gzip').sample(frac=0.7, random_state=42)
        df['review'].fillna('', inplace=True)

        # Representación vectorial con TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['review'])

        # Asegurarse de que la matriz contenga solo valores numéricos
        tfidf_matrix = tfidf_matrix.astype(float)

        # Calcular la similitud de coseno
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Crear un DataFrame de similitud
        cosine_sim_df = pd.DataFrame(cosine_sim, index=df.index, columns=df.index)

        # Obtener el ID del juego dado el nombre
        juego_id = df[df['app_name'] == nombre_juego].index[0]

        # Obtener la fila correspondiente al juego dado
        juego_fila = cosine_sim_df.loc[juego_id]

        # Ordenar los juegos por similitud (en orden descendente)
        juegos_similares = juego_fila.sort_values(ascending=False)

        # Excluir el juego dado de la lista de recomendaciones
        juegos_recomendados = juegos_similares.drop(juego_id)

        # Tomar los primeros 'num_recomendaciones' juegos
        juegos_recomendados = juegos_recomendados.head(num_recomendaciones)

        # Obtener los nombres de los juegos recomendados como lista
        nombres_recomendados = df.loc[juegos_recomendados.index, 'app_name'].tolist()

        # Crear un diccionario de recomendaciones
        recomendaciones = {}
        for i, juego_id in enumerate(juegos_recomendados.index, start=1):
            juego_nombre = df.loc[juego_id, 'app_name']
            recomendacion = {f"{i}": juego_nombre}
            recomendaciones.append(recomendacion)

        return recomendaciones
                
    except Exception as e:
        return {"Error": str(e)}