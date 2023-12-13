from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors



app = FastAPI()

@app.get('/')
def bienvenida():
    return {'API de consultas a una base de datos de Steam, /docs en el link para acceder a las funciones de consulta.'}


@app.get('/PlayTimeGenre/ {genero}')

def PlayTimeGenre( genero : str ):
    try:
        consulta_final = pd.read_csv('./Datasets/merged/playtime_genre.csv.gz')
        # Filtrar el DataFrame por el género especificado
        genero_df = consulta_final[consulta_final['genres'] == genero]

        # Obtener el año con más horas jugadas para el género dado
        max_year = genero_df['max_year'].iloc[0]

        # Retornar el resultado en un formato deseado
        #return {"Año de lanzamiento con más horas jugadas para Género {}:".format(genero): max_year}
        return "Año de lanzamiento con más horas jugadas para Género {}: {}".format(genero, int(max_year))
    
    except Exception as e:
        return {"Error":str(e)}


@app.get('/UserForGenre/ {genero}')

def UserForGenre(genre):
    try:
            consulta_final = pd.read_csv('./Datasets/merged/user_genre.csv.gz', index_col=['index'])
            user_max = consulta_final.loc[genre].nombre
            diccionario = ast.literal_eval(consulta_final.loc[genre].year)
            diccionario['Horas_Jugadas'] = diccionario.pop('playtime_forever')
            return f"Usuario con más horas jugadas para Género {genre}: {user_max},\n Horas jugadas: {[str(diccionario)]}"

        
    except Exception as e:
        return {"Error":str(e)}



@app.get('/UsersRecommend/ {año}')

def UsersRecommend(year: int):
    try:
        function3 = pd.read_csv('./Datasets/merged/function3.csv.gz', compression='gzip')
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
        function3 = pd.read_csv('./Datasets/merged/function3.csv.gz', compression='gzip')
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
        function3 = pd.read_csv('./Datasets/merged/function3.csv.gz', compression='gzip')
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
def recomendacion_juego(item_id :int):
    try:
        consulta_06 = pd.read_csv('./Datasets/merged/recomendacion_juego.csv.gz',compression='gzip')
        
        nombre_juego = consulta_06.set_index('item_id').loc[item_id].values[0].split(',')[0]

        #Eliminaremos las stopwords

        stop_words_steams = ['aaaaaa', 'ab', 'abbey','abe', 'abramenko']
        stop = list(stopwords.words('english'))
        stop += stop_words_steams


        tf = TfidfVectorizer(stop_words=stop, token_pattern=r'\b[a-zA-Z]\w+\b' )

        data_vector = tf.fit_transform(consulta_06['features'])

        data_vector_df = pd.DataFrame(data_vector.toarray(), index=consulta_06['item_id'], columns = tf.get_feature_names_out())
            
        vector_similitud_coseno = cosine_similarity(data_vector_df.values)
        
        cos_sim_df = pd.DataFrame(vector_similitud_coseno, index=data_vector_df.index, columns=data_vector_df.index)
        
        ##top5
        juegos_similares = cos_sim_df.loc[item_id].nlargest(6)

        top5 = juegos_similares.iloc[1:6]
        
        
        resultado = consulta_06.set_index('item_id').loc[top5.index]['features'].apply(lambda x: x.split(',')[0]).values
        print(f"Los juegos similares a {nombre_juego} son :\n")
        for name in resultado:
            print("\n",name)
            
        resultado = consulta_06.set_index('item_id').loc[top5.index]['features'].apply(lambda x: x.split(',')[0]).values
            
        return list(resultado)
  
    except Exception as e:
        return {"Error": str(e)}
    
    

@app.get('/recomendacion_usuario/ {user_id}')
def recomendacion_usuario(usuario_id, steam):
    try:
        steam = pd.read_csv (r'../Datasets/merged/function3.csv.gz', compression='gzip', usecols=['user_id','item_id','sentiment_analysis','recommend','app_name'])
        # Filtrar el DataFrame para obtener las entradas del usuario específico
        usuario_df = steam[steam['user_id'] == usuario_id]

        if usuario_df.empty:
            return f"No se encontraron datos para el usuario con ID {usuario_id}."

        # Crear una matriz de usuario-juego para el modelo de vecinos más cercanos
        matriz_usuario_juego = pd.pivot_table(steam, values='sentiment_analysis', index='user_id', columns='app_name', fill_value=0)

        # Transponer la matriz para trabajar en el espacio de juegos
        matriz_juego_usuario = matriz_usuario_juego.T

        # Inicializar el modelo de vecinos más cercanos
        modelo_vecinos = NearestNeighbors(metric='cosine', algorithm='brute')
        modelo_vecinos.fit(matriz_juego_usuario)

        # Encontrar vecinos más cercanos al juego específico
        juego_interes = usuario_df['app_name'].iloc[0]
        indice_juego = matriz_juego_usuario.index.get_loc(juego_interes)
        distancias, indices = modelo_vecinos.kneighbors(matriz_juego_usuario.iloc[indice_juego, :].values.reshape(1, -1), n_neighbors=6)

        # Construir la lista de juegos recomendados
        juegos_recomendados = []
        for i, indice_juego_recomendado in enumerate(indices.flatten()[1:]):  # Excluir al propio juego
            juego_recomendado = matriz_juego_usuario.index[indice_juego_recomendado]
            juegos_recomendados.append(juego_recomendado)

        return juegos_recomendados

    except Exception as e:
        return {"Error": str(e)}