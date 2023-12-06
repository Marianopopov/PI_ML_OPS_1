from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

@app.get('/')
def bienvenida():
    return {'API de consultas a una base de datos de Steam, /docs en el link para acceder a las funciones de consulta.'}


@app.get('/PlayTimeGenre/ {genero}')

def PlayTimeGenre(genero: str):
    try:
        
        function1 = pd.read_csv('./Datasets/function1.csv')
        genre_df = function1[function1['genres'].apply(lambda x: genero in x)]
        grouped_df = genre_df.groupby('year')['playtime_forever'].sum()
        max_playtime_year = grouped_df.idxmax()
        result = {"Año de lanzamiento con más horas jugadas para Género X": str(max_playtime_year)}
        return result
    
    except Exception as e:
        return {"Error":str(e)}


@app.get('/UserForGenre/ {genero}')

def UserForGenre(genero):
    try:
        function2 = pd.read_csv('./Datasets/function1.csv')
        usuario_agrupado = function2[['user_id','playtime_forever','year']].groupby(['user_id','year']).sum()
        usuario_agrupado_1 = usuario_agrupado['playtime_forever'].idxmax()
        genero = function2[function2['genres'].apply(lambda x: genero in x)]
        df_usuario_genero = function2[(function2['user_id'] == usuario_agrupado_1 [0])]
        poranio = df_usuario_genero.groupby('year')[['playtime_forever']].sum().reset_index().rename(columns={'year':'Año', 'playtime_forever': 'Horas jugadas'}).to_dict(orient='records')
        dicc = {
            'Uruario con mas horas jugadas por genero': usuario_agrupado_1 [0],
            'Horas Jugadas': poranio
        }
        
        return dicc
    
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


@app.get('/sentiment_analysis/ {año}')
def recomendacion_juego(df, input_item_id, num_recommendations=5):
    try:
            df = pd.read_csv (r'Datasets\function3.csv.gz', compression='gzip', usecols=['user_id','item_id','sentiment_analysis','recommend','app_name'])
            # Filtrar el DataFrame para obtener solo las filas relacionadas con el juego de entrada
            input_item = df[df['item_id'] == input_item_id]

            if input_item.empty:
                return "El ID de producto no se encuentra en el conjunto de datos."

            # Extraer el usuario y sus recomendaciones para ese juego
            user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='recommend', fill_value=0)

            # Calcular la similitud coseno entre el juego de entrada y todos los demás juegos
            similarity_scores = cosine_similarity(user_item_matrix.T)

            # Obtener las puntuaciones de similitud para el juego de entrada
            input_item_scores = similarity_scores[input_item_id]

            # Obtener los índices de los juegos más similares (excluyendo el juego de entrada)
            similar_items_indices = input_item_scores.argsort()[:-1][-num_recommendations:]

            # Obtener los IDs y nombres de los juegos más similares
            similar_items = user_item_matrix.columns[similar_items_indices]

            # Obtener nombres de juegos correspondientes a los IDs
            game_names_df = df[df['item_id'].isin(similar_items)][['item_id', 'app_name']]

            # Fusionar los DataFrames para obtener nombres correspondientes a los IDs
            merged_df = pd.merge(pd.DataFrame(similar_items, columns=['item_id']), game_names_df, on='item_id')

            # Agrupar juegos recomendados por ID
            grouped_recommendations = merged_df.groupby(['item_id', 'app_name']).size().reset_index(name='counts')

            # Imprimir las recomendaciones de juegos agrupadas por ID
            print(f"Juegos recomendados similares al juego con ID: {input_item_id}")
            for idx, (item_id, game_name, count) in enumerate(zip(grouped_recommendations['item_id'], grouped_recommendations['app_name'], grouped_recommendations['counts']), start=1):
                print(f"{idx}. Juego ID: {item_id}, Name: {game_name}")
    
    except Exception as e:
        return {"Error": str(e)}