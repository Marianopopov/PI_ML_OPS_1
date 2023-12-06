from fastapi import FastAPI
import pandas as pd

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

