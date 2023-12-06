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
def UsersRecommend(year: str):
    try:   
        function3 = pd.read_csv('./Datasets/function3.csv')
        filtered_df = function3[(function3['recommend'] == 1) & (function3['sentiment_analysis'].isin([1, 2])) & (function3['year'] == year)]
        game_recommendations = filtered_df.groupby('app_name')['recommend'].count().sort_values(ascending=False)
        top_3_games = game_recommendations.head(3)
        result = [{"Puesto {}: {}".format(i + 1, juego)} for i, juego in enumerate(top_3_games)]
        
        return result

    except Exception as e:
        return {"Error": str(e)}
