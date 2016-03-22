from flask import Flask, request, render_template
import cPickle as pickle
from nn import NN
from scrape_food2fork import food2fork_api, food2fork_recipe
app = Flask(__name__)




# home page
@app.route('/')
def index():


    return render_template('index.html')



@app.route('/submit')
def submission_page():
    return '''
        <!DOCTYPE html>
        <html>
            <head>
                <meta charset="utf-8">
                <title>What Section Does Your Articles Belongs to </title>
            </head>
                <body>
                    <!-- page content -->
                    <h1>Check your articles sections!</h1>

                    <form action="/recipe" method='POST' >
                        <input type="text" name="user_input" />
                        <input type="submit" />
                    </form>
                </body>
        </html>

        '''



# My word counter app
@app.route('/recipe', methods=['POST'] )
def recipe():
    text = str(request.form['user_input'])

    response_recipe = unirest.get("https://spoonacular-recipe-food-nutrition-v1.p.mashape.com/recipes/findByIngredients?ingredients="+text+"&limitLicense=false&number=5&ranking=1",
      headers={
        "X-Mashape-Key": api_key,
        "Accept": "application/json"
      }
    )

    r_id = response_recipe.body[0]['id']

    response = unirest.get("https://spoonacular-recipe-food-nutrition-v1.p.mashape.com/recipes/"+ str(r_id) + "/information",
      headers={
        "X-Mashape-Key": api_key
      }
    )



    return response.body['sourceUrl']



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
