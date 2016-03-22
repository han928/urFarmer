from flask import Flask, request, render_template
import cPickle as pickle
from nn import NN
from scrape_food2fork import food2fork_api, food2fork_recipe
app = Flask(__name__)

nn = NN()
nn.build()

# home page
@app.route('/')
def index():


    return render_template('index.html')



# My word counter app
@app.route('/recipe', methods=['POST'] )
def recipe():
    image = str(request.form['image'])

    _, im = nn.preprocess_image(image)

    veg = nn.predict(im)

    recipe_list = food2fork_api(veg[0])

    recipe_title, image_url, ingredients = food2fork_recipe(recipe_list)
    return render_template('index.html',veg=veg[0], recipe_title = recipe_title, image_url = image_url, ingredients = ingredients , section = "recipe")



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
