from scrape_food2fork import food2fork_api, food2fork_recipe
from flask import Flask, request, render_template
from werkzeug import secure_filename
import cPickle as pickle
from dhdhd import NN_1
import os


UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

nn = NN_1()
nn.build()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# home page
@app.route('/')
def index():

    return render_template('index.html')



# My word counter app
@app.route('/recipe', methods=['POST'] )
def recipe():
    if request.method == 'POST':

        file = request.files['file']
        print file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    image_path = UPLOAD_FOLDER+filename

    # _, im = nn.preprocess_image(image_path)

    veg = nn.predict(image_path)
    print "\n"+ veg[0] + "\n"
    recipe_list = food2fork_api(veg[0].split(',')[0])

    recipe_title, image_url, ingredients = food2fork_recipe(recipe_list)

    ingredients.replace('\n', '<br>')
    return render_template('index.html',veg=veg[0].split(',')[0], \
    recipe_title = recipe_title, image_url = image_url, ingredients = ingredients\
     , section = "recipe")



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
