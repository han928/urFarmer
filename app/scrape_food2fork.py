# "Powered By Food2Fork.com".

import requests
from bs4 import BeautifulSoup
import json
import random

with open('food2fork_api_key.json', 'r') as f:
    api_key = json.load(f)['key']


def food2fork_api(ingredient):
    url = 'http://food2fork.com/api/search?key='+api_key +'&q='+ingredient
    response = requests.get('http://food2fork.com/api/search?key=0a808e45c8ca52288b17d5e956b54431&q='+ingredient).json()

    return response['recipes']


def food2fork_recipe(recipe_list):
    """
    Get the recipe from the food2fork network
    url: url from food2fork api json response
    recipe_title: title of the recipe
    image_url: url for the recipe title image_url
    ingredients: ingredients listed on food2fork
    """
    recipe = random.choice(recipe_list)
    url = recipe['f2f_url']
    print "html:", url
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser')
    tag = soup.find(class_ = 'span5 offset1 about-container')

    recipe_title = tag.find(class_ = 'recipe-title').text
    image_url = tag.find('img')['src']
    ingredients = tag.find('ul').prettify()
    print 'funct', ingredients

    return (recipe_title, image_url, ingredients)
