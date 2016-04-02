import urllib


i=0
with open('../veg_lisst', 'r') as f:

    for line in f:
        i += 1
        out = line.split(':')
        name =  out[0].strip()
        _id =  out[1].strip()

        url_pre = "http://image-net.org/download/synset?wnid="+ _id +"&username=a1264029&accesskey=ec8dde9ec11bdc1098eb8de9ad8471674f2e4e56&release=latest&src=stanford"
        fname = name+'.tar'

        urllib.urlretrieve(url_pre, fname)

        print "Finish %i item of %s " %(i, name)


# veg list
#
# downloaded:
# 'broccoli'
# 'cauliflower'
# strawberry
# blueberry
# raspberry
# blackberry
# cranberry
# kidney beans: n07727048
# navy beans: n07727140
# pinto beans: n07727252
# black beans: n07727458
# yellow beans: n07728708
# lima beans: n07729000
# broad beans: n07729384
# green soybeans: n07729828
# flageolet: n07727741
# soy bean: n07729485
# lentil: n07725255
# green pea: n07725531
# black eyed pea: n07726672
# bean sprout: n07719616
# chard: n07720277
# buttercrunch: n07723968
# iceberg lettuce: n07724269
# romaine: n07724492
# chicory: n07730855
# cress: n07732747
# dandelion green: n07733217
# wild spinach: n07733712
# turnip greens: n07736256
# sorrel: n07736371
# spinach: n07736692
# potato: n07710616
# eggplant: n07713074
# sweet pepper: n07720615
# tobasco: n07722052
# chili pepper: n07721456
# beefsteak tomato: n07734183
# plum tomato: n07734417
# tomatillo: n07734555
# sweet potato: n07712063
# jerusalem artichoke: n07719058
# beet: n07719839
# carrot: n07730207
# salsify: n07735294
# radish: n07735687
# turnip: n07735803
# taro: n07736813
# rhubarb: n07713267
# kale: n07714078
# chinese cabbage: n07714287
# bok choy: n07714448
# head cabbage: n07714571
# brussels sprouts: n07715221
# turnip cabbage: n07733567
# acorn squash: n07717410
# butternut squash: n07717556
# buttercup squash: n07718068
# yellow squash: n07716034
# zucchini: n07716358
# spaghetti squash: n07716906
# cucumber: n07718472
# artichoke: n07718747
# asparagus: n07719213
# bamboo shoot: n07719330
# green onion: n07722485
# shallot: n07723177
# spanish onion: n07722763
# leek: n07723039
# cardoon: n07730033
# celery: n07730406
# okra: n07733394
# mushroom: n07734744
# pumpkin: n07735510
# plaintain: n07768423
# fennel: n07817871
# star fruit:n07746551
# orange:n07747607
# tangerine:n07748416
# kumquat:n07749446
# lemon:n07749582
# lime:n13134947
# grape fruit:n07749969
# pomelo: n07750146
# apricot: n07750872
# peach: n07751004
# nectarine: n07751148
# pitahaya: n07751280
# damson plum: n07751737
# greengage plum: n07751858
# beach plum: n07751977
# fig: n07753113
# pineapple: n07753275
# anchovy pear: n07753448
