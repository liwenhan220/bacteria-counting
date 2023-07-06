from extract_feature import *
import os


directories = ["extracted_data", "debug"]

for dir in directories:
    if not os.path.isdir(dir):
        os.mkdir(dir)

raw_folder = "raw_data/"

ecoli = "Ecoli-positive/"
styphi = "Styphi-positive/"
negative = "negative/"

test_bacts = []
train_bacts = []
max_shape = (1, 1)
test_split_index = 1

# load ecoli data
for i in range(1, 11):
    name = "E.coli + {}.bmp".format(i)
    input_img_path = raw_folder + ecoli + name
    img = cv2.imread(input_img_path)
    bacts, shape = generate_bacts(img, 0, debug = True, debug_path = directories[1] + '/' + name, image_name=name, threshold = 0.7, max_diameter=float('inf'))
    if i <= test_split_index:
        test_bacts += bacts
    else:
        train_bacts += bacts
    max_shape = max_box(max_shape, shape)

# load styphi data
for i in range(1, 17):
    name = "S.typhi + {}.bmp".format(i)
    input_img_path = raw_folder + styphi + name
    img = cv2.imread(input_img_path)
    bacts, shape = generate_bacts(img, 0, debug = True, debug_path = directories[1] + '/' + name, image_name=name, threshold = 0.0,  max_diameter=9)
    if i <= test_split_index:
        test_bacts += bacts
    else:
        train_bacts += bacts
    max_shape = max_box(max_shape, shape)

# load negative data
for i in range(1, 12):
    name = "swab-{}.bmp".format(i)
    input_img_path = raw_folder + negative + "swab-{}.bmp".format(i)
    img = cv2.imread(input_img_path)
    if i > 1:
        bacts, shape = generate_bacts(img, 1, cover_corners= False, debug = True, debug_path = directories[1] + '/' + name, image_name=name, threshold = 0, max_diameter=float('inf'))
    else:
        bacts, shape = generate_bacts(img, 1, cover_corners= False, debug = True, debug_path = directories[1] + '/' + name, image_name=name, threshold = 0, max_diameter=float('inf'))

    if i <= test_split_index:
        test_bacts += bacts
    else:
        train_bacts += bacts
    max_shape = max_box(max_shape, shape)

X = []
y = []
for bact in train_bacts:
    bact.pad_img(max_shape)
    X.append(bact.bg_normalized())
    y.append(bact.label)

np.save(directories[0] + "/trainX.npy", X)
np.save(directories[0] + "/trainY.npy", y)

testX = []
testY = []

for bact in test_bacts:
    bact.pad_img(max_shape)
    testX.append(bact.bg_normalized())
    testY.append(bact.label)

np.save(directories[0] + "/testX.npy", testX)
np.save(directories[0] + "/testY.npy", testY)
