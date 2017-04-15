import cv2
import numpy as np

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.utils.inception_v3 import InceptionV3, conv2d_bn
from vis.visualization import visualize_saliency, visualize_cam, visualize_activation, get_num_filters

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

#  Build the VGG16 network with ImageNet weights
model_vgg = VGG16(weights='imagenet', include_top=True)
print('Model loaded.')
model = model_vgg

# Build the InceptionV3 network with ImageNet weights
# model = InceptionV3(weights='imagenet', include_top=True)
# print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py or inception_v3.py)
layer_name = 'predictions'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Images corresponding to tiger, penguin, dumbbell, speedboat, spider
image_paths = [
    "http://www.tigerfdn.com/wp-content/uploads/2016/05/How-Much-Does-A-Tiger-Weigh.jpg",
    "http://www.slate.com/content/dam/slate/articles/health_and_science/wild_things/2013/10/131025_WILD_AdeliePenguin.jpg.CROP.promo-mediumlarge.jpg",
    "http://www.kshs.org/cool2/graphics/dumbbell1lg.jpg",
    "http://media1.britannica.com/eb-media/80/150980-004-EE46999B.jpg",
]

heatmaps = []
# vis_images = []

for path in image_paths:
    
    # For InceptionV3
    seed_img = utils.load_img(path, target_size=(299, 299))
    pred = model.predict(preprocess_input(np.expand_dims(img_to_array(seed_img), axis=0)))
    
    # For VGG16
    # seed_img = utils.load_img(path, target_size=(224, 224))
    # pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))
    # print(utils.get_imagenet_label(pred_class))
    
    
    # seed_img = utils.load_img(path, target_size=(224, 224))
    # pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))
    
    print('Predicted:', decode_predictions(pred))
    print('Predicted:', decode_predictions(pred)[0][0][1])
    
    # pred_class = np.argmax(model.predict(preprocess_input(np.array([img_to_array(seed_img)]))))

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    # heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img, text=utils.get_imagenet_label(pred_class))
    heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img, text=utils.get_imagenet_label(pred_class))
    heatmaps.append(heatmap)
    
    # Generate three different images of the same output index.
    # vis_images = [visualize_activation(model, layer_idx, filter_indices=idx, text=str(idx), max_iter=500) for idx in [294, 294, 294]]
    # vis_images.append(vis_image)

name = "Gradient-based Localization map"
cv2.imshow(name, utils.stitch_images(heatmaps))
cv2.waitKey(-1)
cv2.destroyWindow(name)


# name = "Visualizations » Dense Layers"
# cv2.imshow(name, utils.stitch_images(vis_images))
# cv2.waitKey(-1)
# cv2.destroyWindow(name)



