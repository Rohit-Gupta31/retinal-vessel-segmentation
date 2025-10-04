import os
import cv2
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
np.random.seed(0)

# CLAHE
def clahe_equalized(imgs):    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))    
    imgs_equalized = clahe.apply(imgs)
    return imgs_equalized

patch_size = 512

#loading model architectures
from model import unetmodel, residualunet
from tensorflow.keras.optimizers import Adam
from evaluation_metrics import IoU_coef,IoU_loss

IMG_HEIGHT = patch_size
IMG_WIDTH = patch_size
IMG_CHANNELS = 1

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# model = unetmodel(input_shape) 
model = residualunet(input_shape)
model.compile(optimizer = Adam(learning_rate=1e-3), loss=IoU_loss, metrics=['accuracy', IoU_coef])
model.load_weights('/teamspace/studios/this_studio/RetinalVesselSegmentation/retina_Unet_100epochs.keras') #loading weights


path1 = '/teamspace/studios/this_studio/RetinalVesselSegmentation/retinalVesselData/test/image'    #test dataset images directory path
path2 = '/teamspace/studios/this_studio/RetinalVesselSegmentation/retinalVesselData/test/mask'     #test dataset mask directory path


from sklearn.metrics import jaccard_score,confusion_matrix

testimg = []
ground_truth = []
prediction = []
global_IoU = []
global_accuracy = []

testimages = sorted(os.listdir(path1))
testmasks =  sorted(os.listdir(path2))

for idx, image_name in enumerate(testimages):  
   if image_name.lower().endswith((".jpg", ".png", ".tif")): 
      predicted_patches = []
      test_img = skimage.io.imread(path1+"/"+image_name)
     
      test = test_img[:,:,1] #selecting green channel
      test = clahe_equalized(test) #applying CLAHE
      SIZE_X = (test_img.shape[1]//patch_size)*patch_size #getting size multiple of patch size
      SIZE_Y = (test_img.shape[0]//patch_size)*patch_size #getting size multiple of patch size
      test = cv2.resize(test, (SIZE_X, SIZE_Y))
      testimg.append(test)           
      test = np.array(test)

      patches = patchify(test, (patch_size, patch_size), step=patch_size) #create patches(patch_sizexpatch_sizex1)

      for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                  single_patch = patches[i,j,:,:]
                  single_patch_norm = (single_patch.astype('float32')) / 255.
                  single_patch_norm = np.expand_dims(np.array(single_patch_norm), axis=-1)
                  single_patch_input = np.expand_dims(single_patch_norm, 0)
                  single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8) #predict on single patch
                  predicted_patches.append(single_patch_prediction)
      predicted_patches = np.array(predicted_patches)
      predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], patch_size,patch_size) )
      reconstructed_image = unpatchify(predicted_patches_reshaped, test.shape) #join patches to form whole img
      prediction.append(reconstructed_image) 

      groundtruth=[]
      groundtruth = skimage.io.imread(path2+'/'+testmasks[idx]) #reading mask of the test img      
      SIZE_X = (groundtruth.shape[1]//patch_size)*patch_size 
      SIZE_Y = (groundtruth.shape[0]//patch_size)*patch_size  
      groundtruth = cv2.resize(groundtruth, (SIZE_X, SIZE_Y))  
      ground_truth.append(groundtruth)

      y_true = groundtruth
      y_pred = reconstructed_image
      labels = [0, 1]
      IoU = []
      for label in labels:
          jaccard = jaccard_score(y_pred.flatten(),y_true.flatten(), pos_label=label, average='weighted')
          IoU.append(jaccard)
      IoU = np.mean(IoU) #jacard/IoU of single image
      global_IoU.append(IoU)

      cm=[]
      accuracy = []
      cm = confusion_matrix(y_true.flatten(),y_pred.flatten(), labels=[0, 1])
      accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]) #accuracy of single image
      global_accuracy.append(accuracy)


avg_acc =  np.mean(global_accuracy)
mean_IoU = np.mean(global_IoU)

print('Average accuracy is',avg_acc)
print('mean IoU is',mean_IoU)


#checking segmentation results
import random
test_img_number = random.randint(0, len(testimg) - 1)   # ✅ fixed


plt.figure(figsize=(30, 18))

# 1. Original RGB image
plt.subplot(231)
plt.title('Original RGB Test Image')
plt.xticks([])
plt.yticks([])
plt.imshow(test_img)  # full RGB

# 2. Green channel / CLAHE image (used as input)
plt.subplot(232)
plt.title('Green Channel + CLAHE')
plt.xticks([])
plt.yticks([])
plt.imshow(testimg[test_img_number])

# 3. Ground truth mask
plt.subplot(233)
plt.title('Ground Truth Mask')
plt.xticks([])
plt.yticks([])
plt.imshow(ground_truth[test_img_number], cmap='gray')

# 4. Predicted mask
plt.subplot(234)
plt.title('Predicted Mask')
plt.xticks([])
plt.yticks([])
plt.imshow(prediction[test_img_number], cmap='gray')

# Save the figure
plt.savefig("test_image_4views.png", dpi=300, bbox_inches='tight')
plt.show()


# prediction on single image
from datetime import datetime 
reconstructed_image = []
test_img = skimage.io.imread(
    '/teamspace/studios/this_studio/RetinalVesselSegmentation/retinalVesselData/test/image/3.png'
)  # test image

predicted_patches = []
start = datetime.now()   

test = test_img[:, :, 1]  # selecting green channel
test = clahe_equalized(test)  # applying CLAHE
SIZE_X = (test_img.shape[1] // patch_size) * patch_size
SIZE_Y = (test_img.shape[0] // patch_size) * patch_size
test = cv2.resize(test, (SIZE_X, SIZE_Y))        
test = np.array(test)

patches = patchify(test, (patch_size, patch_size), step=patch_size)

for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        single_patch = patches[i, j, :, :]
        single_patch_norm = (single_patch.astype('float32')) / 255.
        single_patch_norm = np.expand_dims(np.array(single_patch_norm), axis=-1)
        single_patch_input = np.expand_dims(single_patch_norm, 0)
        single_patch_prediction = (
            model.predict(single_patch_input)[0, :, :, 0] > 0.5
        ).astype(np.uint8)
        predicted_patches.append(single_patch_prediction)

predicted_patches = np.array(predicted_patches)
predicted_patches_reshaped = np.reshape(
    predicted_patches, (patches.shape[0], patches.shape[1], patch_size, patch_size)
)
reconstructed_image = unpatchify(predicted_patches_reshaped, test.shape)

stop = datetime.now()
print('Execution time: ', (stop - start))

plt.figure(figsize=(14, 7))
plt.subplot(121)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.imshow(test_img)

plt.subplot(122)
plt.title('Prediction')
plt.xticks([])
plt.yticks([])
plt.imshow(reconstructed_image, cmap='gray')

plt.savefig("test_result_residualUnet.png", dpi=300, bbox_inches='tight')
plt.show()

