import keras
from keras.models import *
from keras.layers import *
from types import MethodType
from data_loader import *

IMAGE_ORDERING = 'channels_last'

def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          epochs=6,
          batch_size=16,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=32,
          load_weights=None,
          steps_per_epoch=512,
          optimizer_name='adadelta'
          ):
          
    checkpoints_path = "drive/My Drive/NNDS/project/weights/segnet3_weights.h5"
    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if optimizer_name is not None:
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    train_gen = image_segmentation_generator(
        train_images, train_annotations,  batch_size,  n_classes,
        input_height, input_width, output_height, output_width)

    if validate:
        val_gen = image_segmentation_generator(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, output_height, output_width)

    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, epochs=1,
                                workers=-1,
                                use_multiprocessing=True)
            model.save_weights(checkpoints_path)
            print("Finished Epoch", ep)
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch,
                                validation_data=val_gen,
                                validation_steps=200,  epochs=1,
                                workers=-1,
                                use_multiprocessing=True)
            model.save_weights(checkpoints_path)
            print("Finished Epoch", ep)

def predict(model = None, inp=None, out_fname=None, input_height = 416, input_width = 608, colors = class_colors):
    
    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c)*(colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if out_fname is not None:
        cv2.imwrite(out_fname, seg_img)

    return pr

def predict_segmentation(model, input_height = 416, input_width = 608, 
                         path = "drive/My Drive/NNDS/project/weights/segnet3_weights.h5"):
    ## load model
    model.load_weights(path)

    test_folder = glob.glob('drive/My Drive/NNDS/project/CamVid/test/*.png')
    for file in tqdm(test_folder):
        file_name = file.split('/')[-1]
        out = predict(model,
                      inp=file,
                      out_fname=f"drive/My Drive/NNDS/project/CamVid/Predictions/{file_name}")

def evaluate_segmentation(model, inp_images_dir=None, annotations_dir=None,
                          img_height = 416, img_width = 618,
                          path = "drive/My Drive/NNDS/project/weights/segnet3_weights.h5"):
    
    model.load_weights(path)
        
    images = glob.glob(inp_images_dir + "*.jpg") + glob.glob(inp_images_dir + "*.png") + glob.glob(inp_images_dir + "*.jpeg")
    images.sort()
    segmentations = glob.glob(annotations_dir + "*.jpg") + glob.glob(annotations_dir + "*.png") + glob.glob(annotations_dir + "*.jpeg")
    segmentations.sort()
        
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    n_pixels = np.zeros(n_classes)
    
    for inp, ann in tqdm(zip(images, segmentations)):
        pr = predict(model, inp)
        gt = get_segmentation_array(ann, n_classes, model.output_width, model.output_height, no_reshape=True)
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()
                
        for cl_i in range(n_classes):
            
            tp[ cl_i ] += np.sum((pr == cl_i) * (gt == cl_i))
            fp[ cl_i ] += np.sum((pr == cl_i) * ((gt != cl_i)))
            fn[ cl_i ] += np.sum((pr != cl_i) * ((gt == cl_i)))
            n_pixels[cl_i] += np.sum(gt == cl_i)
            
    cl_wise_score = tp / ( tp + fp + fn + 0.000000000001 )
    n_pixels_norm = n_pixels/np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)
    return {"frequency_weighted_IU":frequency_weighted_IU , "mean_IU":mean_IU , "class_wise_IU":cl_wise_score }
