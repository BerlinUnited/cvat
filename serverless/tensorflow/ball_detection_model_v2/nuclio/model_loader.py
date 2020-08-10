
import numpy as np
from PIL import Image
import tensorflow as tf

class ModelLoader:
    def __init__(self, model_path):
        self.model = tf.saved_model.load(model_path)


    def infer(self, image):
        width, height = image.size
        if width > 1920 or height > 1080:
            image = image.resize((width // 2, height // 2), Image.ANTIALIAS)
        image_np = np.array(image.getdata()).reshape((image.height, image.width, 3)).astype(np.uint8)
        image_np = np.expand_dims(image_np, axis=0)

        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        model_fn = model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() 
                     for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        print("output dict:")
        print(output_dict)
        return output_dict
