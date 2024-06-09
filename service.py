import bentoml
import numpy as np
import tensorflow as tf
import keras_cv
import onnxruntime as ort
from keras_cv.src.backend import ops
from PIL.Image import Image as PILImage
from PIL import ImageDraw, Image
from pathlib import Path
from typing import Annotated  # Python 3.9 or above
from typing_extensions import Annotated  # Older than 3.9
from bentoml.validators import ContentType

IMAGE_WH = (640, 640)
BOX_REGRESSION_CHANNELS = 64

# --- NMS Функции ---
def decode_regression_to_boxes(preds):
    preds_bbox = tf.keras.layers.Reshape((-1, 4, BOX_REGRESSION_CHANNELS // 4))(
        preds
    )
    preds_bbox = ops.nn.softmax(preds_bbox, axis=-1) * ops.arange(
        BOX_REGRESSION_CHANNELS // 4, dtype="float32"
    )
    return ops.sum(preds_bbox, axis=-1)

def get_anchors(
    image_shape,
    strides=[8, 16, 32],
    base_anchors=[0.5, 0.5],
):
    base_anchors = ops.array(base_anchors, dtype="float32")

    all_anchors = []
    all_strides = []
    for stride in strides:
        hh_centers = ops.arange(0, image_shape[0], stride)
        ww_centers = ops.arange(0, image_shape[1], stride)
        ww_grid, hh_grid = ops.meshgrid(ww_centers, hh_centers)
        grid = ops.cast(
            ops.reshape(ops.stack([hh_grid, ww_grid], 2), [-1, 1, 2]),
            "float32",
        )
        anchors = (
            ops.expand_dims(
                base_anchors * ops.array([stride, stride], "float32"), 0
            )
            + grid
        )
        anchors = ops.reshape(anchors, [-1, 2])
        all_anchors.append(anchors)
        all_strides.append(ops.repeat(stride, anchors.shape[0]))

    all_anchors = ops.cast(ops.concatenate(all_anchors, axis=0), "float32")
    all_strides = ops.cast(ops.concatenate(all_strides, axis=0), "float32")

    all_anchors = all_anchors / all_strides[:, None]

    all_anchors = ops.concatenate(
        [all_anchors[:, 1, None], all_anchors[:, 0, None]], axis=-1
    )
    return all_anchors, all_strides

def dist2bbox(distance, anchor_points):
    left_top, right_bottom = ops.split(distance, 2, axis=-1)
    x1y1 = anchor_points - left_top
    x2y2 = anchor_points + right_bottom
    return ops.concatenate((x1y1, x2y2), axis=-1)  # xyxy bbox

def decode_predictions(prediction_decoder, pred, images):
        boxes = pred[0]
        scores = pred[1]

        boxes = decode_regression_to_boxes(boxes)

        anchor_points, stride_tensor = get_anchors(image_shape=images.shape[1:])
        stride_tensor = ops.expand_dims(stride_tensor, axis=-1)

        box_preds = dist2bbox(boxes, anchor_points) * stride_tensor
        box_preds = keras_cv.bounding_box.convert_format(
            box_preds,
            source="xyxy",
            target="xywh",
            images=images,
        )

        return prediction_decoder(box_preds, scores)

prediction_decoder = keras_cv.layers.NonMaxSuppression(
                bounding_box_format='xywh',
                from_logits=False,
                confidence_threshold=0.2,
                iou_threshold=0.7,
            )

def predict_onnx(onnx_sess, image, prediction_decoder):
    y_pred = onnx_sess.run(output_names=None, input_feed= {'input': image.numpy()})
    y_pred = decode_predictions(prediction_decoder, y_pred, image)
    return y_pred

def draw_bounding_boxes(image, boxes, color='red', width=2):
  """Рисует прямоугольники (bounding boxes) на изображении.

  Args:
    image: PIL Image, на котором нужно рисовать.
    boxes: NumPy массив с bounding boxes в формате [x_min, y_min, x_max, y_max].
    color: Цвет прямоугольников.
    width: Толщина линий.
  """
  image = Image.fromarray(image)
  draw = ImageDraw.Draw(image)
  for box in boxes:
      x, y, w, h = box
      x0 = int(x)
      y0 = int(y)
      x1 = int(x + w)  # Вычисляем x1
      y1 = int(y + h)  # Вычисляем y1
      draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
  return image
# --- END---

@bentoml.service(
    resources={"gpu": 1},  
    traffic={"timeout": 10},  
)
class FaceDetectionService:
    def __init__(self) -> None:
       self.onnx_sess = ort.InferenceSession("final_model_yolov8s_best_bs16_e50.onnx", providers=["CUDAExecutionProvider"])

    @bentoml.api
    def predict(self, input_image: Annotated[Path, ContentType('image/jpeg')]) -> PILImage:
        # Предобработка изображения
        image = Image.open(input_image)
        image = tf.keras.utils.img_to_array(image)
        image = tf.expand_dims(image, axis=0)
        resizing = keras_cv.layers.Resizing(640, 640, pad_to_aspect_ratio=True)
        image = resizing(image)
        
        # Предсказание bounding boxes
        y_pred = predict_onnx(self.onnx_sess, image, prediction_decoder)

        # Оставляем только релевантные bbox
        array = y_pred['boxes'].numpy()[0]
        indices = np.where(np.all(array == -1, axis=1))[0]
        filtered_array = np.delete(array, indices, axis=0)
                
        return draw_bounding_boxes(tf.cast(image, dtype=tf.uint8).numpy()[0], filtered_array)