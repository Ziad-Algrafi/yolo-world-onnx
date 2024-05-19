import cv2 as cv
import numpy as np
import onnxruntime as ort
import torch
import clip

def select_device(device):
    if device == "0":
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            return "cuda"
        elif 'ROCMExecutionProvider' in providers:
            return "rocm"
        elif 'DmlExecutionProvider' in providers:
            return "dml"
        elif 'OpenVINOExecutionProvider' in providers:
            return "openvino"
        elif 'TensorrtExecutionProvider' in providers:
            return "tensorrt"
        else:
            return "cpu"
    elif device == "cpu":
        return "cpu"
    else:
        print("Invalid device input or GPU is not Available. For chossing a CPU device enter 'cpu' and for a GPU enters '0', defaulting to a cpu")
        return "cpu"

class TextEmbedder:
    def __init__(self, model_name="ViT-B/32", device="cpu"):
        self.device = select_device(device)
        self.clip_model, _ = clip.load(model_name, device=self.device)

    def __call__(self, text):
        return self.embed_text(text)

    def embed_text(self, text):
        if not isinstance(text, list):
            text = [text]

        text_token = clip.tokenize(text).to(self.device)
        txt_feats = [self.clip_model.encode_text(token).detach() for token in text_token.split(1)]
        txt_feats = torch.cat(txt_feats, dim=0)
        txt_feats /= txt_feats.norm(dim=1, keepdim=True)
        txt_feats = txt_feats.unsqueeze(0)

        return txt_feats

class YOLOWORLD:
    def __init__(self, path, device="cpu"):
        self.device = select_device(device)
        providers = self.get_providers(self.device)
        self.session = ort.InferenceSession(path, providers=providers)
        self.names = []
        self.text_embedder = TextEmbedder(device=self.device)
        self.class_embeddings = None

        self.get_input_details()
        self.get_output_details()

    def get_providers(self, device):
        if device == "cuda":
            return ["CUDAExecutionProvider"]
        elif device == "rocm":
            return ["ROCMExecutionProvider"]
        elif device == "dml":
            return ["DmlExecutionProvider"]
        elif device == "openvino":
            return ["OpenVINOExecutionProvider"]
        elif device == "tensorrt":
            return ["TensorrtExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]

    def __call__(self, image, conf=0.2, imgsz=640, iou=0.3):
        return self.detect_objects(image, conf, imgsz, iou)

    def set_classes(self, classes):
        self.names = classes
        self.class_embeddings = self.text_embedder(classes)

    def detect_objects(self, image, conf, imgsz, iou):
        input_tensor = self.prepare_input(image, imgsz)
        class_embeddings = self.prepare_embeddings(self.class_embeddings)

        outputs = self.inference(input_tensor, class_embeddings)

        return self.process_output(outputs, conf, imgsz, iou)

    def prepare_input(self, image, imgsz):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        input_img = cv.resize(input_img, (imgsz, imgsz))

        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def prepare_embeddings(self, class_embeddings):
        if class_embeddings.shape[1] != self.num_classes:
            class_embeddings = torch.nn.functional.pad(class_embeddings, (0, 0, 0, self.num_classes - class_embeddings.shape[1]), mode='constant', value=0)

        return class_embeddings.cpu().numpy().astype(np.float32)

    def inference(self, input_tensor, class_embeddings):
        outputs = self.session.run(self.output_names,
                                   {self.input_names[0]: input_tensor, self.input_names[1]: class_embeddings})
        return outputs

    def process_output(self, output, conf, imgsz, iou):
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores >= conf, :]
        scores = scores[scores > conf]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        boxes = self.extract_boxes(predictions, imgsz)

        detections = [(class_id, x, y, w, h, score)
                      for class_id, (x, y, w, h), score in zip(class_ids, boxes, scores)]

        nms_detections = self.apply_nms(detections, iou)

        boxes = []
        scores = []
        class_ids = []
        for det in nms_detections:
            class_id, x_nms, y_nms, w_nms, h_nms, score = det
            boxes.append([x_nms, y_nms, w_nms, h_nms])
            scores.append(score)
            class_ids.append(class_id)

        return boxes, scores, class_ids

    def extract_boxes(self, predictions, imgsz):
        boxes = predictions[:, :4]

        boxes[:, 0] /= imgsz
        boxes[:, 1] /= imgsz
        boxes[:, 2] /= imgsz
        boxes[:, 3] /= imgsz 

        boxes[:, 0] *= self.img_width
        boxes[:, 1] *= self.img_height
        boxes[:, 2] *= self.img_width
        boxes[:, 3] *= self.img_height

        return boxes

    def apply_nms(self, detections, iou_threshold):
        boxes = []
        for det in detections:
            (cls_id, x, y, w, h, confidence) = det
            boxes.append([x, y, w, h, cls_id, confidence])

        sorted_boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
        selected_boxes = []

        while len(sorted_boxes) > 0:
            selected_boxes.append(sorted_boxes[0])
            remaining_boxes = []

            for box in sorted_boxes[1:]:
                x1_a, y1_a, w1_a, h1_a, _, _ = selected_boxes[-1]
                x1_b, y1_b, w1_b, h1_b, _, _ = box

                x2_a = x1_a + w1_a
                y2_a = y1_a + h1_a
                x2_b = x1_b + w1_b
                y2_b = y1_b + h1_b

                intersection = max(0, min(x2_a, x2_b) - max(x1_a, x1_b)) * max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
                union = w1_a * h1_a + w1_b * h1_b - intersection

                if union == 0:
                    iou = 0
                else:
                    iou = intersection / union

                if iou < iou_threshold:
                    remaining_boxes.append(box)

            sorted_boxes = remaining_boxes

        nms_detections = []
        for box in selected_boxes:
            x, y, w, h, cls_id, confidence = box
            nms_detections.append((cls_id, x, y, w, h, confidence))

        return nms_detections

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.num_classes = model_inputs[1].shape[1]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]