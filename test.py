import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import ImageToText
from mltu.utils.text_utils import get_cer
from mltu.configs import BaseModelConfigs

configs = BaseModelConfigs.load("./Models/handwriting_recognition/202301111911/configs.yaml")
model = ImageToText(model_path=configs.model_path, characters=configs.vocab)
validation_data = pd.read_csv("./Models/handwriting_recognition/202301111911/val.csv").values.tolist()
cer_list = []
for img_path, ground_truth in tqdm(validation_data):
    img = cv2.imread(img_path.replace("\\", "/"))
    predicted_text = model.predict(img)
    cer = get_cer(predicted_text, ground_truth)
    print(f"Image: {img_path}, Label: {ground_truth}, Prediction: {predicted_text}, CER: {cer}")
    cer_list.append(cer)
    scaled_img = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4))
    cv2.imshow("Image", scaled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
print(f"Average CER: {np.average(cer_list)}")
