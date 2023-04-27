import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import cv2

from data import load_data, tf_dataset
from evaluate import iou_dice_score, f1score
from Unet import build_unet
from sklearn.metrics import jaccard_score, accuracy_score, confusion_matrix

H = 512
W = 512
num_classes = 3

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Dataset """
    path = r"E:\post 17-1-2021\FYP\god"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data()
    print(f"Dataset: Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    filename = "model100_attunet2d"

    """ Model """
    folder = os.path.dirname(os.path.abspath(__file__))
    model = tf.keras.models.load_model(fr"E:\post 17-1-2021\FYP\god\models\{filename}.h5")

    # Recording metrics
    count = 0; mIoUc = mIoUs = 0; mdicec = mdices = 0; mPA = 0; mf1c = mf1s = 0
    f = open(os.path.join(folder, f"results\{filename}\printout {filename}.txt"), 'w')

    """ Saving the masks """
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        count += 1
        name = x.split("\\")[-1]
        
        if count == 1: f.write(f"Image {count}: {name}\n")
        else: f.write(f"\n\nImage {count}: {name}\n")

        ## Read image
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (W, H))
        x = x / 255.0
        x = x.astype(np.float32)

        ## Read mask
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        y = cv2.resize(y, (W, H))   ## (512, 512)
        y = np.expand_dims(y, axis=-1) ## (512, 512, 1)
        y = y.astype(np.int32)
        y = np.concatenate([y, y, y], axis=2)

        ## Prediction
        p = model.predict(np.expand_dims(x, axis=0))[0]
        p = np.argmax(p, axis=-1)
        p = np.expand_dims(p, axis=-1)
        p[p == 0] = 0
        p[p == 1] = 128
        p[p == 2] = 255
        p = p.astype(np.int32)
        p = np.concatenate([p, p, p], axis=2)

        x = x * 255.0
        x = x.astype(np.int32)

        h, w, _ = x.shape
        line = np.ones((h, 10, 3)) * 255

        # print(x.shape, line.shape, y.shape, line.shape, p.shape)

        final_image = np.concatenate([x, line, y, line, p], axis=1)
        
        cv2.imwrite(fr"E:\post 17-1-2021\FYP\god\results\{filename}\{name}", final_image)
        
        f.write(str(np.sum(y == 0)) + " " + str(np.sum(y == 128)) + " " + str(np.sum(y == 255)) + "\n")
        f.write(str(np.sum(p == 0)) + " " + str(np.sum(p == 128)) + " " + str(np.sum(p == 255)) + "\n")
        
        ## Evaluate
        C = 128
        S = 255
        
        # Metric 1: IoU
        f.write("\nMetric 1: IoU\n")
        iouc, dicec = iou_dice_score(y.flatten(), p.flatten(), C)
        ious, dices = iou_dice_score(y.flatten(), p.flatten(), S)
        f.write(f"IoU for Choroid: {iouc:.3f}\n")
        f.write(f"IoU for Sclera: {ious:.3f}\n")
        mIoUc += iouc
        mIoUs += ious
        
        # Metric 2: Dice Coefficient
        f.write("\nMetric 2: Dice Coefficient\n")
        f.write(f"Dice for Choroid: {dicec:.3f}\n")
        f.write(f"Dice for Sclera: {dices:.3f}\n")
        mdicec += dicec
        mdices += dices
        
        # Metric 3: Pixel Accuracy
        f.write("\nMetric 3: Pixel Accuracy\n")
        acc = accuracy_score(y.flatten(), p.flatten())
        f.write(f"Pixel Accuracy: {acc}\n")
        mPA += acc
        
        # Metric 4: F1 Score
        f.write("\nMetric 4: F1 Score\n")
        (tpc, fpc, fnc, tnc, precisionc, recallc, f1c) = f1score(y, p, C)
        (tps, fps, fns, tns, precisions, recalls, f1s) = f1score(y, p, S)
        f.write(f"F1 Score for Choroid: {f1c:.3f}\n")
        f.write(f"Precision and Recall for Choroid: {precisionc:.3f} {recallc:.3f}\n")
        f.write(f"F1 Score for Sclera: {f1s:.3f}\n")
        f.write(f"Precision and Recall for Sclera: {precisions:.3f} {recalls:.3f}\n")
        mf1c += f1c
        mf1s += f1s
        
        # Metric 5: Confusion Matrix
        f.write("\nMetric 5: Confusion Matrix\n")
        f.write("Confusion Matrix for Choroid:\n")
        f.write(f"TP: {tpc}, FP: {fpc}, FN: {fnc}, TN: {tnc}\n")
        f.write("Confusion Matrix for Sclera:\n")
        f.write(f"TP: {tps}, FP: {fps}, FN: {fns}, TN: {tns}\n")
        

    mIoUc /= count
    mIoUs /= count
    mdicec /= count
    mdices /= count
    mPA /= count
    mf1c /= count
    mf1s /= count
    f.write(f"\nMean IoU for Choroid: {mIoUc:.3f}\n")
    f.write(f"Mean IoU for Sclera: {mIoUs:.3f}\n")
    f.write(f"Mean Dice for Choroid: {mdicec:.3f}\n")
    f.write(f"Mean Dice for Sclera: {mdices:.3f}\n")
    f.write(f"Mean Pixel Accuracy: {mPA:.3f}\n")
    f.write(f"Mean F1 Score for Choroid: {mf1c:.3f}\n")
    f.write(f"Mean F1 Score for Sclera: {mf1s:.3f}\n")
