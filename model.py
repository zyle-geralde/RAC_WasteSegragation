'''from datasets import load_dataset

ds = load_dataset("nflechas/recycling_app", "full")
print(ds)


image = ds["train"][0]["image"]   # PIL.Image
label = ds["train"][0]["objects"]   # integer index
label_name = ds["train"].features["objects"].feature["category"].int2str(0)

print(image)
print(label)
print(label_name)'''

from datasets import load_dataset
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import Counter
import numpy as np
import faiss
from PIL import Image
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import concatenate_datasets


#load the dataset
ds = load_dataset("nflechas/recycling_app", "full")
print(ds)

#example of accessing data (not directly used in the embedding process)
image_example = ds["train"][2]["image"]   #PIL.Image
label_example = ds["train"][2]["objects"]   #integer index
label_name_example = ds["train"].features["objects"].feature["category"].int2str(2)

print(image_example)
print(label_example)
print(label_name_example)

#load MobileNetV2 model as a feature extractor
feature_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

#convert image to embedding
def get_embedding(img):
    img = img.resize((224, 224)).convert('RGB')
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_input(img_array)
    embedding = feature_extractor.predict(img_array)
    return embedding[0].astype('float32')

#load prebuilt FAISS index and labels
def load_index(index_path="faiss_index.bin", label_path="labels.pkl"):
    index = faiss.read_index(index_path)
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)
    return index, labels

#predict via nearest neighbor lookup
def predict(img, index, labels, k=5):
    query_vec = get_embedding(img).reshape(1, -1)
    distances, indices = index.search(query_vec, k)
    nearest_labels = [labels[i] for i in indices[0]]
    return max(set(nearest_labels), key=nearest_labels.count)

#preprocessing Code (Run once offline to build index)
def build_embedding_index_from_dataset(dataset, save_index="faiss_index.bin", save_labels="labels.pkl"):
    all_embeddings = []
    all_labels = []
    for item in dataset:
        image = item["image"]
        categories = item["objects"]["category"]

        if not categories:
            continue  #skip if there are no detected categories

        #count frequency of each category and select the most common
        most_common_label = Counter(categories).most_common(1)[0][0]

        embedding = get_embedding(image)
        all_embeddings.append(embedding)
        all_labels.append(most_common_label)

    #save FAISS index
    embeddings = np.array(all_embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, save_index)

    #save corresponding labels
    with open(save_labels, 'wb') as f:
        pickle.dump(all_labels, f)

#build the index using the loaded dataset (run this once)
#build_embedding_index_from_dataset(ds["train"])

#load the index and labels for prediction
loaded_index, loaded_labels = load_index()

#example of making a prediction using an image from the dataset
'''test_image = ds["test"][67]["image"]
prediction = predict(test_image, loaded_index, loaded_labels)
print(f"Prediction for test image: {prediction}")
print(f"True label for test image: {ds['test'][0]['objects']['category']}")'''


def evaluate_model(dataset, index, labels, k=5):
    true_labels = []
    predicted_labels = []
    for item in dataset:

        image = item["image"]
        categories = item["objects"]["category"]

        if not categories:
            continue

        #use the same logic as in build_embedding_index to get the true label
        true_label = Counter(categories).most_common(1)[0][0]
        predicted_label = predict(image, index, labels, k)

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)



    return true_labels, predicted_labels

def predict_from_path(image_path, index, labels, k=5):
    classes = ['biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic']
    img = Image.open(image_path)
    prediction = predict(img, index, labels, k)
    return classes[prediction]

print(predict_from_path("C:\\Users\\saylo\\OneDrive\\Desktop\\Zyle CIT 1st year shool works\\3rd year\\Intelligent System 2\\ASLClassification\\archive (4)\\DATASET\\TEST\\O\\O_13867.jpg",loaded_index, loaded_labels))

#----For Evaluation -----#

#evaluate on the training set
'''train_true_labels, train_predicted_labels = evaluate_model(ds["train"], loaded_index, loaded_labels)
print("\n--- Training Data Evaluation ---")
train_accuracy = accuracy_score(train_true_labels, train_predicted_labels)
train_classification_report_str = classification_report(train_true_labels, train_predicted_labels, zero_division=0)
train_confusion_mat = confusion_matrix(train_true_labels, train_predicted_labels)

print(f"Accuracy on the training set: {train_accuracy:.4f}")
print("\nClassification Report on the training set:\n", train_classification_report_str)'''

#plot the confusion matrix for the training set
'''plt.figure(figsize=(10, 8))
sns.heatmap(train_confusion_mat, annot=True, fmt='d', cmap='Greens',
            xticklabels=np.unique(train_true_labels), yticklabels=np.unique(train_true_labels))
plt.xlabel('Predicted Label (Training)')
plt.ylabel('True Label (Training)')
plt.title('Confusion Matrix on the Training Set')
plt.show()'''

#evaluate on the test set
#combine validation and test datasets
'''combined_test_dataset = concatenate_datasets([ds["validation"], ds["test"]])'''

#evaluate on the combined dataset
'''test_true_labels, test_predicted_labels = evaluate_model(combined_test_dataset, loaded_index, loaded_labels)
print("\n--- Test Data Evaluation ---")
test_accuracy = accuracy_score(test_true_labels, test_predicted_labels)
test_classification_report_str = classification_report(test_true_labels, test_predicted_labels,zero_division=0)
test_confusion_mat = confusion_matrix(test_true_labels, test_predicted_labels)

print(f"Accuracy on the test set: {test_accuracy:.4f}")
print("\nClassification Report on the test set:\n", test_classification_report_str)'''

#plot the confusion matrix for the test set
'''plt.figure(figsize=(10, 8))
sns.heatmap(test_confusion_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(test_true_labels), yticklabels=np.unique(test_true_labels))
plt.xlabel('Predicted Label (Test)')
plt.ylabel('True Label (Test)')
plt.title('Confusion Matrix on the Test Set')
plt.show()'''