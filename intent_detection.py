import sys
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer

class IntentDetector:

    def __init__(self):

        self.embedder = SentenceTransformer("distiluse-base-multilingual-cased-v1")
        self.reducer = PCA(n_components=32)
        self.classifier = SVC()


    def fit(self, X, y):
        
        embeddings = self.embedder.encode(X)
        embeddings_reduced = self.reducer.fit_transform(embeddings)
        self.classifier.fit(X=embeddings_reduced, y=y)


    def predict(self, X):

        embeddings = self.embedder.encode(X)
        embeddings_reduced = self.reducer.transform(embeddings)
        return self.classifier.predict(embeddings_reduced)


    def evaluate(self, y_true, y_pred):

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        return accuracy, precision, recall

if __name__ == '__main__':

    print("-----Training the model-----")

    df1 = pd.read_csv('intent-detection-train.csv')
    df2 = pd.DataFrame(data=[{"text": "Je ne retrouve pas mes valises", "label": "lost_luggage"},
                         {"text": "J'ai perdu mes valises", "label": "lost_luggage"},
                         {"text": "Est-ce que mon vol sera en retard", "label": "flight_status"},
                         {"text": "Est-ce risqué d'aller en Ukraine", "label": "travel_alert"}])
    df_train = pd.concat([df1, df2])

    intent_detector = IntentDetector()
    
    intent_detector.fit(df_train["text"].values, df_train["label"].values)
    
    print("-----Model trained-----")
    
    df_test = pd.read_csv(sys.argv[1])
    df_test["prediction"] = intent_detector.predict(df_test["text"].values)
    df_test.to_csv("intent-detection-results.csv", index=False)

    accuracy, precision, recall = intent_detector.evaluate(df_test["label"], df_test["prediction"])

    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
