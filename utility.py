import pandas as pd
import numpy as np
import librosa
import pickle
from lightgbm import Booster
# recom_model = pickle.load(open('recom-3/lgb-3-cluster.sav', 'rb'))
recom_model = Booster(model_file='recom-3/mode.txt')
# recom_scaler = pickle.load(open('recom-3/scaler-3-cluster-all-models.pkl', 'rb'))

emotion_model = pickle.load(open('mlp-3/MLP-3-sentiment-us.sav', 'rb'))
emotion_scaler = pickle.load(open('mlp-3/scaler-3-sentiment-us.pkl', 'rb'))

max_recommendations = 20

def extract_feature(file_name):
    print("\n\n")
    print("......................................................Extracting Features.........................................................")
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    result = np.array([])
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    result = np.hstack((result, mfccs))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    result = np.hstack((result, chroma))
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    result = np.hstack((result, mel))
    print("......................................................Features Extracted..........................................................")
    print("\n\n")
    return result

def recommendations(data,emotion):
    print("\n\n")
    print("......................................................Making Recommendations......................................................")
    if(emotion == 'positive'):
        emotion = 2
    elif (emotion == 'negative'):
        emotion = 0
    else:
        emotion = 1
    req_ids = ""
    data = pd.DataFrame(data).dropna()
    print("\n\nInitial Data :\n")
    print(data.head(2))
    ids = pd.DataFrame(data['id'],columns=["id"])
    print("\n\nID only:\n")
    print(ids.head(2))    
    data = data.drop('id',axis=1)
    print("\n\nData without ID:\n")
    print(data.head(2))    
    data = data.reindex(sorted(data.columns), axis=1)
    # col_features = ['danceability', 'energy', 'valence', 'loudness']
    # data_1 = pd.DataFrame(data[col_features],columns=col_features)
    # data = data.drop(col_features, axis = 1)
    # print("\n\nData without 4 features:\n")
    # print(data.head(2))
    # print("\n\n4 features data :\n")
    # print(data_1.head(2))
    # data_1 = pd.DataFrame(recom_scaler.fit_transform(data_1),columns=col_features)
    # print("\n\n4 features data scaled :\n")
    # print(data_1.head(2))    
    # for i in data_1.columns:
    #     data[i] = data_1[i].values
    pred = recom_model.predict(data)
    preds = []
    for i in pred:
      preds.append(np.where(i == np.amax(i))[0][0])
    print(preds)
    data['id'] = ids['id'].values
    data['cluster'] = preds
    print("\n\nData after all process :\n")
    print(data.head(2))
    if emotion == 0:
        data = data.sort_values(by=['energy'], ascending=False)
    else:
        data = data.sort_values(by=['energy'], ascending=True)
    data = data.loc[:, data.columns.intersection(['id','cluster'])]
    print("\n\nData after removing useless cols :\n")
    print(data.head(2))
    
    print("\n\nID and Cluster after sorting :\n")
    print(data.head(2))
    print("\n\n")
    print("Required Cluster ",emotion)
    print("Total clusters ",len(data))
    print("Total number 0 cluster ",(data["cluster"] == 0).sum())
    print("Total number 1 cluster ",(data["cluster"] == 1).sum())
    print("Total number 2 cluster ",(data["cluster"] == 2).sum())
    print("\n\n")
    data1 = data[data["cluster"] == 2]
    data = data[data["cluster"] == emotion]
    availableRecoms = (data["cluster"] == emotion).sum()
    if availableRecoms >= max_recommendations :
        recom = list(data.sample(n=max_recommendations)["id"])
    elif availableRecoms==0 :
        if (data["cluster"] == 2).sum() < max_recommendations:
            recom = list(data1.sample(n=(data["cluster"] == 2).sum())["id"])
        else:
            recom = list(data.sample(n=max_recommendations)["id"])
    else:
        recom = list(data.sample(n=availableRecoms)["id"])
    recom = list(set(recom))
    print("\n\nRecommended ids are : ",len(recom),"\n")
    print(*recom,sep="\n")
    print("\n\nRecommendation after joining :\n")
    req_ids = ",".join(recom)
    print(req_ids)
    print("......................................................Recommendations Found.......................................................")
    print("\n\n")
    return req_ids

def predict_emotion():
    print("\n\n")
    print("......................................................Predicting Emotion..........................................................")
    features_extracted = extract_feature("file-c.wav")
    features_extracted = features_extracted[np.newaxis,...]
    features_extracted = emotion_scaler.transform(features_extracted)
    predicted_emotion = emotion_model.predict(features_extracted)
    print("\n\nPredicted Emotion is :",predicted_emotion[0],"\n")
    print("......................................................Emotion Predicted...........................................................")
    print("\n\n")
    return predicted_emotion[0]