# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Accuracy:
```
security
             precision    recall  f1-score   support

          0       0.98      1.00      0.99      5152
          1       0.50      0.02      0.04        92

avg / total       0.97      0.98      0.97      5244

Accuracy: 0.9824561403508771
hospitals
             precision    recall  f1-score   support

          0       0.99      1.00      0.99      5188
          1       0.60      0.11      0.18        56

avg / total       0.99      0.99      0.99      5244

Accuracy: 0.9897025171624714
medical_help
             precision    recall  f1-score   support

          0       0.94      0.98      0.96      4816
          1       0.58      0.34      0.43       428

avg / total       0.91      0.93      0.92      5244

Accuracy: 0.9258199847444699
direct_report
             precision    recall  f1-score   support

          0       0.91      0.93      0.92      4296
          1       0.64      0.56      0.60       948

avg / total       0.86      0.86      0.86      5244

Accuracy: 0.86441647597254
shops
  'precision', 'predicted', average, warn_for)
             precision    recall  f1-score   support

          0       1.00      1.00      1.00      5222
          1       0.00      0.00      0.00        22

avg / total       0.99      1.00      0.99      5244

Accuracy: 0.9958047292143402
aid_related
             precision    recall  f1-score   support

          0       0.84      0.78      0.81      3143
          1       0.70      0.78      0.74      2101

avg / total       0.79      0.78      0.78      5244

Accuracy: 0.7807017543859649
request
             precision    recall  f1-score   support

          0       0.94      0.95      0.94      4398
          1       0.72      0.66      0.69       846

avg / total       0.90      0.90      0.90      5244

Accuracy: 0.9038901601830663
other_infrastructure
             precision    recall  f1-score   support

          0       0.96      1.00      0.98      5041
          1       0.26      0.03      0.06       203

avg / total       0.94      0.96      0.94      5244

Accuracy: 0.9588100686498856
storm
             precision    recall  f1-score   support

          0       0.97      0.97      0.97      4754
          1       0.72      0.66      0.69       490

avg / total       0.94      0.94      0.94      5244

Accuracy: 0.9439359267734554
search_and_rescue
             precision    recall  f1-score   support

          0       0.98      1.00      0.99      5092
          1       0.64      0.16      0.26       152

avg / total       0.97      0.97      0.97      5244

Accuracy: 0.9731121281464531
death
             precision    recall  f1-score   support

          0       0.98      0.99      0.98      5026
          1       0.66      0.46      0.55       218

avg / total       0.96      0.97      0.97      5244

Accuracy: 0.9679633867276888
transport
             precision    recall  f1-score   support

          0       0.97      0.99      0.98      5035
          1       0.61      0.24      0.35       209

avg / total       0.95      0.96      0.96      5244

Accuracy: 0.9635774218154081
earthquake
             precision    recall  f1-score   support

          0       0.98      0.99      0.98      4744
          1       0.87      0.77      0.82       500

avg / total       0.97      0.97      0.97      5244

Accuracy: 0.9677726926010679
weather_related
             precision    recall  f1-score   support

          0       0.92      0.92      0.92      3820
          1       0.78      0.78      0.78      1424

avg / total       0.88      0.88      0.88      5244

Accuracy: 0.881769641495042
medical_products
             precision    recall  f1-score   support

          0       0.97      0.99      0.98      4994
          1       0.63      0.38      0.47       250

avg / total       0.95      0.96      0.95      5244

Accuracy: 0.9597635392829901
missing_people
             precision    recall  f1-score   support

          0       0.99      1.00      1.00      5194
          1       0.57      0.08      0.14        50

avg / total       0.99      0.99      0.99      5244

Accuracy: 0.9906559877955758
fire
             precision    recall  f1-score   support

          0       0.99      1.00      1.00      5199
          1       0.60      0.27      0.37        45

avg / total       0.99      0.99      0.99      5244

Accuracy: 0.9921815408085431
other_weather
             precision    recall  f1-score   support

          0       0.96      0.99      0.98      4987
          1       0.53      0.18      0.27       257

avg / total       0.94      0.95      0.94      5244

Accuracy: 0.9521357742181541
infrastructure_related
             precision    recall  f1-score   support

          0       0.95      0.99      0.97      4932
          1       0.41      0.09      0.15       312

avg / total       0.91      0.94      0.92      5244

Accuracy: 0.9382151029748284
military
             precision    recall  f1-score   support

          0       0.98      0.99      0.98      5072
          1       0.58      0.38      0.46       172

avg / total       0.97      0.97      0.97      5244

Accuracy: 0.9706331045003814
food
             precision    recall  f1-score   support

          0       0.97      0.98      0.98      4659
          1       0.81      0.79      0.80       585

avg / total       0.96      0.96      0.96      5244

Accuracy: 0.956140350877193
aid_centers
             precision    recall  f1-score   support

          0       0.99      1.00      0.99      5189
          1       1.00      0.04      0.07        55

avg / total       0.99      0.99      0.99      5244

Accuracy: 0.9898932112890922
offer
             precision    recall  f1-score   support

          0       1.00      1.00      1.00      5226
          1       0.00      0.00      0.00        18

avg / total       0.99      1.00      0.99      5244

Accuracy: 0.9965675057208238
cold
             precision    recall  f1-score   support

          0       0.98      1.00      0.99      5129
          1       0.74      0.30      0.43       115

avg / total       0.98      0.98      0.98      5244

Accuracy: 0.9824561403508771
clothing
             precision    recall  f1-score   support

          0       0.99      1.00      1.00      5173
          1       0.69      0.51      0.59        71

avg / total       0.99      0.99      0.99      5244

Accuracy: 0.9902745995423341
related
             precision    recall  f1-score   support

          0       0.69      0.50      0.58      1246
          1       0.85      0.93      0.89      3953
          2       0.71      0.22      0.34        45

avg / total       0.81      0.82      0.81      5244

Accuracy: 0.8199847444698704
refugees
             precision    recall  f1-score   support

          0       0.97      0.99      0.98      5074
          1       0.56      0.23      0.33       170

avg / total       0.96      0.97      0.96      5244

Accuracy: 0.9691075514874142
money
             precision    recall  f1-score   support

          0       0.98      0.99      0.99      5134
          1       0.51      0.27      0.36       110

avg / total       0.97      0.98      0.98      5244

Accuracy: 0.9792143401983219
floods
             precision    recall  f1-score   support

          0       0.97      0.99      0.98      4850
          1       0.80      0.60      0.69       394

avg / total       0.96      0.96      0.96      5244

Accuracy: 0.9586193745232647
shelter
             precision    recall  f1-score   support

          0       0.97      0.98      0.97      4817
          1       0.72      0.63      0.67       427

avg / total       0.95      0.95      0.95      5244

Accuracy: 0.9496567505720824
buildings
             precision    recall  f1-score   support

          0       0.97      0.99      0.98      5026
          1       0.59      0.41      0.48       218

avg / total       0.96      0.96      0.96      5244

Accuracy: 0.9637681159420289
tools
             precision    recall  f1-score   support

          0       0.99      1.00      1.00      5211
          1       0.00      0.00      0.00        33

avg / total       0.99      0.99      0.99      5244

Accuracy: 0.9935163996948894
water
             precision    recall  f1-score   support

          0       0.98      0.98      0.98      4923
          1       0.73      0.73      0.73       321

avg / total       0.97      0.97      0.97      5244

Accuracy: 0.9668192219679634
electricity
             precision    recall  f1-score   support

          0       0.99      0.99      0.99      5153
          1       0.54      0.33      0.41        91

avg / total       0.98      0.98      0.98      5244

Accuracy: 0.9834096109839817
other_aid
             precision    recall  f1-score   support

          0       0.90      0.97      0.93      4588
          1       0.52      0.26      0.34       656

avg / total       0.85      0.88      0.86      5244

Accuracy: 0.8770022883295194

Overall Accuracy: 0.9508499509643676
```

### Web App:
Below is a screenshot from web app to give an overview of data.
![trainging](https://user-images.githubusercontent.com/18242446/83634289-8ef61b80-a5a2-11ea-8d09-66e600df7a5f.png)

Here you can put a message and hit classify_message button and table below will show the result.
![message](https://user-images.githubusercontent.com/18242446/83634556-075cdc80-a5a3-11ea-89f7-f494358178d5.png)
![result](https://user-images.githubusercontent.com/18242446/83634565-0a57cd00-a5a3-11ea-9169-cc5077901983.png)
