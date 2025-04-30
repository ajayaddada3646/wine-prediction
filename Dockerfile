FROM bitnami/spark:latest

WORKDIR /app

COPY target/wine-prediction-1.0-SNAPSHOT.jar wine-prediction.jar
COPY TrainingDataset.csv .
COPY ValidationDataset.csv .

CMD ["sh", "-c", "\
  echo '--- TRAINING ---' && \
  spark-submit --class com.wineprediction.ModelTrainer wine-prediction.jar && \
  echo '--- PREDICTION ---' && \
  spark-submit --class com.wineprediction.ModelPredictor wine-prediction.jar ValidationDataset.csv \
"]
