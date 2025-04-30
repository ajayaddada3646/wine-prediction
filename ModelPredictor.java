package com.wineprediction;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class ModelPredictor {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: ModelPredictor <TestDataset.csv>");
            System.exit(1);
        }

        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Predictor")
                .config("spark.master", "local[*]")
                .getOrCreate();

        Dataset<Row> validationData = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .option("quote", "\"")
                .option("escape", "\"")
                .option("delimiter", ";")
                .csv(args[0]);

        PipelineModel model = PipelineModel.load("wineQualityModel");

        Dataset<Row> predictions = model.transform(validationData);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1Score = evaluator.evaluate(predictions);

        System.out.println("F1 Score = " + f1Score);

        spark.stop();
    }
}
