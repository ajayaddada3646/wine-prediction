package com.wineprediction;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;

public class ModelTrainer {
public static void main(String[] args) throws Exception {
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Trainer")
                .config("spark.master", "local[*]")
                .getOrCreate();

        Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .option("delimiter", ";")
                .csv("TrainingDataset.csv");

        String[] featureCols = new String[]{
                "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                "density", "pH", "sulphates", "alcohol"};

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("quality")
                .setFeaturesCol("features")
                .setMaxIter(100);

        Pipeline pipeline = new Pipeline().setStages(new org.apache.spark.ml.PipelineStage[]{assembler, lr});

        PipelineModel model = pipeline.fit(data);

        model.save(System.getProperty("user.dir") + "/wineQualityModel");

        spark.stop();
    }
}
