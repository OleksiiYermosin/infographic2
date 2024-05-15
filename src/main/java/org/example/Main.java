package org.example;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;

public class Main {
    public static final String inputFilePath = "src/main/resources/scores.csv";

    public static final String outputFilePath = "src/main/resources/output";

    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .appName("LinearRegression")
                .config("spark.master", "local")
                .getOrCreate();

        Dataset<Row> data = spark.read().option("header", "true").csv(inputFilePath);

        Dataset<Row> labeledData = data.selectExpr("cast(Hours as double)", "cast(Scores as double)");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"Hours"})
                .setOutputCol("features");

        Dataset<Row> assembledData = assembler.transform(labeledData)
                .select(col("features"), col("Scores").as("label"));

        Dataset<Row>[] randomedSplit = assembledData.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = randomedSplit[0];
        Dataset<Row> testData = randomedSplit[1];

        LinearRegression linearRegression = new LinearRegression();
        LinearRegressionModel model = linearRegression.fit(trainingData);

        double coefficient = model.coefficients().toArray()[0];
        double intercept = model.intercept();
        double r2 = model.summary().r2();

        Dataset<Row> predictions = model.transform(testData);

        predictions.select("features", "label", "prediction").show(Integer.MAX_VALUE, false);

        Dataset<Row> selectedFeatures = predictions.select("prediction", "label");

        double correlation = selectedFeatures.stat().corr("prediction", "label");

        System.out.println("Коефіцієнт детермінації моделі (R^2): " + r2);
        System.out.println("Коефіцієнт кореляції моделі: " + correlation);
        System.out.println("Коефіцієнт регресії: " + coefficient + " Інтерсепт: " + intercept);

        ChartBuilder builder = new ChartBuilder(labeledData, coefficient, intercept);
        builder.makeVisualisation(outputFilePath);

        spark.stop();
    }
}