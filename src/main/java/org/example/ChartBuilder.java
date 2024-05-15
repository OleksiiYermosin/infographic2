package org.example;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.awt.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ChartBuilder {
    private Dataset<Row> data;
    private double coefficient;
    private double intercept;

    public ChartBuilder(Dataset<Row> data, double coefficient, double intercept) {
        this.data = data;
        this.coefficient = coefficient;
        this.intercept = intercept;
    }

    public void makeVisualisation(String outputPath) {
        List<Double> hours = data.select("Hours").as(Encoders.DOUBLE()).collectAsList();
        List<Double> scores = data.select("Scores").as(Encoders.DOUBLE()).collectAsList();

        List<Double> trend = new ArrayList<>();
        for (Double x : hours) {
            double Y = coefficient * x + intercept;
            trend.add(Y);
        }

        XYChart diagram = new XYChart(800, 600);
        diagram.setTitle("Залежність оцінки від часу навчання");
        diagram.setXAxisTitle("Час, годин на добу");
        diagram.setYAxisTitle("Оцінка, балів");

        diagram.addSeries("Залежність оцінки від часу навчання\n", hours, scores)
                .setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter)
                .setMarkerColor(Color.ORANGE)
                .setLineStyle(SeriesLines.NONE);

        XYSeries newSeries = diagram.addSeries(String.format("Лінія тренду: Y = %.2f * x + %.2f", coefficient, intercept), hours, trend);
        newSeries.setMarker(SeriesMarkers.NONE);
        newSeries.setLineColor(Color.RED);

        try {
            BitmapEncoder.saveBitmap(diagram, outputPath + ".png", BitmapEncoder.BitmapFormat.PNG);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
