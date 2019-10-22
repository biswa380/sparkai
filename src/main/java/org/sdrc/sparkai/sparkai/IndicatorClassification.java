package org.sdrc.sparkai.sparkai;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.LoggerFactory;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;

public class IndicatorClassification {

	public static void main(String[] args) {
		System.setProperty("hadoop.home.dir", "C:\\Program Files\\Hadoop");
		Logger root = (Logger)LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
		root.setLevel(Level.WARN);
		
		SparkSession spark = SparkSession.builder()
				.appName("Indicator Recommender")
				.config("spark.warehouse.dir", "file:///c:/tmp/")
				.master("local[*]")
				.getOrCreate();
		
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("E:\\python-workspace\\datasets\\user_indicator_view.csv");
		
		csvData.show();
		
		ALS als = new ALS()
				.setMaxIter(10)
				.setUserCol("userId")
				.setItemCol("indicatorId")
				.setRatingCol("count");
		
		als.setColdStartStrategy("drop");
		
		ALSModel model = als.fit(csvData);
		
		//This process will take another test dataset and will transform the test dataset over trained model
		Dataset<Row> test_data = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("E:\\python-workspace\\datasets\\user_indicator_test.csv");
		Dataset<Row> predictions = model.transform(test_data);
		predictions.show();
		RegressionEvaluator evaluator=new RegressionEvaluator()
				.setMetricName("rmse")
				.setLabelCol("count")
				.setPredictionCol("prediction");
		double rmse = evaluator.evaluate(predictions);
		System.out.println("Root mean square error for this model is :: "+rmse);
		
		
//		This process will take the entire data on which it has been trained and will fill the gaps of each user
		Dataset<Row> userRecommendations = model.recommendForAllUsers(5);
		userRecommendations.show();
		
		userRecommendations.takeAsList(5).forEach(u -> {
			System.out.println("User "+u.getAs(0)+", we suggest you to see "+u.getAs(1).toString()+" indicators");
		});
		
		Dataset<Row> indicatorRecommendations = model.recommendForAllItems(5);
		indicatorRecommendations.show();
		indicatorRecommendations.takeAsList(5).forEach(indicator -> {
			System.out.println("Users who might see Indicator-"+indicator.getAs(0)+" are "+indicator.getAs(1).toString());
		});
	}

}
