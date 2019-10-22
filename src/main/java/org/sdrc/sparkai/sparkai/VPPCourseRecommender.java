package org.sdrc.sparkai.sparkai;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.LoggerFactory;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import static org.apache.spark.sql.functions.*;

import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;

public class VPPCourseRecommender {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.setProperty("hadoop.home.dir", "C:\\Program Files\\Hadoop");
		Logger root = (Logger)LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
		root.setLevel(Level.WARN);
		
		SparkSession spark = SparkSession.builder()
				.appName("K-Means Clustering")
				.config("spark.warehouse.dir", "file:///c:/tmp/")
				.master("local[*]")
				.getOrCreate();
		
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("E:\\python-workspace\\datasets\\Starting workspaces\\Chapter 12\\VPPcourseViews.csv");
		
		csvData = csvData.withColumn("proportionWatched", col("proportionWatched").multiply(100));
		csvData.show();
//		csvData.groupBy("userId").pivot("courseId").sum("proportionWatched").show();
		
//		Dataset<Row>[] trainingAndHoldOutData = csvData.randomSplit(new double[] {0.8, 0.2});
//		Dataset<Row> trainingData = trainingAndHoldOutData[0];
//		Dataset<Row> holdOutData = trainingAndHoldOutData[1];
		
		ALS als=new ALS()
				.setMaxIter(10)
				.setRegParam(0.1)
				.setUserCol("userId")
				.setItemCol("courseId")
				.setRatingCol("proportionWatched");
		
		ALSModel alsModel = als.fit(csvData);
		alsModel.setColdStartStrategy("drop"); //to drop all those records whose future could not be predicted
//		Dataset<Row> prediction = alsModel.transform(holdOutData);
		Dataset<Row> prediction = alsModel.recommendForAllUsers(5);
		prediction.show();
		
		prediction.takeAsList(5).forEach(user -> {
			System.out.println("user : "+user.getAs(0)+" we recommend you to watch "+user.getAs(1).toString());
		});
	}

}
