package org.sdrc.sparkai.sparkai;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.slf4j.LoggerFactory;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;

public class KMeansHouseClustering {
public static void main(String[] args) {
	System.setProperty("hadoop.home.dir", "C:\\Program Files\\Hadoop");
	Logger root = (Logger)LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
	root.setLevel(Level.WARN);
	
	SparkSession spark = SparkSession.builder()
			.appName("K-Means Clustering")
			.config("spark.warehouse.dir", "file:///c:/tmp/")
			.master("local[*]")
			.getOrCreate();
	
	Dataset<Row> houseData = spark.read()
			.option("header", true)
			.option("inferSchema", true)
			.csv("E:\\python-workspace\\datasets\\kc_house_data.csv");
	houseData = houseData.withColumn("sqft_above_percentage", functions.col("sqft_above").divide(functions.col("sqft_living")));
//	houseData.show();
	
	VectorAssembler vectorAssembler = new VectorAssembler();
	Dataset<Row> inputData=vectorAssembler.setInputCols(new String[] {"bedrooms", "bathrooms", "sqft_living", "price"})
	.setOutputCol("features")
	.transform(houseData)
	.select("features");
	
	KMeans kmeans = new KMeans();
	for(int clusterCount=4;clusterCount<=9;clusterCount++) {
		kmeans.setK(clusterCount);
		System.out.println("clusterCount :: "+clusterCount);
		KMeansModel kMeansModel = kmeans.fit(inputData);
		Dataset<Row> predictions = kMeansModel.transform(inputData);
//		predictions.show();
		
		Vector[] clusterCentres = kMeansModel.clusterCenters();
			for(Vector v : clusterCentres) {
				System.out.println(v);
			}
		
		predictions.groupBy("prediction").count().show();
		System.out.println("SSE is " + kMeansModel.computeCost(inputData) );
		
		ClusteringEvaluator evaluator = new ClusteringEvaluator();
		System.out.println("Slihouette with squared euclidean distance is " + evaluator.evaluate(predictions));
		System.out.println("*******************************************************************");
	}
}
}
