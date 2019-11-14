package org.sdrc.sparkai.sparkai;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import com.mongodb.spark.MongoSpark;

public class IndicatorRecommender {
public static void main(String[] args) {

	// TODO Auto-generated method stub

//	System.setProperty("hadoop.home.dir", "C:\\Program Files\\Hadoop");
//	Logger.getLogger("org.apache").setLevel(Level.ERROR);
	SparkConf config = new SparkConf().set("spark.mongodb.input.uri",
			"mongodb://localhost:27017/rani-prod.userIndicatorSearch");
	SparkSession spark = SparkSession.builder()
			.appName("Indicator Recommender")
			.config("spark.warehouse.dir", "file:///c:/tmp/")
			.master("local[*]")
			.config(config)
			.getOrCreate();
	JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
			jsc.setLogLevel("ERROR");
	Dataset<Row> csvData = MongoSpark.load(jsc).toDF();
	
	StringIndexer usernameIndexer = new StringIndexer();
	csvData = usernameIndexer.setInputCol("username")
		.setOutputCol("userId")
		.fit(csvData)
		.transform(csvData);
	
	
//	csvData.show();
	csvData.groupBy("username").pivot("indicatorId").sum("count").show();
	
	ALS als = new ALS()
			.setMaxIter(10)
			.setUserCol("userId")
			.setItemCol("indicatorId")
			.setRatingCol("count");
	
	als.setColdStartStrategy("drop");
	
	ALSModel model = als.fit(csvData);
	
//	This process will take the entire data on which it has been trained and will predict the gaps of each user
	Dataset<Row> userRecommendations = model.recommendForAllUsers(5);

	userRecommendations = userRecommendations.join(csvData).where(csvData.col("userId").equalTo(userRecommendations.col("userId")))
			.select("username", "recommendations").dropDuplicates();
//	userRecommendations.show();
	
	Dataset<Row> indicatorRecommendations = model.recommendForAllItems(5);
	indicatorRecommendations.takeAsList(5).forEach(indicator -> {
		System.out.println("Users who might see Indicator-"+indicator.getAs(0)+" are "+indicator.getAs(1).toString());
	});
	jsc.close();

}
}
