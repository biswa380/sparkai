package org.sdrc.sparkai.sparkai.service;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.image.ImageSchema;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;

//https://www.programcreek.com/java-api-examples/?class=org.apache.spark.sql.DataFrame&method=randomSplit

public class ImageClassifierDeepLearning {
	
//	Logger root = (Logger)LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
//	root.setLevel(Level.WARN);
	// Create Java spark context
	SparkConf conf = new SparkConf().setAppName("ANN Test")
			.setMaster("local[*]")
			.set("spark.driver.allowMultipleContexts", "true");
	JavaSparkContext sc = new JavaSparkContext(conf);
	
	String image_dir = "E:\\Image repo";
	Dataset<Row> cats_df = ImageSchema.readImages(image_dir+"\\cats").withColumn("label", functions.lit(1));
	Dataset<Row> dogs_df = ImageSchema.readImages(image_dir+"\\dogs").withColumn("label", functions.lit(0));
	
	Dataset<Row>[] _catsplit = cats_df.randomSplit(new double[] {0.9, 0.1});
	Dataset<Row> _cattrain = _catsplit[0];
	Dataset<Row> _cattest = _catsplit[1];
	
	Dataset<Row>[] _dogsplit = dogs_df.randomSplit(new double[] {0.9, 0.1});
	Dataset<Row> _dogtrain = _dogsplit[0];
	Dataset<Row> _dogtest = _dogsplit[1];
	
	Dataset<Row> _trainingset=_cattrain.unionAll(_dogtrain);
	Dataset<Row> _testingset = _cattest.unionAll(_dogtest);
	
	RandomForestRegressionModel regressionModel = new RandomForestRegressor().setFeaturesCol("features").fit(_trainingset);
	
}
