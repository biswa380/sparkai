package org.sdrc.sparkai.sparkai;

import static org.apache.spark.sql.functions.*;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.slf4j.LoggerFactory;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;

public class IndicatorClassification {

	public static void main(String[] args) {
		System.setProperty("hadoop.home.dir", "C:\\Program Files\\Hadoop");
		Logger root = (Logger)LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
		root.setLevel(Level.WARN);
		
		SparkSession spark = SparkSession.builder()
				.appName("Indicator Clustering")
				.config("spark.warehouse.dir", "file:///c:/tmp/")
				.master("local[*]")
				.getOrCreate();
		
		Dataset<Row> indicatorData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("E:\\rani-workspace\\indicator.csv");
		indicatorData.show();
		
		indicatorData = indicatorData.select("indicatorNid", "indicatorName", "unit", "subgroup", "formId", "numerator")
				.where(col("subgroup").isNotNull().and(col("unit").isNotNull()).and(col("unit").equalTo("number")).and(col("numerator").isNotNull()))
				.withColumn("formId", functions.when(col("formId").isNull(), 0).otherwise(col("formId")))
				.withColumn("indicatorName", lower(col("indicatorName")));
		
		Tokenizer indicatorTokenizer = new Tokenizer()
				.setInputCol("indicatorName")
				.setOutputCol("indicator_token");
		Dataset<Row> tokenizedData= indicatorTokenizer.transform(indicatorData);
		tokenizedData.show();
		
		String[] stopwords = new String[] {"number", "percent", "of", "the", "is", "are", "for", "with", "percentage", "conducted", 
				"where", "were", "which", "to", "their", "was", "sessions", "conducted", "there", "theme", "demonstration", "engagement", "meetings",
				"\\", "\""};
		StopWordsRemover indicatorFilter = new StopWordsRemover()
				.setStopWords(stopwords)
				.setInputCol("indicator_token")
				.setOutputCol("filtered_indicator");
		Dataset<Row> filteredData = indicatorFilter.transform(tokenizedData);
		filteredData.show();
		
		NGram ngramTransformer = new NGram()
				.setN(3)
				.setInputCol("filtered_indicator")
				.setOutputCol("indicator_ngrams");
		
		Dataset<Row> ngramData = ngramTransformer.transform(filteredData);
		ngramData.show();
		
		indicatorData = ngramData.selectExpr("indicatorNid", "indicatorName", "indicator_ngrams[0]","numerator");
		indicatorData.show();
		
		StringIndexer indicatorIndexer = new StringIndexer()
				.setInputCol("indicator_ngrams[0]")
				.setOutputCol("indicator_index").setHandleInvalid("skip");
		indicatorData = indicatorIndexer.fit(indicatorData)
				.transform(indicatorData);
		
		indicatorData = new StringIndexer()
				.setInputCol("numerator")
				.setOutputCol("numerator_index").setHandleInvalid("skip")
				.fit(indicatorData)
				.transform(indicatorData);
		
		indicatorData.show();
		System.out.println("after string indexing*********************************");
		
		OneHotEncoderEstimator encoder = new OneHotEncoderEstimator();
		indicatorData = encoder.setInputCols(new String[] {"indicator_index", "numerator_index"})
		.setOutputCols(new String[] {"indicator_vector", "numerator_vector"})
		.fit(indicatorData)
		.transform(indicatorData);
		
		VectorAssembler vectorAssembler = new VectorAssembler();
		Dataset<Row> inputData = vectorAssembler.setInputCols(new String[] {"indicator_vector", "numerator_vector"})
				.setOutputCol("features")
				.transform(indicatorData)
				.select("indicatorNid", "indicator_ngrams[0]", "indicatorName", "features");
		
		inputData.show();
		
		KMeans kmeans = new KMeans();
		for (int i = 210; i <= 210; i++) {
			kmeans.setK(i);
			System.out.println("k = "+i);
			KMeansModel kMeansModel = kmeans.fit(inputData);
			Dataset<Row> predictions = kMeansModel.transform(inputData);
//			predictions.show();
			predictions.select("indicatorName", "indicator_ngrams[0]", "prediction").write().format("com.databricks.spark.csv")
			.option("header", "true").save("E://indicator_cluster"+i);
//			predictions.groupBy("prediction").count().show();
			System.out.println("SSE is " + kMeansModel.computeCost(inputData));
			
			ClusteringEvaluator evaluator = new ClusteringEvaluator();
			System.out.println("Slihouette with squared euclidean distance is " + evaluator.evaluate(predictions));
			System.out.println("*******************************************************************");
		}
			
	}

}
