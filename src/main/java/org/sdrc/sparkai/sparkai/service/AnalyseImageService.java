package org.sdrc.sparkai.sparkai.service;

import java.io.Serializable;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import scala.Tuple2;
import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;

@Service
public class AnalyseImageService implements Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public void testModel(String trainingPath, String testImagePath, Integer iterations) {
		// TODO Auto-generated method stub

		
		Logger root = (Logger)LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
		root.setLevel(Level.WARN);
		// Create Java spark context
		SparkConf conf = new SparkConf().setAppName("SVM vs Navie Bayes")
				.setMaster("local[*]")
				.set("spark.driver.allowMultipleContexts", "true");
		JavaSparkContext sc = new JavaSparkContext(conf);

		// RDD training = MLUtils.loadLabeledData(sc, args[0]);
		// RDD test = MLUtils.loadLabeledData(sc, args[1]); // test set

		JavaRDD<LabeledPoint> training = sc.textFile(trainingPath).cache()
				.map(new Function<String, LabeledPoint>() {

					/**
					 * 
					 */
					private static final long serialVersionUID = 1L;

					@Override
					public LabeledPoint call(String v1) throws Exception {
						double label = Double.parseDouble(v1.substring(v1.indexOf(",")+1,
								v1.indexOf(",")+2));
						String featureString[] = v1
								.substring(v1.lastIndexOf(",") + 1).trim()
								.split(" ");
						double[] v = new double[featureString.length];
						int i = 0;
						for (String s : featureString) {
							if (s.trim().equals(""))
								continue;
							v[i++] = Double.parseDouble(s.trim());
						}
						return new LabeledPoint(label, Vectors.dense(v));
					}

				});
		System.out.println(training.count());
		JavaRDD<LabeledPoint> test = sc.textFile(testImagePath).cache()
				.map(new Function<String, LabeledPoint>() {

					/**
					 * 
					 */
					private static final long serialVersionUID = 1L;

					@Override
					public LabeledPoint call(String v1) throws Exception {
						double label = Double.parseDouble(v1.substring(v1.indexOf(",")+1,
								v1.indexOf(",")+2));
						String featureString[] = v1
								.substring(v1.lastIndexOf(",") + 1).trim()
								.split(" ");
						double[] v = new double[featureString.length];
						int i = 0;
						for (String s : featureString) {
							if (s.trim().equals(""))
								continue;
							v[i++] = Double.parseDouble(s.trim());
						}
						return new LabeledPoint(label, Vectors.dense(v));
					}

				});
		System.out.println(test.count());
		final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);

		JavaPairRDD<Double, Double> predictionAndLabel = test
				.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
					/**
					 * 
					 */
					private static final long serialVersionUID = 1L;

					@Override
					public Tuple2<Double, Double> call(LabeledPoint p) {
						return new Tuple2<Double, Double>(model.predict(p
								.features()), p.label());
					}
				});
		double accuracy = 1.0
				* predictionAndLabel.filter(
						new Function<Tuple2<Double, Double>, Boolean>() {
							/**
							 * 
							 */
							private static final long serialVersionUID = 1L;

							@Override
							public Boolean call(Tuple2<Double, Double> pl) {
								 System.out.println(pl._1() + " -- " +
								 pl._2());
								return pl._1().intValue() == pl._2().intValue();
							}
						}).count() / (double) test.count();
		System.out.println("navie bayes accuracy : " + accuracy);

		final SVMModel svmModel = SVMWithSGD.train(training.rdd(),iterations);

		JavaPairRDD<Double, Double> predictionAndLabelSVM = test
				.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
					/**
					 * 
					 */
					private static final long serialVersionUID = 1L;

					@Override
					public Tuple2<Double, Double> call(LabeledPoint p) {
						return new Tuple2<Double, Double>(svmModel.predict(p
								.features()), p.label());
					}
				});
		double accuracySVM = 1.0
				* predictionAndLabelSVM.filter(
						new Function<Tuple2<Double, Double>, Boolean>() {
							/**
							 * 
							 */
							private static final long serialVersionUID = 1L;

							@Override
							public Boolean call(Tuple2<Double, Double> pl) {
								 System.out.println(pl._1() + " -- " +
								 pl._2());
								return pl._1().intValue() == pl._2().intValue();
							}
						}).count() / (double) test.count();
		System.out.println("svm accuracy : " + accuracySVM);

		sc.close();
	}
}
