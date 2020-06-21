package org.sdrc.sparkai.sparkai.service;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;

import javax.imageio.ImageIO;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import scala.Tuple2;

@Service
public class ClassifyImageService implements Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	
	SparkConf conf = new SparkConf().setAppName("image classifier")
			.setMaster("local[*]")
			.set("spark.driver.allowMultipleContexts", "true");
	private transient JavaSparkContext sc = new JavaSparkContext(conf);
	
	public void classifySingleImage(String path, Integer label){
		int scaledWidth = 180;
		int scaledHeight = 200;
		
	}
	
	public static void toGray(BufferedImage image) {
		int width = image.getWidth();
		int height = image.getHeight();
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Color c = new Color(image.getRGB(j, i));
				int red = (int) (c.getRed() * 0.21);
				int green = (int) (c.getGreen() * 0.72);
				int blue = (int) (c.getBlue() * 0.07);
				int sum = red + green + blue;
				Color newColor = new Color(sum, sum, sum);
				image.setRGB(j, i, newColor.getRGB());
			}
		}
	}
	public static BufferedImage createResizedCopy(BufferedImage originalImage,
			int scaledWidth, int scaledHeight, boolean preserveAlpha) {
		System.out.println("resizing...");
		int imageType = preserveAlpha ? BufferedImage.TYPE_INT_RGB
				: BufferedImage.TYPE_INT_ARGB;
		BufferedImage scaledBI = new BufferedImage(scaledWidth, scaledHeight,
				imageType);
		Graphics2D g = scaledBI.createGraphics();
		if (preserveAlpha) {
			g.setComposite(AlphaComposite.Src);
		}
		g.drawImage(originalImage, 0, 0, scaledWidth, scaledHeight, null);
		g.dispose();
		return scaledBI;
	}
	JavaRDD<String> imageData;
	public void getFile(String path, Integer label) throws IOException{
		int scaledWidth = 180;
		int scaledHeight = 200;
		int extIndex = path.indexOf('.');
		String extension = path.substring(extIndex + 1);

		System.out.println(path + " extension " + extension);

		File input = new File(path.toString());
		BufferedImage image = createResizedCopy(ImageIO.read(input),
				scaledWidth, scaledHeight, Boolean.TRUE);
		toGray(image);
		File output = new File(path);
		ImageIO.write(image, extension, output);
		
		
		File file1 = new File(path);
		ArrayList<String> imageDataList=new ArrayList<String>();
		String sbf="";
		BufferedImage img = ImageIO.read(file1);
		Raster raster = img.getData();
		int w = raster.getWidth(), h = raster.getHeight();
		sbf=sbf+""+file1.getName();
		sbf=sbf+""+"," + label + ",";
		for (int x = 0; x < w; x++) {
			for (int y = 0; y < h; y++) {
				sbf=sbf+""+raster.getSample(x, y, 0) + " ";
			}
			sbf=sbf+""+" ";
		}
		sbf=sbf+"";
		System.out.println(sbf.split(",")[2]);
		imageDataList.add(sbf.toString());
		imageData=sc.parallelize(imageDataList);
		analyzeImage();
	}
	
	
	public void analyzeImage(){
	String trainingPath = "C:\\Users\\SDRC_DEV\\Pictures\\training";
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
	
	JavaRDD<LabeledPoint> test = imageData.map(new Function<String, LabeledPoint>() {

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
			System.out.println(featureString);
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
	}
}
