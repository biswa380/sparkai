package org.sdrc.sparkai.sparkai.controller;

import java.io.IOException;

import org.sdrc.sparkai.sparkai.service.AnalyseImageService;
import org.sdrc.sparkai.sparkai.service.ClassifyImageService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class SparkMllibController {
	
	@Autowired
	private AnalyseImageService analyseImageService;
	
	@Autowired
	private ClassifyImageService classifyImageService;
	
	@PostMapping("testModel")
	public String testModel(@RequestParam("trainingPath") String trainingPath, 
			@RequestParam("testImagePath") String testImagePath, @RequestParam("iterations") Integer iterations){
		System.out.println("check");
		analyseImageService.testModel(trainingPath, testImagePath, iterations);

		return "Analysis complete.";
	}
	
	@PostMapping("/check")
	public String check(@RequestParam("imagePath") String imagePath, @RequestParam("label") Integer label) 
			throws IOException
	{
		classifyImageService.getFile(imagePath, label);
		return "completed";
	}
}
