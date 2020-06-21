package org.sdrc.sparkai.sparkai.controller;

import java.io.IOException;

import org.sdrc.sparkai.sparkai.service.MachineTrainingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class MachineTrainingController {
	
	@Autowired
	private MachineTrainingService trainingService;
	@PostMapping("trainmachine")
	public String trainMachine(@RequestParam("path") String path, 
			@RequestParam("label") Integer label) throws IOException{
		trainingService.trainMachine(path, label);
		System.out.println("completed");
		return "completed.";
	}
}
