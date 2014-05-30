package cs276.pa4;

import java.util.*;

import cs276.pa4.Query;
import cs276.pa4.Pair;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.functions.LinearRegression;

public class PointwiseLearner extends Learner {

	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
		
		/*
		 * @TODO: Below is a piece of sample code to show 
		 * you the basic approach to construct a Instances 
		 * object, replace with your implementation. 
		 */
		
		
		Instances dataset = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		dataset = new Instances("train_dataset", attributes, 0);
		
		/* Add data */
		/*double[] instance = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
		Instance inst = new DenseInstance(1.0, instance); 
		dataset.add(inst);*/
		
		//Code to generate feature vectors from the files and add them to the dataset
		Map<Query, List<Document>> trainQueryDocs = null;
		try{
		trainQueryDocs = Util.loadTrainData(train_data_file);
		}catch(Exception e){
			System.out.println(e.getMessage());
			return null;
		}
		
		Map<String, Map<String, Double>> trainQueryScores = null;
		try{
			trainQueryScores = Util.loadRelData(train_rel_file);
		}catch(Exception e){
			System.out.println(e.getMessage());
			return null;
		}
		
		CosineSimilarityScorer instanceBuilder = new CosineSimilarityScorer(idfs);
		
		for(Query q: trainQueryDocs.keySet()){
			String queryString = q.toString();
			List<Document> trainDocs = trainQueryDocs.get(q);
			Map<String, Double> trainScores = trainQueryScores.get(queryString);
			for(Document doc: trainDocs){
				Map<String, Double> atts = instanceBuilder.getvects(doc, q);
				
				double[] instance = {atts.get("url"), atts.get("title"), atts.get("body"), atts.get("header"), atts.get("anchor"), trainScores.get(doc.url)};
				Instance inst = new DenseInstance(1.0, instance); 
				dataset.add(inst);
			}
		}
		
		
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		
		
		return dataset;
	}

	@Override
	public Classifier training(Instances dataset) {
		/*
		 * @TODO: Your code here
		 */
		LinearRegression model = new LinearRegression();
		try{
		model.buildClassifier(dataset);
		}catch(Exception e){
			System.out.println(e.getMessage());
			return null;
		}
		return model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		/*
		 * @TODO: Your code here
		 */
		TestFeatures featStore = new TestFeatures();
		//Instances dataset = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		featStore.features = new Instances("train_dataset", attributes, 0);
		featStore.indexMap = new HashMap<String, Map<String, Integer>>();
		CosineSimilarityScorer instanceBuilder = new CosineSimilarityScorer(idfs);
		Integer count = 0;
		
		
		Map<Query, List<Document>> testQueryDocs = null;
		try{
		testQueryDocs = Util.loadTrainData(test_data_file);
		}catch(Exception e){
			System.out.println(e.getMessage());
			return null;
		}
		
		
		
		for(Query q: testQueryDocs.keySet()){
			String queryString = q.toString();
			List<Document> testDocs = testQueryDocs.get(q);
			Map<String, Integer> index = new HashMap<String, Integer>(); 
			for(Document doc: testDocs){
				Map<String, Double> atts = instanceBuilder.getvects(doc, q);
				
				double[] instance = {atts.get("url"), atts.get("title"), atts.get("body"), atts.get("header"), atts.get("anchor"), 0.0};
				Instance inst = new DenseInstance(1.0, instance); 
				featStore.features.add(inst);
				index.put(doc.url, count);
				count +=1;
			}
			featStore.indexMap.put(queryString, index);
		}
		
		
		return featStore;
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		/*
		 * @TODO: Your code here
		 */
		
		Map<String,List<String>> queryRankings = new HashMap<String,List<String>>();
		
		for (String query : tf.indexMap.keySet())
		{	//System.out.println(query);
			//loop through urls for query, getting scores
			List<Pair<String,Double>> urlAndScores = new ArrayList<Pair<String,Double>>(tf.indexMap.get(query).size());
			for (String url : tf.indexMap.get(query).keySet())
			{	
				//System.out.print(url+"  ");
				double score = 0.0;
				try{
					score = model.classifyInstance(tf.features.get(tf.indexMap.get(query).get(url)));
					//System.out.println(score);
				}catch(Exception e){
					System.out.println(e.getMessage());
				}
				urlAndScores.add(new Pair<String,Double>(url,score));
			}

			//sort urls for query based on scores
			Collections.sort(urlAndScores, new Comparator<Pair<String,Double>>() {
				@Override
				public int compare(Pair<String, Double> o1, Pair<String, Double> o2) 
				{
					
					if(o1.getSecond()>o2.getSecond())
						return -1;
					if(o1.getSecond()<o2.getSecond())
						return 1;
					return 0;
				}	
			});
			
			//put completed rankings into map
			List<String> curRankings = new ArrayList<String>();
			for (Pair<String,Double> urlAndScore : urlAndScores)
				curRankings.add(urlAndScore.getFirst());
			queryRankings.put(query, curRankings);
		}
		return queryRankings;
	}

}
