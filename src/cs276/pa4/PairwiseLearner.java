package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class PairwiseLearner extends Learner {
  private LibSVM model;
  private Standardize filter;
  public static final String SEPARATOR = "# #";
  public static final String C1 = "Class_1"; 
  public static final String C2 = "Class_2";
  public PairwiseLearner(boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
      throw new RuntimeException();
    }
    
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
  public PairwiseLearner(double C, double gamma, boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
      throw new RuntimeException();
    }
    
    model.setCost(C);
    model.setGamma(gamma); // only matter for RBF kernel
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
		/*
		 * @TODO: Your code here
		 */
		//Instances dataset = null;
		
		/* Build attributes list */
		FastVector<String> Classes = new FastVector<String>();
		Classes.add("Class_1");
		Classes.add("Class_2");
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score", Classes));
		TestFeatures unstandardizedDataset = new TestFeatures();
		unstandardizedDataset.features = new Instances("train_dataset", attributes, 0);
		unstandardizedDataset.indexMap = new HashMap<String, Map<String, Integer>>();
		
		Map<Query, List<Document>> trainQueryDocs = null;
		try{
		trainQueryDocs = Util.loadTrainData(train_data_file);
		}catch(Exception e){
			//System.out.println(e.getMessage());
			throw new RuntimeException();
			//return null;
		}
		
		Map<String, Map<String, Double>> trainQueryScores = null;
		try{
			trainQueryScores = Util.loadRelData(train_rel_file);
		}catch(Exception e){
			System.out.println(e.getMessage());
			throw new RuntimeException();
			//return null;
		}
		
		CosineSimilarityScorer instanceBuilder = new CosineSimilarityScorer(idfs);
		
		
		int count = 0;
		for(Query q: trainQueryDocs.keySet()){
			String queryString = q.toString();
			List<Document> trainDocs = trainQueryDocs.get(q);
			Map<String, Integer> queryMap = new HashMap<String, Integer>();
			Map<String, Double> trainScores = trainQueryScores.get(queryString);
			for(Document doc: trainDocs){
				Map<String, Double> atts = instanceBuilder.getvects(doc, q);
				
				double[] instance = {atts.get("url"), atts.get("title"), atts.get("body"), atts.get("header"), atts.get("anchor"), unstandardizedDataset.features.attribute("relevance_score").indexOfValue(C1)};
				Instance inst = new DenseInstance(1.0, instance); 
				unstandardizedDataset.features.add(inst);
				queryMap.put(doc.url, count);
				count+=1;
			}
			unstandardizedDataset.indexMap.put(q.toString(), queryMap);
		}
		
		
		/* Set last attribute as target */
		unstandardizedDataset.features.setClassIndex(unstandardizedDataset.features.numAttributes() - 1);
		
		this.filter = new Standardize();
		//System.out.println(this.filter);
		try{
			this.filter.setInputFormat(unstandardizedDataset.features);
			unstandardizedDataset.features = Filter.useFilter(unstandardizedDataset.features, this.filter);
		}catch(Exception e){
			System.out.println(e.getMessage());
			throw new RuntimeException();
		}
		
		Instances svmInstances = new Instances("train_dataset", attributes, 0); 
		for(Query q: trainQueryDocs.keySet()){
			String queryString = q.toString();
			List<Document> trainDocs = trainQueryDocs.get(q);
			Map<String, Double> trainScores = trainQueryScores.get(queryString);
			for(int i=0; i<trainDocs.size(); i++){
				Document doc1 = trainDocs.get(i);
				for(int j=i+1; j<trainDocs.size(); j++){
					int index1 = 0;
					int index2 = 0;
					Document doc2 = trainDocs.get(j);
					try{
					index1 = unstandardizedDataset.indexMap.get(q.toString()).get(doc1.url);
					}catch(Exception e){
						//System.out.println(q);
						//System.out.println(doc1.url);
						//System.out.println(unstandardizedDataset.index_map);
						throw new RuntimeException();
						//e.printStackTrace();
					//	return null;
					}
					try{
					index2 = unstandardizedDataset.indexMap.get(q.toString()).get(doc2.url);
					}catch(Exception e){
						e.printStackTrace();
						throw new RuntimeException();
						//return null;
					}
					double[] arr1 = unstandardizedDataset.features.get(index1).toDoubleArray();
					double[] arr2 = unstandardizedDataset.features.get(index2).toDoubleArray();
					double[] arr = new double[arr1.length];
					double[] arrneg = new double[arr1.length];
					//double p = Math.random();
					int k;
					for(k=0; k<Math.min(arr1.length, arr2.length)-1; k++){
						arr[k] = arr1[k] - arr2[k];
						arrneg[k] = arr2[k] - arr1[k];		
					}
					try{
						if(trainScores.get(doc1.url) > trainScores.get(doc2.url)){
							arr[k] = svmInstances.attribute("relevance_score").indexOfValue(C1);
							arrneg[k] = svmInstances.attribute("relevance_score").indexOfValue(C2);
						}
						else{
							arr[k] = svmInstances.attribute("relevance_score").indexOfValue(C2);
							arrneg[k] = svmInstances.attribute("relevance_score").indexOfValue(C1);
						}
						
				}catch(Exception e){
					System.out.println(e);
					System.out.println(svmInstances.attribute("relevance_score"));
					throw new RuntimeException();
				}
				
						//val = Math.signum(trainScores.get(doc2.url) - trainScores.get(doc1.url));
					//arr[arr.length-1] = val;
					Instance inst = new DenseInstance(1.0, arr); 
					Instance inst2 = new DenseInstance(1.0, arrneg);
					svmInstances.add(inst);
					svmInstances.add(inst2);
				}
			}
		}
		
		svmInstances.setClassIndex(svmInstances.numAttributes()-1);
		return svmInstances;
	}

	@Override
	public Classifier training(Instances dataset) {
		/*
		 * @TODO: Your code here
		 */
		try{
		this.model.buildClassifier(dataset);
		}catch(Exception e){
			System.out.println(e);
			throw new RuntimeException();
		}
		return this.model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		/*
		 * @TODO: Your code here
		 */
		//Instances dataset = null;
		
		/* Build attributes list */
		FastVector<String> Classes = new FastVector<String>();
		Classes.add("Class_1");
		Classes.add("Class_2");
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score", Classes));
		TestFeatures unstandardizedDataset = new TestFeatures();
		unstandardizedDataset.features = new Instances("train_dataset", attributes, 0);
		unstandardizedDataset.indexMap = new HashMap<String, Map<String, Integer>>();
		TestFeatures standDataset = new TestFeatures();
		
		Map<Query, List<Document>> testQueryDocs = null;
		try{
		testQueryDocs = Util.loadTrainData(test_data_file);
		}catch(Exception e){
			System.out.println(e.getMessage());
			throw new RuntimeException();
			//return null;
		}
		
		CosineSimilarityScorer instanceBuilder = new CosineSimilarityScorer(idfs);
				
		int count = 0;
		for(Query q: testQueryDocs.keySet()){
			String queryString = q.toString();
			List<Document> testDocs = testQueryDocs.get(q);
			Map<String, Integer> queryMap = new HashMap<String, Integer>();
			
			for(Document doc: testDocs){
				Map<String, Double> atts = instanceBuilder.getvects(doc, q);
				
				double[] instance = {atts.get("url"), atts.get("title"), atts.get("body"), atts.get("header"), atts.get("anchor"), unstandardizedDataset.features.attribute("relevance_score").indexOfValue(C1)};
				Instance inst = new DenseInstance(1.0, instance); 
				unstandardizedDataset.features.add(inst);
				queryMap.put(doc.url, count);
				count+=1;
			}
			unstandardizedDataset.indexMap.put(q.toString(), queryMap);
		}
		
		
		/* Set last attribute as target */
		unstandardizedDataset.features.setClassIndex(unstandardizedDataset.features.numAttributes() - 1);
		
		this.filter = new Standardize();
		try{
			this.filter.setInputFormat(unstandardizedDataset.features);
			unstandardizedDataset.features = Filter.useFilter(unstandardizedDataset.features, this.filter);
		}catch(Exception e){
			System.out.println(e.getMessage());
			//System.out.println(this.filter);
			throw new RuntimeException();
		}
		
		count = 0;
		standDataset.features = new Instances("train_dataset", attributes, 0); 
		standDataset.indexMap = new HashMap<String, Map<String, Integer>>();
		for(Query q: testQueryDocs.keySet()){
			String queryString = q.toString();
			List<Document> testDocs = testQueryDocs.get(q);
			//Map<String, Double> trainScores = trainQueryScores.get(queryString);
			Map<String, Integer> index = new HashMap<String, Integer>();
			for(int i=0; i<testDocs.size(); i++){
				Document doc1 = testDocs.get(i);
				for(int j=i+1; j<testDocs.size(); j++){
					Document doc2 = testDocs.get(j);
					int index1 = unstandardizedDataset.indexMap.get(q.toString()).get(doc1.url);
					int index2 = unstandardizedDataset.indexMap.get(q.toString()).get(doc2.url);
					double[] arr1 = unstandardizedDataset.features.get(index1).toDoubleArray();
					double[] arr2 = unstandardizedDataset.features.get(index2).toDoubleArray();
					double[] arr = new double[arr1.length];
					//double p = Math.random();
					double val = 0;
					String indexer;
					for(int k=0; k<Math.min(arr1.length, arr2.length)-1; k++){
						arr[k] = arr1[k] - arr2[k];
											
					}
					indexer = doc1.url + SEPARATOR + doc2.url;
					
					arr[arr.length-1] = unstandardizedDataset.features.attribute("relevance_score").indexOfValue(C1);
					Instance inst = new DenseInstance(1.0, arr); 
					standDataset.features.add(inst);
					index.put(indexer, count);
					count+=1;
				}
			}
			standDataset.indexMap.put(q.toString(), index);
			
		}
		
		standDataset.features.setClassIndex(standDataset.features.numAttributes()-1);
		return standDataset;
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		/*
		 * @TODO: Your code here
		 */
		Map<String, List<String>> rankedMap = new HashMap<String, List<String>>();
		final TestFeatures tf1 = tf;
		final Classifier model1 = model;
		for(String q: tf.indexMap.keySet()){
			final Map<String, Integer> index = tf.indexMap.get(q);
			Set<String> urlSet = new HashSet<String>();
			List<String> urlList;
			//List<String> sortedList = new ArrayList<String>();
			for(String docPair: index.keySet()){
				String[] docs = docPair.split(SEPARATOR);
				if(docs.length!=2){
					System.out.println("Fatal Error in doc split");
					return null;
				}
				urlSet.add(docs[0]);
				urlSet.add(docs[1]);						
			}
			urlList = new ArrayList<String>(urlSet);
			Comparator<String> A = new Comparator<String>() {

				@Override
				public int compare(String arg0, String arg1) {
					double clas = 0.0;
					if(index.containsKey(arg0 + SEPARATOR + arg1)){
						try{
						clas =  2*(-0.5+model1.classifyInstance(tf1.features.get(index.get(arg0 + SEPARATOR + arg1))));
						//System.out.println(model1.classifyInstance(tf1.features.get(index.get(arg0 + SEPARATOR + arg1))));
						}catch(Exception e){
							System.out.println("Classification error");
							throw new RuntimeException();
						}
					}
					else if(index.containsKey(arg1 + SEPARATOR + arg0)){
						try{
						clas =  -2*(-0.5+model1.classifyInstance(tf1.features.get(index.get(arg1 + SEPARATOR + arg0))));
						}catch(Exception e){
							System.out.println("Classification error");
							throw new RuntimeException();
						}
					}
					return (int)Math.signum(clas);
				}
				
			};
			Collections.sort(urlList, A );;
			rankedMap.put(q, urlList);
		}
		return rankedMap;
	}

}
