package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.management.RuntimeErrorException;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.TestInstances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class PairwiseLearner1 extends PointwiseLearner {
  private LibSVM model;
  public PairwiseLearner1(boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();throw new RuntimeException();
    }
    
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
  public PairwiseLearner1(double C, double gamma, boolean isLinearKernel){
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
  
  public static Instance getFeatureInstance (TestFeatures tf, String query, String url) {
	return  tf.features.get(tf.indexMap.get(query).get(url));
  }
  
  public static Instances standardizeInstance(Instances X) {
	  Standardize filter = new Standardize();
	  try {
		filter.setInputFormat(X);
		Instances stdX = Filter.useFilter(X, filter);
		return stdX;
	} catch (Exception e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
		throw new RuntimeException();
	}
	  
	  
  }
  
	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
		TestFeatures t = extractTrainFeatures(train_data_file,train_rel_file, idfs);
		Instances output = createInstancesTemplate();
		Instances stdX = standardizeInstance(t.features);
		t.features=stdX;

		for(String query : t.indexMap.keySet()) {
			Map<String, Integer> m = t.indexMap.get(query);
			Set<String> keySet = m.keySet();
			List<String> urls = new ArrayList<String>(keySet);
			//do pairwise 
			int ctr=0;
			for(int i=0;i<urls.size();i++){
				String url1 = urls.get(i);
				Instance i1 = getFeatureInstance(t, query, url1);
				for(int j=i+1;j<urls.size();j++){
					String url2 = urls.get(j);
					Instance i2 = getFeatureInstance(t, query, url2);
					addTrainingInstance (output,i1,i2);
				}
			}
			
		}
		return output;
	}


	@Override
	public Classifier training(Instances dataset) {
		try{
		this.model.buildClassifier(dataset);
		}catch(Exception e){
			e.printStackTrace();
			throw new RuntimeException();
		}
		return this.model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		TestFeatures t = extractTestFeatures(test_data_file, idfs);
		t.features=standardizeInstance(t.features);
		return t;
				}

	@Override
	public Map<String, List<String>> testing(final TestFeatures tf,
			final Classifier model) {
		Map<String,List<String>> answer= new HashMap<String,List<String>>();
		for(final String query : tf.indexMap.keySet()) {
			Map<String, Integer> map = tf.indexMap.get(query);
			List<String> urlList = new ArrayList<String>(map.keySet());
			
			Collections.sort(urlList, new Comparator<String>() {
	               
					@Override
					public int compare(String url1, String url2) {
						Instance i1 = getFeatureInstance(tf, query, url1);
						Instance i2 = getFeatureInstance(tf, query, url2);
						
						Instance diff = subtractInstances (i1,i2);
						
						Instances dataUnlabeled = createInstancesTemplate(); //new Instances("TestInstances", atts, 0);
						dataUnlabeled.add(diff);
						dataUnlabeled.setClassIndex(dataUnlabeled.numAttributes() - 1);        
						//double classif = ibk.classifyInstance(dataUnlabeled.firstInstance());

						try {
							double classs = model.classifyInstance(dataUnlabeled.firstInstance());
							if (classs==0)
								return -1;
							else 
								return 1;
						} catch (Exception e) {
							e.printStackTrace();
							throw new RuntimeException();
						}
						
					}
					
				});
			answer.put(query, urlList);
			
			
		}
		return answer;
	}	
	
	
	protected Instance subtractInstances(Instance i1, Instance i2) {
	double[] arr1=i1.toDoubleArray();
	double[] arr2=i2.toDoubleArray();
	double[] result = new double[arr1.length];
	for(int i=0;i<arr1.length;i++) {
		result[i]=arr1[i]-arr2[i];
	}
	Instance inst = new DenseInstance(1, result);
	
	return inst;
}

	private void addTrainingInstance(Instances output, Instance i1, Instance i2) {
		double[] arr1 = i1.toDoubleArray();
		double[] arr2 = i2.toDoubleArray();
		double rel1 = arr1[arr1.length-1];
		double rel2 = arr2[arr1.length-1];
		double[] result = new double [arr1.length];
		double[] resultneg = new double [arr1.length];
		int i=0;
		for(i=0;i<result.length-1;i++) {
		if(rel1>rel2) {
			result[i]=arr1[i]-arr2[i];
			resultneg[i]=arr2[i]-arr1[i];
		}
		else {
		    result[i]=arr2[i]-arr1[i];
		    resultneg[i]=arr1[i]-arr2[i];
		}
	}
		result[i]=0;
		resultneg[i]=1;
		Instance r = new DenseInstance(1.0, result);
		Instance rneg = new DenseInstance(1.0,resultneg);

		output.add(r);
		output.add(rneg);
		
	}
		

	
	public static Instances createInstancesTemplate () {
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		FastVector<String> myNomVals = new FastVector<String>();
		myNomVals.add("+1"); myNomVals.add("-1");
		attributes.add(new Attribute("class", myNomVals)); 
		
		Instances dataset = new Instances("train_dataset", attributes, 0);
		dataset.setClassIndex(5); 
		return dataset;
	}
	
	
	public static TestFeatures extractTrainFeatures(String train_data_file,
			String rel_file,
			Map<String, Double> idfs) 
	{
		TestFeatures t = new TestFeatures();
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("rel"));
		
		Instances X = new Instances("train_dataset", attributes, 0);

		CosineSimilarityScorer scorer = new CosineSimilarityScorer(idfs);
		
		Map<Query, List<Document>> trainingDocData = null;
		Map<String, Map<String, Double>> relData = null;
		try{
		trainingDocData = Util.loadTrainData(train_data_file);
		relData = Util.loadRelData(rel_file);
		}catch(Exception e){
			System.out.println(e.getMessage());
			throw new RuntimeException();
		}
		
		
        int	ctr=-1;
		for(Query q: trainingDocData.keySet()){
			String queryString = q.toString();
			List<Document> testDocs = trainingDocData.get(q);
			Map<String, Integer> index = new HashMap<String, Integer>(); 
			for(Document doc: testDocs){
				Map<String, Double> atts = scorer.getvects(doc, q);
				
				double[] instance = {atts.get("url"), atts.get("title"), atts.get("body"), atts.get("header"), atts.get("anchor"), relData.get(queryString).get(doc.url)};
				Instance inst = new DenseInstance(1.0, instance); 
				ctr++;
				X.add(inst);
				Map<String, Integer> map = t.indexMap.get(queryString);
				if(map==null) {
					map=new HashMap<String,Integer>();
					t.indexMap.put(queryString, map);
				}
				map.put(doc.url, ctr);
			}
		}
		t.features=X;
		return t;
		
	}
	
	
	public static TestFeatures extractTestFeatures(String test_data_file,
			Map<String, Double> idfs) 
	{
		TestFeatures t = new TestFeatures();
		Instances X = createInstancesTemplate ();
		CosineSimilarityScorer scorer = new CosineSimilarityScorer(idfs);
		
		Map<Query, List<Document>> testDocData = null;
		try{
		testDocData = Util.loadTrainData(test_data_file);
		}catch(Exception e){
			e.printStackTrace(); throw new RuntimeException();
		}
		
        int	ctr=-1;
		for(Query q: testDocData.keySet()){
			String queryString = q.toString();
			List<Document> testDocs = testDocData.get(q);
			for(Document doc: testDocs){
				Map<String, Double> atts = scorer.getvects(doc, q);
				
				double[] instance = {atts.get("url"), atts.get("title"), atts.get("body"), atts.get("header"), atts.get("anchor"), 0};
				Instance inst = new DenseInstance(1.0, instance); 
				ctr++;
				X.add(inst);
				Map<String, Integer> map = t.indexMap.get(queryString);
				if(map==null) {
					map=new HashMap<String,Integer>();
					t.indexMap.put(queryString, map);
				}
				map.put(doc.url, ctr);
			}
		}
		t.features=X;
		return t;
		
	}

	
	


}
