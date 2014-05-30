package cs276.pa4;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

public class TestFeatures {
	
	/* This is just a sample class to store the result */	
	
	/* Test features */
	Instances features;	
	
	/* Associate query-doc pair to its index within FEATURES instances
	 * {query -> {doc -> index}}
	 * 
	 * For example, you can get the feature for a pair of (query, url) using:
	 *   features.get(index_map.get(query).get(url));
	 * */
	Map<String, Map<String, Integer>> indexMap=new HashMap<String, Map<String,Integer>>();
}