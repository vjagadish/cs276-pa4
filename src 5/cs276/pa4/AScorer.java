package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class AScorer 
{
	public static final boolean querySublinear = false;
	
	Map<String,Double> idfs;
	String[] TFTYPES = {"url","title","body","header","anchor"};
	//String[] difTF = {"url","title","header","anchor"};
	
	public AScorer(Map<String,Double> idfs)
	{
		this.idfs = idfs;
	}
	
	//scores each document for each query
	public abstract Map<String, Double> getvects(Document d, Query q);
	
	//handle the query vector
	public Map<String,Double> getQueryFreqs(Query q)
	{
		Map<String,Double> tfQuery = new HashMap<String,Double>();
		
		/*
		 * @//TODO : Your code here
		 */
		for(String term: q.words){
			term = term.toLowerCase();
			if(tfQuery.containsKey(term)){
				if(idfs.containsKey(term))
					tfQuery.put(term, tfQuery.get(term) + 1/idfs.get(term));
				else
					tfQuery.put(term, tfQuery.get(term) + Math.log10(100000.0));
			}
			else{
				if(idfs.containsKey(term))
					tfQuery.put(term, 1/idfs.get(term));
				else
					tfQuery.put(term, Math.log10(100000.0));
			}
		}
		if(!querySublinear)
			return tfQuery;
		Map<String, Double> tfQuerySublinear = new HashMap<String, Double>();
		for(String term: q.words){
			term = term.toLowerCase();
			if(!tfQuerySublinear.containsKey(term)){
				tfQuerySublinear.put(term, 1 + Math.log10(tfQuery.get(term)));
			}
		}
		return tfQuerySublinear;
	}
	

	
	////////////////////Initialization/Parsing Methods/////////////////////
	
	/*
	 * @//TODO : Your code here
	 */
	
	//Parsing from a document to obtain string of words from url, header, title, anchor and body
	public static List<String> getListString(String combined){
		//split on everythng except # and alphabets.
		//coho-stanford
		
		List<String> temp = new ArrayList<String>();
		for(String t: combined.split("\\W+")){
			if(t.length()>0)
				temp.add(t.toLowerCase());
		}
		return temp;
	}
	
    ////////////////////////////////////////////////////////
	
	
	/*/
	 * Creates the various kinds of term frequences (url, title, body, header, and anchor)
	 * You can override this if you'd like, but it's likely that your concrete classes will share this implementation
	 */
	public Map<String,Map<String, Double>> getDocTermFreqs(Document d, Query q)
	{
		//map from tf type -> queryWord -> score
		Map<String,Map<String, Double>> tfs = new HashMap<String,Map<String, Double>>();
		
		////////////////////Initialization/////////////////////
		
		/*
		 * @//TODO : Your code here
		 */
		
		for(String tfType: TFTYPES){
			tfs.put(tfType, new HashMap<String, Double>());
		}
		
	    ////////////////////////////////////////////////////////
		
		//////////handle counts//////
		
					
		//Maintaining count for url
		Map<String, Double> temp = tfs.get("url");
		List<String> url = getListString(d.url);
		
		for (String queryWord : q.words){
			int count =0;
			for(String term: url){
				if(term.equals(queryWord))
					count +=1;
			}
			temp.put(queryWord, (double)count);
			
		}
		tfs.put("url", temp);
		
		//Maintaining count for title
		temp = tfs.get("title");
		List<String> title = getListString(d.title);
		
		for (String queryWord : q.words){
			int count =0;
			for(String term: title){
				if(term.equals(queryWord))
					count +=1;
			}
			temp.put(queryWord, (double)count);
			
		}
		tfs.put("title", temp);
		
		
		//Maintaining count for headers
		temp = tfs.get("header");
		List<String> header = new ArrayList<String>();
		boolean flag = true;
		if(d.headers == null)
			flag = false;
		if(flag){
			for(String head: d.headers){
				header.addAll(getListString(head));
			}
		}
		for (String queryWord : q.words){
			int count =0;
			for(String term: header){
				if(term.equals(queryWord))
					count +=1;
			}
			temp.put(queryWord, (double)count);
			
		}
		tfs.put("header", temp);
		
		
		//Maintaining count for anchor
		temp = tfs.get("anchor");
		
		for (String queryWord : q.words){
			temp.put(queryWord, 0.0);
		}
		flag = true;
		if(d.anchors == null)
			flag = false;
		if(flag){
			for(String anchor: d.anchors.keySet()){
				List<String> anch = getListString(anchor);
				for (String queryWord : q.words){
					
					double count =0;
					if(temp.containsKey(queryWord)){
						count = temp.get(queryWord);	
					}
							
					for(String term: anch){
						if(term.equals(queryWord))
							count += d.anchors.get(anchor);
					}
					temp.put(queryWord, count);
					
				}
				tfs.put("anchor", temp);
			}
		}
		//Maintaining count for Document body
		temp = tfs.get("body");
		flag = true;
		if(d.body_hits==null)
			flag = false;
		if(flag){
			for(String bodyTerm: d.body_hits.keySet()){
				temp.put(bodyTerm, (double)d.body_hits.get(bodyTerm).size());
			}
			for(String queryWord: q.words){
				if(!temp.containsKey(queryWord)){
					temp.put(queryWord, 0.0);
				}
			}
		}
		else{
			for(String queryWord: q.words){
					temp.put(queryWord, 0.0);
				
			}
		}
		tfs.put("body", temp);
				
		return tfs;
	}
	

}
