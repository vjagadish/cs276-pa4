package cs276.pa4;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class CosineSimilarityScorer extends AScorer
{
	public CosineSimilarityScorer(Map<String,Double> idfs)
	{
		super(idfs);
	}
	
	///////////////weights///////////////////////////
       
    double smoothingBodyLength = 0;
    //////////////////////////////////////////
    
    public static final boolean docSublinear = true;
	
	public Map<String,Double> vectorize(Map<String,Map<String, Double>> tfs, Query q, Map<String,Double> tfQuery,Document d)
	{
		//double score = 0.0;
		Map<String, Double> vect = new HashMap<String, Double>();
		//System.out.println("boo****************************************************************");
		/*
		 * @//TODO : Your code here
		 */
		//System.out.println(q.words);
		//System.out.println(d);
		for(String tftype: TFTYPES){
			Map<String, Double> mapper = tfs.get(tftype);
			double interScore = 0.0;
			for(String queryWord: q.words){
				
				if(mapper.containsKey(queryWord)){
					//System.out.println(mapper.get(queryWord));
					if(tfQuery.containsKey(queryWord)){
						//System.out.println(tfQuery.get(queryWord));
						interScore += mapper.get(queryWord)*tfQuery.get(queryWord);
						
					}
				} 
			}
			vect.put(tftype, interScore);
			
		}
		//System.out.println(score);
		
		return vect;
	}

	
	
	
	public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q)
	{
		/*
		 * @//TODO : Your code here
		 */
		
		for(String tftype: TFTYPES){
			Map<String, Double> mapper = tfs.get(tftype);
			for(String queryWord: q.words){
				if(!mapper.containsKey(queryWord))
					mapper.put(queryWord, 0.0);
				else{
					if(!docSublinear){
						mapper.put(queryWord, mapper.get(queryWord)/(d.body_length + smoothingBodyLength));
					}
					else{
						double temp = mapper.get(queryWord);
						if(temp!=0)
							mapper.put(queryWord, (1 + Math.log10(mapper.get(queryWord)))/(1 + Math.log10(d.body_length + smoothingBodyLength)));
					}
				}
			}
			tfs.put(tftype, mapper);
		}
	}

	
	public Map<String, Double> getvects(Document d, Query q) 
	{
		
		Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
		
		this.normalizeTFs(tfs, d, q);
		
		Map<String,Double> tfQuery = getQueryFreqs(q);
		
		
        return vectorize(tfs,q,tfQuery,d);
	}
	
	
	
	
}
