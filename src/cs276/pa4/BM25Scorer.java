package cs276.pa4;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;


public class BM25Scorer extends CosineSimilarityScorer
{
	Map<Query,Map<String, Document>> queryDict;
	
	public BM25Scorer(Map<String,Double> idfs,Map<Query,Map<String, Document>> queryDict)
	{
		super(idfs);
		this.queryDict = queryDict;
		
		this.BFs.put("url", burl);
		this.BFs.put("body", bbody);
		this.BFs.put("header", bheader);
		this.BFs.put("anchor", banchor);
		this.BFs.put("title", btitle);
		
		this.calcAverageLengths();
	}

	
	///////////////bm25 specific weights///////////////////////////
	double burl = 1;
    double btitle  = 1;
    double bbody = 0;
    double bheader = .5;
    double banchor = 0.5;
    
    ///////weights///////////////
    double urlweight=0.9;
    double titleweight= 1.5;
    double bodyweight=0.25;
    double headerweight=1;
    double anchorweight=0.9;

    double k1=1;
    double pageRankLambda=0.6;
    double pageRankLambdaPrime=0.5;
    //////////////////////////////////////////
    
    ////////////bm25 data structures--feel free to modify ////////
    
    Map<Document,Map<String,Double>> lengths;
    Map<String,Double> avgLengths;
    Map<Document,Double> pagerankScores;
    Map<String,Double> BFs=new HashMap<String,Double>();
    
	public double docCount;
    
	
	double aveBodyLen=0;
	double aveURLLen=0;
	double aveHeaderLen=0;
	double aveTitleLen=0;
	double aveAnchorLen=0;
	
	
    //////////////////////////////////////////
	
	public static double getW(String candidate, Query query) {
		List<String> words = query.words;
		List<String> candidateTerms = AScorer.getListString(candidate);
		Map<String, List<Integer>> map = new HashMap<String,List<Integer>>();
		Integer ctr=0;

		for(String candidateTerm : candidateTerms){
			List<Integer> list = map.get(candidateTerm);
			if(list==null){
				list=new ArrayList<Integer>();
				map.put(candidateTerm,list);
			}
			list.add(ctr);
			ctr++;
		}
		//System.out.println(map);
		return getSmallestWindow(query,map);
	}
	
	
	
	public static double getSmallestWindow(Query q, Map<String, List<Integer>> body_hits){
		if(body_hits==null){
		//	System.out.println("BANG");
			return Double.POSITIVE_INFINITY;
		}
		
		List<List<Integer>> posList = new ArrayList<List<Integer>>();
		List<Integer> indexList = new ArrayList<Integer>();
		double minWindow = Double.POSITIVE_INFINITY;
		for(String term: q.words){
			if(!body_hits.containsKey(term))
				return minWindow;
			posList.add(body_hits.get(term));
			indexList.add(0);
		}
		int numList = posList.size();
		boolean flag = true;
		double smallestPosition = Double.POSITIVE_INFINITY;
		double largestPosition = 0;
		int smallestIndex = -1;
		int largestIndex = -1;
		while(flag){
			for(int i=0; i<numList; i++){
				if(indexList.get(i) >= posList.get(i).size())
					return minWindow;
				Integer temp = posList.get(i).get(indexList.get(i));
				if(temp < smallestPosition){
					smallestPosition = temp;
					smallestIndex = i;
				}
				if(temp > largestPosition){
					largestPosition = temp;
					largestIndex = i;
				}
			}
			if(largestPosition - smallestPosition + 1 < minWindow)
				minWindow =largestPosition - smallestPosition + 1;
			if(indexList.get(smallestIndex)+1 >= posList.get(smallestIndex).size())
				flag = false;
			else{
				indexList.set(smallestIndex, indexList.get(smallestIndex)+1);
			}
			smallestPosition = Double.POSITIVE_INFINITY;
		}
		
		return minWindow;
	}
	
	public double getMin(Collection<Double> list){
		double minData= Double.MAX_VALUE;
		
		for(Double data : list){
			if(data<minData){
				minData=data;
			}
		}
		return minData;
	}
	public static double getWCollection(Collection<String> list, Query q) {
		double minWindow = Double.POSITIVE_INFINITY;
		if(list==null)
			return minWindow;
		for(String candidate : list )
		{
			double window = getW(candidate,q);
			if(window<minWindow){
				minWindow=window;
			}
		}	
			
		return minWindow;
	}
	

	public double getMinWindow(Document d, Query q) {
		
		double minWindow = Double.MAX_VALUE;
		double window;
		Map<String, Integer> anchors = d.anchors;
		List<Double> windowLengths = new ArrayList<Double>();
		if(anchors!=null){
			window = getWCollection(anchors.keySet(), q);
			windowLengths.add(window);
		}
		if(d.body_hits!=null){
			window = getWCollection(d.body_hits.keySet(), q);
			windowLengths.add(window);
		}
		
		windowLengths.add(getW(d.url, q));
		windowLengths.add(getW(d.title,q));
		windowLengths.add(getWCollection(d.headers, q));
		
		minWindow= Math.min(minWindow,getMin(windowLengths));
		return minWindow;
	}

	
    //sets up average lengths for bm25, also handles pagerank
    public void calcAverageLengths()
    {
    	lengths = new HashMap<Document,Map<String,Double>>();
    	avgLengths = new HashMap<String,Double>(); 
    	pagerankScores = new HashMap<Document,Double>();
    	
		/*
		 * @//TODO : Your code here
		 */
    	
    	//normalize avgLengths
			
			Set<Query> queries = this.queryDict.keySet();
			for(Query query : queries){
				Map<String, Document> map = queryDict.get(query);
				Set<String> urls = map.keySet();
				for(String url : urls){
					Document doc = map.get(url);
				//	System.out.println("processing document " + doc);
					pagerankScores.put(doc,(double)doc.page_rank);
					double bodyLen=doc.body_length;
					double urlLen = getURLLen(doc.url);
					double hdrLen = getHeaderLen(doc.headers);
					double titleLen = getTitleLen(doc.title);
					double anchorLen = getAnchorLen(doc.anchors);
					
					Map<String, Double> lenMap = lengths.get(doc);
					if(lenMap==null){
						lenMap=new HashMap<String,Double>();
						lengths.put(doc, lenMap);
					}
					lenMap.put("body", bodyLen);
					lenMap.put("url", urlLen);
					lenMap.put("header", hdrLen);
					lenMap.put("title", titleLen);
					lenMap.put("anchor", anchorLen);
					
					
					
					aveBodyLen+=bodyLen;
					aveURLLen+=urlLen;
					aveHeaderLen+=hdrLen;
					aveTitleLen+=titleLen;
					aveAnchorLen+=anchorLen;
					
					
					docCount++;
					
				}
			
			/*
			 * @//TODO : Your code here
			 */
		}
			
			aveBodyLen/=docCount;
			aveURLLen/=docCount;
			aveHeaderLen/=docCount;
			aveTitleLen/=docCount;
			aveAnchorLen/=docCount;
			
			avgLengths.put("body", aveBodyLen);
			avgLengths.put("url",aveURLLen);
			avgLengths.put("header", aveHeaderLen);
			avgLengths.put("title",aveTitleLen);
			avgLengths.put("anchor", aveAnchorLen);

			
		//	System.out.println(avgLengths);
			

    }
    
    private double getAnchorLen(Map<String, Integer> anchors) {
    	if(anchors==null)
    		return 0;
		Set<String> anchorSet = anchors.keySet();
		double len=0;
		for(String anchor : anchorSet){
			List<String> terms = getListString(anchor);
		//	System.out.println("Anchor " + anchor + " " + terms);
			len+=terms.size();
		}
	//	System.out.println(" len is " + len + " anchorset sz " + anchorSet.size() +" " + len/(double)anchorSet.size());
		return len/(double)anchorSet.size();
//		return 0;
	}

	private double getTitleLen(String title) {
		List<String> list = getListString(title);
	//	System.out.println(title + " TITLE " + list);
		return list.size();

	}

	private double getURLLen(String url) {
		List<String> list = getListString(url);
	//	System.out.println(url + " URL " + list);
		return list.size();
	}

	private double getHeaderLen(List<String> headers) {
		double len=0;
		if(headers==null)
			return 0;
		for(String header : headers ){
			List<String> terms = getListString(header);
	//		System.out.println(header + " HDR " + terms);
			len+=terms.size();
		}
		// TODO Auto-generated method stub
	//	System.out.println(headers + " len is " + len +" hdr sz " + len/(double)headers.size() );
		return len/(double)headers.size();
	}

	////////////////////////////////////
    
    
	public double getNetScore(Map<String,Map<String, Double>> tfs, Query q, Map<String,Double> tfQuery,Document d)
	{
		double score = 0.0;
		
		Map<String, Double> mapUrl = tfs.get("url");
		Map<String, Double> mapTitle = tfs.get("title");
		Map<String, Double> mapAnchor = tfs.get("anchor");
		Map<String, Double> mapBody = tfs.get("body");
		Map<String, Double> mapHeader = tfs.get("header");
		
		
		Map<String,Double> term_wt = new HashMap<String,Double>();
		
		for(String term : q.words){
			Double weight = burl*mapUrl.get(term) + banchor*mapAnchor.get(term) + bbody*mapAnchor.get(term)+bheader*mapHeader.get(term) + btitle*mapTitle.get(term);
			term_wt.put(term, weight);
			double idf=5.0;
			
			if(this.idfs.containsKey(term)){
				idf=idfs.get(term);
			}
			
			weight= idf*(weight/(this.k1+weight));
			score+=weight;
		}
		double pageRank = this.pagerankScores.get(d);
		score+=(this.pageRankLambda/(1 + Math.exp(-pageRankLambdaPrime*pageRank)));
		//score+=(this.pageRankLambda*Math.log(pageRankLambdaPrime*pageRank));
		
		return score;
		
		
		}

	//do bm25 normalization
	/* query : gold course phone number and Document d
	 * {anchor={number=0.0, golf=0.0, phone=0.0, course=0.0}, header={number=0.0, golf=0.0, phone=0.0, course=0.0}, title={number=0.0, golf=0.0, phone=0.0, course=0.0}
	 */
	public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q)
	{
	//	System.out.println("PROCESSING11 document " + d + " query " + q);
		
        for(String tfType : this.TFTYPES){
        	Map<String, Double> type = tfs.get(tfType);
        	for(String term : type.keySet()){
        		double unnormalizedTF = type.get(term);
        	//	System.out.println("unnormalized tf  " + unnormalizedTF + " in term  " + tfType +" " + term);
        		
        		
        		if(unnormalizedTF!=0) {
        		Map<String, Double> map = lengths.get(d);
        		if(map==null){
        		}
        		double len_d_f=0;
        	//	System.out.println(map + " " + term);
        		len_d_f=map.get(tfType);
        		
        		len_d_f/=avgLengths.get(tfType);
        		double factor = (len_d_f-1.0)*this.BFs.get(tfType);
        		factor+=1.0;
        		
        		double normalizedTF = unnormalizedTF/factor;
        		type.put( term,normalizedTF );
        		}
        		
        		else{
        			type.put(term,0.0);
        		}
        		
        		
        	}
        }		
		
		/*
		 * @//TODO : Your code here
		 */
	}
	
	public Map<String, Double> getvects(Document d, Query q) 
	{
		Map<String, Double> m = super.getvects(d, q);
		m.put("body", 0.);
		
		m.put("bm25score", getSimScore(d, q));
		m.put("pagerank", this.pagerankScores.get(d));
		m.put("window",	getMinWindow(d, q));

		return m;
		
	}

	
	public double getSimScore(Document d, Query q) 
	{
		
		Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
		
		this.normalizeTFs(tfs, d, q);
		
		Map<String,Double> tfQuery = getQueryFreqs(q);
		
		
        return getNetScore(tfs,q,tfQuery,d);
	}

	public double getSimScore1(Document d, Query q) {
		double B = 1.35;    	    
	    double boostmod = .3;
	    
		Map<String, Integer> anchors = d.anchors;
		List<String> headers = d.headers;
		
		double score = getSimScore(d, q);
		Map<String, List<Integer>> body_hits = d.body_hits;
		if(body_hits==null){
		//	System.out.println("****");
		//	System.out.println(d);
		//	System.out.println("****");
			
		//	System.out.println(q);
		}
		//double window=getSmallestWindow(q, body_hits);
		double window = getMinWindow(d, q);
		
		double queryLen = q.words.size();
		double ff=(window-queryLen)/queryLen;
		//double factor = 1+(B-1)/(ff + 1);
		double factor = 1 + (B-1)*Math.exp(-boostmod*ff);
	//	System.out.println("factor is " + factor + " window size " + window + " q len " + queryLen);
		return score*factor;
//		return 0;
	}

	
	
}
