# -*- coding: UTF-8 -*-

import functools
import hasher

import numpy as np
import scipy.spatial.distance as distance

from pyspark.mllib.linalg import SparseVector
from pyspark import RDD


class PyAdaLSHModel:
    """
    Wrapper class for LSH model
    """
    def __init__(self, k=10, mode="exp", inc=2, n_stages=5, start_budget=20, target_threshold=0.5):
        """
        Initialize the LSH model. Only one of the parameters target_threshold or
        n_bands is required.

        Parameters
        ----------
        k: int
            The number of top entities to filter.
        mode: str
            There are two modes:
            "exp": Exponential mode, where the budget increases of a multiplying factor inc every stage;
            "lin": Linear mode, where the budget increases of the same quantity inc every stage.
        inc: int
            The increment factor of the budget.
        n_stages: int
            The number of stages.
        start_budget : integer
            Total number of rows used to start the process.
        target_threshold: float
            Value of desired threshold if bands not specified.
        """
        self.k = k
        self.mode = mode
        self.inc = inc
        self.start_budget = start_budget 
        self.target_threshold = target_threshold
        self.bin_clusters = None
        self.stages = self.__tune_parameters(n_stages)      
        # RDD with all the minhash values computed
        self.signatures = None
        # RDD of the final_clusters
        self.final_clusters = None
        
        
    def __tune_parameters(self, n_stages):
        stages = []
        for n_stage in range(1, n_stages + 1):
            if n_stage == 1:
                budget = self.start_budget
            elif self.mode == "exp":
                budget = budget * self.inc
            else:
                budget = budget + self.inc
           
            for bands in xrange(1, budget / 2 + 1):
                if budget % bands == 0:
                    rows = budget / bands
                    threshold = (1.0 / bands) ** (1.0 / rows)
                    if (threshold < self.target_threshold):
                        break
            
            stage_dict = {"budget": budget, "bands": bands, "rows": rows, "threshold": threshold}
            stages.append(stage_dict)               
        return stages
    
    def __pop_max_cluster(self):
        """
        Return the largest cluster in the form <n_stage, list of vector idx>
        """
        for i, binc in reversed(list(enumerate(self.bin_clusters))):
            if binc:
                max_cluster = binc.reduce(lambda x, y: max(x, y, key=lambda item: len(item[1])))
                binc = binc.filter(lambda x: x[1].data != max_cluster[1].data)
                self.bin_clusters[i] = binc if binc.count() > 0 else None
                return max_cluster
        return None
    
    def __run_stage(self, zdata, n_stage, p, m, min_size=2):
        """
        Runs a single stage iteration on a cluster
        
        Parameters
        ----------
        zdata : RDD[(Vector, int)]
            RDD of data points. Acceptable vector types are numpy.ndarray,
            list or PySpark SparseVector. The second value is the element index.
        n_stage: int
            The number of the iteration/stage on the data of the cluster.
        m : integer
            Number of bins for hashing.
        min_size : integer
            Minimum allowable cluster size.
        """
        stage = self.stages[n_stage]
        computed_hashes = 0 if n_stage == 0 else self.stages[n_stage - 1]["budget"]
        n_hashes = stage["budget"] - computed_hashes
        
        # The new minhash functions are half of the total, the others are been already computed.
        seeds = np.vstack([np.random.random_integers(p, size=n_hashes), 
                           np.random.random_integers(0, p, size=n_hashes)]).T
        hashes = [functools.partial(hasher.minhash, a=s[0], b=s[1], p=p, m=m) for s in seeds]

        # Start by generating the signatures for each data point.
        # Output format is:
        # <(vector idx, row idx), minhash>
        sigs = zdata.flatMap(lambda x: [[(x[1], i + computed_hashes), h(x[0])] for i, h in enumerate(hashes)])
        if n_stage > 0:
            self.signatures = self.signatures.union(sigs).cache()
            # LSH only on the signatures of the records that belong to the current cluster
            data_idx = zdata.values().collect()
            sigs = self.signatures.filter(lambda x: x[0][0] in data_idx)
        else:
            self.signatures = sigs.cache()
            
        # Put together the vector minhashes in the same band.
        # Output format is:
        # <(band idx, hash minhash-list), vector idx>
        bands = sigs.map(lambda x: ((x[0][0], x[0][1] % stage["bands"]), (x[0][1], x[1]))) \
            .groupByKey().mapValues(sorted) \
            .map(lambda x: [(x[0][1], hash(tuple(x[1]))), x[0][0]]) \
            .groupByKey().cache()

        # Filter the bucket with size < min_clusters
        if min_size > 0:
            bands = bands.filter(lambda x: len(x[1]) >= min_size).cache()

        # Remaps each element to a cluster / bucket index.
        # Output format is:
        # <vector idx, bucket idx>
        vector_bucket = bands.map(lambda x: frozenset(sorted(x[1]))).distinct() \
            .zipWithIndex().flatMap(lambda x: map(lambda y: (np.long(y), x[1]), x[0])) \
            .cache()

        # Joins indices up with original data to provide clustering results.
        # Output format is:
        # <bucket idx, list of vectors>
        buckets = zdata.map(lambda x: (x[1], x[0])).join(vector_bucket) \
            .map(lambda x: (x[1][1], x[1][0])).groupByKey().cache()

        # Cluster generation
        buckets_count = buckets.mapValues(lambda idx: len(idx))  
        vector_bucket_count = vector_bucket.map(lambda x: (x[1], x[0])) \
            .join(buckets_count).map(lambda x: (x[1][0], (x[0], x[1][1])))
        vector_cluster = vector_bucket_count.groupByKey() \
            .mapValues(lambda b_count: max(b_count, key=lambda item: item[1])[0]) \

        return vector_cluster.map(lambda x: (x[1], x[0])).groupByKey() \
            .filter(lambda x: len(x[1]) >= min_size).cache()
                
    def run(self, data, p, min_size=2):
        """
        Starts the main LSH process.

        Parameters
        ----------
        data : RDD[Vector]
            RDD of data points. Acceptable vector types are numpy.ndarray,
            list or PySpark SparseVector.
        p : integer
            Prime number larger than the largest value in data.
        min_clusters : integer
            Minimum allowable cluster size.
        """
        # At the beginning all the records are in the same cluster
        zdata = data.zipWithIndex().cache()
        czdata = zdata.cache()
        
        # The bin-based structure is implement with an array of RDD. 
        # Each RDD contains elements of the form <stage-id, list of vector idx>.
        self.bin_clusters = np.empty(np.log2(data.count()).astype(int), dtype=RDD)
        
        n_stage = 0
        while True:
            print "Cluster size: {}. Stage: {}".format(czdata.count(), n_stage)
            clusters = self.__run_stage(czdata, n_stage, p, czdata.count(), min_size)
            print "--> Stage completed. %d clusters generated.\n" % clusters.count()
        
            # If the there is the last stage the clusters generated are added to the
            # collection of the final clusters
            if n_stage == len(self.stages) - 1:
                if self.final_clusters:
                    self.final_clusters = self.final_clusters.union(clusters.values()).cache()
                else: 
                    self.final_clusters = clusters.values().cache()
                topk_clusters = self.final_clusters.sortBy(lambda x: len(x), ascending=False).cache()
            
            # Else the clusters are added to bin-based structure
            else:    
                clogs = clusters.map(lambda x: (np.log2(len(x[1])).astype(int), x)).cache()
                for i in range(self.bin_clusters.size):
                    bins = clogs.filter(lambda x: x[0] == i).map(lambda x: (n_stage, x[1][1]))
                    if bins.count() > 0:
                        if self.bin_clusters[i]:
                            self.bin_clusters[i] = self.bin_clusters[i].union(bins).cache()
                        else:
                            self.bin_clusters[i] = bins.cache() 

            # Pop the largest cluster. 
            max_cluster = self.__pop_max_cluster()
            
            # Check the termination condition: if there are not clusters in the bin-based structure
            # or the k largest clusters in final_clusters are larger than the current max_cluster 
            # then the process finished.
            if not max_cluster:
                break
            elif self.final_clusters and self.final_clusters.count() >= self.k:
                ksize_final = min(map(lambda x: len(x), topk_clusters.take(self.k)))
                if ksize_final >= len(max_cluster[1]):
                    break
            
            n_stage = max_cluster[0] + 1
            czdata = zdata.filter(lambda x: x[1] in max_cluster[1].data).cache()
        
        # Return the set of vector indices of the largest k clusters
        return frozenset(reduce(lambda x,y: x + y, map(list, topk_clusters.take(self.k))))  
