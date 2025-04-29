import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN

class StudentGrouping:
    def __init__(self, df: pd.DataFrame):
        self.df = df.drop(columns=['Student_ID'] if 'Student_ID' in df.columns else [])
        self.graph = nx.Graph()

    def buildSimilarityGraph(self, threshold=0.8):
        similarities = cosine_similarity(self.df)
        numberofStudents = len(self.df)

        for i in range(numberofStudents):
            self.graph.add_node(i)

        for i in range(numberofStudents):
            for j in range(i + 1, numberofStudents):
                similarityScore = similarities[i][j]
                if similarityScore >= threshold:
                    self.graph.add_edge(i, j, weight=similarityScore)

        return self.graph

    def clusterStudents(self, algorithm='agglomerative', **kwargs):
        """
        Clusters students based on the specified algorithm.
        Supported algorithms: 'Agglomerative', 'KMeans', 'DBSCAN'
        
        Suggested parameters for each algorithm:
        - Agglomerative: n_clusters, affinity, linkage
        - KMeans: n_clusters, init, n_init, max_iter, random_state
        - DBSCAN: eps, min_samples, metric, algorithm
        """
        if algorithm.lower() == 'agglomerative':
            clustering = AgglomerativeClustering(**kwargs)
        elif algorithm.lower() == 'kmeans':
            clustering = KMeans(**kwargs)
        elif algorithm.lower() == 'dbscan':
            clustering = DBSCAN(**kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        labels = clustering.fit_predict(self.df)
        return labels

    def getStudentGroups(self, labels):
        grouped = {}
        for index, label in enumerate(labels):
            grouped.setdefault(label, []).append(index)
        return grouped