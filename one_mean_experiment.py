from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import distance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from planar import BoundingBox
import numpy as np
import copy
import math




class OneMeanExperiment:
    def __init__(self, *args,**kwargs):
        self.data = args
        self.r_data = kwargs
        self.X = args[0] if args else None 
        self.first_x = None
        self.vor = None
        self.kmeans = None
        self.furthest_point = None
        self.bbox = None
        self.poison_count_arr = np.array([])
        self.scores = np.array([])
        self.max_dist = float('-inf')
        self.size_to_poision_proportion = None
        self.max_location = np.array([float('-inf'),float('-inf')], dtype=object)
        self.is_random = True if 'random' in self.r_data.keys() else False
        self.__compute()


    def __compute(self):
        if self.is_random:
            if not ('low' in self.r_data.keys() and 'high' in self.r_data.keys() and 'size' in self.r_data.keys()):
                print("Pass low, high, and size as keyword arguments")
                exit()

            """ set bounding box """
            if not ('p1' in self.r_data.keys() and 'p2' in self.r_data.keys()):
                print("Need to include two points p1, p2 for bounding box")
                exit()

            self.bbox = BoundingBox([self.r_data['p1'],self.r_data['p2']])

            """ set random data """
            self.X = np.random.uniform(low=self.r_data['low'], high=self.r_data['high'], size=self.r_data['size'])
            self.first_x = copy.deepcopy(self.X)
            
        else:
            self.first_x = copy.deepcopy(self.X)
            min_point = np.array([float('inf'),float('inf')], dtype=object)
            max_point = np.array([float('-inf'),float('-inf')], dtype=object)
            for p in self.X:
                if p[0] <= min_point[0] and p[1] <= min_point[1]:
                    min_point = p

                if p[0] >= max_point[0] and p[1] >= max_point[1]:
                    max_point = p

            max_point = [round(num, 1) for num in max_point]
            min_point = [round(num, 1) for num in min_point]
            self.bbox = BoundingBox([(min_point[0], min_point[1]),(max_point[0], max_point[1])])



        """ Create a voronoi partition of the data """
        self.vor = Voronoi(self.X,)

        """ Initiate kmeans clustering classificatio algorithm """
        self.kmeans = KMeans(n_clusters=1, random_state=0).fit(self.X)

        self.__compute_furthest_point()





    def __compute_furthest_point(self):

        """ Check if furthest point is one of the voronoi vertices. """
        min_point = self.bbox.min_point
        max_point = self.bbox.max_point
        for p in self.vor.vertices:
            if (min_point[0] >= p[0] and min_point[0] <= p[0]) and (max_point[1] >= p[1] and max_point[1] <= p[1]):
                temp_max = distance.euclidean(self.kmeans.cluster_centers_, p)
                if  temp_max > self.max_dist:
                    self.max_dist = temp_max
                    self.max_location = p

        """ Preparing the bounding box for testing """
        bottom_line = np.linspace(min_point[0], max_point[0], num=5) # bottom line
        top_line    = np.linspace(max_point[1], max_point[0], num=5) # top line
        left_line   = np.linspace(min_point[0], max_point[1], num=5) # far left line
        right_line  = np.linspace(max_point[0], max_point[1], num=5) # far left line
        
        # range of points on each line
        bl = [np.array([x,min_point[0]], dtype=object) for x in bottom_line]
        tl = [np.array([x,max_point[1]], dtype=object) for x in top_line]
        ll = [np.array([min_point[0], x], dtype=object) for x in left_line]
        rl = [np.array([max_point[0], x], dtype=object) for x in right_line]

        box_points = np.concatenate((bl, tl), axis=0)
        box_points = np.concatenate((box_points, ll), axis=0)
        box_points = np.concatenate((box_points, rl), axis=0)
        x1 = np.random.rand(box_points.shape[1])
        y = box_points.dot(x1)
        unique, index = np.unique(y, return_index=True)
        box_points[index]

        for p in box_points[index]:
            temp_max = distance.euclidean(self.kmeans.cluster_centers_, p)
            if  temp_max > self.max_dist:
                self.max_dist = temp_max
                self.max_location = p

    def show_plot(self):
        red = [1, 0, 0]
        blue = [0, 0, 1]
        colors = [red]
        x = self.kmeans.cluster_centers_[:,0]
        y = self.kmeans.cluster_centers_[:,1]

        fig, ax = plt.subplots(figsize=(15, 10))

        ax.scatter(self.X[:,0], self.X[:,1], c=[blue])
        ax.scatter(x, y, c=colors)

        plt.show()
   
    def show_analysis(self, name):
        blue = [0, 0, 1]

        fig, ax = plt.subplots(figsize=(15, 10))

        ax.scatter(self.poison_count_arr, self.scores, c=[blue])

        plt.title(name)
        plt.xlabel("Number of Poison Points (m)")
        plt.ylabel("Objective Function Score")

        plt.show()

    
    def add_poison(self, m):
        self.X = np.append(self.X, np.array([self.max_location]*m, dtype=object), axis=0)
        self.kmeans = KMeans(n_clusters=1, random_state=0).fit(self.X)

    def get_score(self):
        return -1 * self.kmeans.score(self.X)

    def run_analysis(self):
        increment_val = 10
        poison_count = 0

        previous_score = self.get_score()
        self.poison_count_arr = np.insert(self.poison_count_arr, 0, poison_count)
        self.scores = np.insert(self.scores, 0, previous_score)

        self.add_poison(increment_val)
        poison_count += increment_val

        current_score  = self.get_score()
        self.poison_count_arr = np.insert(self.poison_count_arr, 0, poison_count)
        self.scores = np.insert(self.scores, 0, previous_score)

        
        while not math.isclose(current_score, previous_score, rel_tol=1e-5):

            previous_score = current_score

            self.add_poison(increment_val)
            poison_count += increment_val

            current_score  = self.get_score()
            self.poison_count_arr = np.insert(self.poison_count_arr, 0, poison_count)
            self.scores = np.insert(self.scores, 0, previous_score)

    def get_analysis(self):
        return self.poison_count_arr, self.scores

    def show_vor(self):
        if self.vor:
            fig = voronoi_plot_2d(self.vor)
            plt.show()
        else:
            print("need to set voronoi")

    def get_poison_location(self):
        return self.max_location

    def get_proportion(self):
        print(f"Size of Dataset: {self.first_x.shape[0]}")
        print(f"Size of Poison Set: {self.poison_count_arr[0]}")
        return self.first_x.shape[0] / self.poison_count_arr[0]

    def show_new_mean_plot(self):
        x = self.kmeans.cluster_centers_[:,0]
        y = self.kmeans.cluster_centers_[:,1]

        point1 = self.kmeans.cluster_centers_
        point2 = self.max_location                          # np.array([[1.0,1.0]])
        point3 = np.mean(self.first_x,axis=0).reshape((1,2))

        x_values = [point1[:,0], point2[0]]
        y_values = [point1[:,1], point2[1]]

        x3_values = [point1[:,0], point3[:,0]]
        y3_values = [point1[:,1], point3[:,1]]


        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(x_values, y_values, linestyle="--")
        ax.plot(x3_values, y3_values, linestyle="--")

        red = [1, 0, 0]
        green = [0,1,0]
        blue = [0, 0, 1]
        ax.scatter(self.X[:,0], self.X[:,1], c=[[0, 0, 0]]) # original cluster
        ax.scatter(self.max_location[0], self.max_location[1], c=[red])             # m poision points
        ax.scatter(x, y, c=[blue])                # new mean after (X U P)
        ax.scatter(point3[:,0], point3[:,1], c=[green])
        ax.annotate(r'$\mu_{X}$', (point3[:,0],point3[:,1]), color='black',size=16) 
        ax.annotate(r'$\mu_{X \cup P}$', (x,y-.02), color='black',size=16) 

        ax.annotate(r'$P^*$', (1,1), color='black',size=16) 

        plt.show()


    def show_args(self):
        print("args: ", self.X)
        print("kwargs: ", self.r_data)