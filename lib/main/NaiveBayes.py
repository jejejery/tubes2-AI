import numpy as np
import pandas as pd
import lib.main.Likelihood as lk

class NaiveBayes:
    """
    Naive Bayes classifier untuk model multivariat bernoulli, multinomial dan gaussian
    """

    """
    @param categorical_features: array-like, shape = [n_features]
        Indices of categorical features. contoh ["nama_hewan", "jenis_hewan", "jumlah_kaki"]
        Apabila suatu fitur numerik yang memenuhi tresshold tertentu (misal 10 unique values) maka fitur tersebut dapat dianggap sebagai fitur kategorikal
    @param gaussian_features: array-like, shape = [n_features]
        Indices of gaussian features for continuous feature. contoh ["berat_hewan", "tinggi_hewan", "clock_speed"]
    @param non_gaussian_features: array-like, shape = [n_features]
        Indices of non gaussian features for continuous feature. contoh ["jumlah_prosessor", "jumlah_faskes"].
        Fitur ini merupakan diskritisasi dari fitur yang seharusnya kontinu
    @param alpha: float, optional (default=1.0)
        Smoothing untuk kalkulasi likelihood
    @param prior_probs: array-like, shape = [n_classes]
        Prior probabilities -> merupakan probabilitas dari masing-masing nilai unik dari kolom target
    @param epsilon: float, optional (default=1e-9)
        Epsilon untuk smoothing dan menghindari pembagian dengan nol
    """
    def __init__(self, categorical_features=None, gaussian_features = None, non_gaussian_features = None, alpha=1, 
                 prior_probs=None, epsilon=1e-9):
        
        self.alpha = alpha
        self.prior_probs = prior_probs
        self.epsilon = epsilon

        """
        Variabel untuk menyimpan nilai dari masing-masing fitur
        var_smoothing: float, optional (default=1e-9)
            Melakukan smoothing pada varians apabila melakukan handle data yang merupakan gaussian
        num_features: int
            Jumlah fitur yang ada pada data
        _is_fitted: boolean
        """
        self.var_smoothing = epsilon
        self.num_features = 0
        self._is_fitted = False
        
        """
        Menunjukkan fitur mana saja yang merupakan fitur kategorikal, gaussian, dan non gaussian
        contoh: 
        - categorical_features = ["hewan", "nama_kota"]
        - gaussian_features = ["berat_hewan", "tinggi_hewan"]
        - non_gaussian_features = ["jumlah_prosessor", "jumlah_faskes"]
        """
        self.categorical_features = categorical_features
        self.gaussian_features = gaussian_features
        self.non_gaussian_features = non_gaussian_features

        """
        Komponen untuk menghitung posterior probability
        dimana log_posterior probability = log_prior_probability + log_likelihood
        dengan operator * merupakan operasi perkalian element-wise pada numpy array

        prior_probs: array-like, shape = [n_classes]
            Prior probabilities -> merupakan probabilitas dari masing-masing nilai unik dari kolom target (dalam log probability)
        likehoods: array-like of array-like, shape = [n_features]
            Mendefinisikan OBJEK likelihoods yang telah didefinisikan pada class Likelihood
            akan dipanggil pada method "fit" untuk menginisiasi fungsi likelihood
            akan digunakan sebagai komponen untuk menghitung posterior probability pada method "predict" -> akan mereturn vector likelihood
        """
        self.prior_probs = []
        self.likehoods = []
        self.posteriors = []



    # @staticmethod
    # def hello_world():
    #     print("Hello from NaiveBayes moduleee !")


    """
    X: pandas dataframe kolom fitur
    y: nama kolom target pada dataframe X (dalam bentuk pandas series)
    """

    def fit(self, X, y):
        print("fitting NaiveBayes")

        #1. Kalkulasi prior probability
        if(self.prior_probs is None):
            self.prior_probs = self.count_prior_probs(y)


    """
    y: pandas series dari kolom target
    """
    def count_prior_probs(self, y):
        prior_probs = []
        for target in y.unique():
            prior_probs.append(y[y == target].count()/y.count())
        return np.log(prior_probs)
    #will be implemented later
    """
    input:
        X: data yang ingin diprediksi, contoh X =<p1, p2, p3, p4, p5> , suatu vektor yang akan memprediksi nilai target Y berdasarkan fit
        y: nilai prediksi berdasarkan fitur X, contoh: y = 1 atau y = 0 (kasus binary classification)
    """
    def predict(self, X):
        return None

    def debug():
        print("debugging NaiveBayes module")