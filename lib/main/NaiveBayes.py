import numpy as np
import pandas as pd
from lib.main.Likelihood import Likelihood

class CategoricalNaiveBayes:
    """
    Naive Bayes classifier untuk proses bernoulli (Kategorikal)
    """

    """
    @param categorical_features: array-like, shape = [n_features]. contoh np.array([0, 1, 2, 3, 4]) -> 0, 1, 2, 3, 4 merupakan lokasi dari fitur kategorikal pada data
        Apabila suatu fitur numerik yang memenuhi tresshold tertentu (misal 10 unique values) maka fitur tersebut dapat dianggap sebagai fitur kategorikal
    @param gaussian_features: array-like, shape = [n_features]. contoh seperti di atas
        Indices of gaussian features for continuous feature.
    @param numerical_features: array-like, shape = [n_features]. contoh seperti di atas
        Indices of non gaussian features for continuous feature. Using discritization method
    @param alpha: float, optional (default=1.0)
        Smoothing untuk kalkulasi likelihood
    @param prior_probs: array-like, shape = [n_classes]
        Prior probabilities -> merupakan probabilitas dari masing-masing nilai unik dari kolom target
    @param epsilon: float, optional (default=1e-9)
        Epsilon untuk smoothing dan menghindari pembagian dengan nol
    """
    def __init__(self, categorical_features=[], gaussian_features = [], numerical_features = [], alpha=1, 
                 prior_probs=None, epsilon=1e-9, kernel = False):
        
        self.alpha = alpha
        self.prior_probs = prior_probs
        self.epsilon = epsilon
        self.kernel_method = kernel

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
        contoh jelas seperti di atas, sebuah np.array yang menunjukkan index dari fitur kategorikal, gaussian, dan non gaussian
        """
        self.categorical_features = categorical_features
        self.gaussian_features = gaussian_features
        self.numerical_features = numerical_features

        """
        Komponen untuk menghitung posterior probability
        dimana log_posterior probability = log_prior_probability + log_likelihood
        dengan operator * merupakan operasi perkalian element-wise pada numpy array

        prior_probs: array-like, shape = [n_classes]
            Prior probabilities -> merupakan probabilitas dari masing-masing nilai unik dari kolom target (dalam log probability)
        likelihoods: array-like of array-like, shape = [n_features]
            Mendefinisikan OBJEK likelihoods yang telah didefinisikan pada class Likelihood
            akan dipanggil pada method "fit" untuk menginisiasi fungsi likelihood
            akan digunakan sebagai komponen untuk menghitung posterior probability pada method "predict" -> akan mereturn vector likelihood (dalam log probability)
        """
        self.prior_probs = None
        self.likelihoods = None
        self.posteriors = None



    # @staticmethod
    # def hello_world():
    #     print("Hello from NaiveBayes moduleee !")


    """
    X: np.array 2 dimensi, data yang ingin dilakukan fitting
    y: np.array kolom target, 1 dimensi
    """

    def fit(self, X : any, y : any):
        print("fitting NaiveBayes")
        #jika X masih berupa pandas dataframe, maka akan diubah menjadi numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        #jika y masih berupa pandas series, maka akan diubah menjadi numpy array
        if isinstance(y, pd.Series):
            y = y.values

        #1. Kalkulasi prior probability
   
        self.prior_probs = self.count_prior_probs(y)
        #2. build likelihoods
        self.likelihoods = self.build_likelihoods(X, y)
        self._is_fitted = True


    """
    X: np.array 2 dimensi, data yang ingin dilakukan fitting
    y: np.array kolom target, 1 dimensi
    """
    def build_likelihoods(self, X, y):
        try:
            likelihoods = []
            for i in range(X.shape[1]):
                if i in self.categorical_features:
                    likelihoods.append(Likelihood(X[:, i], y, "categorical", self.alpha))
                elif i in self.gaussian_features:
                    likelihoods.append(Likelihood(X[:, i], y, "gaussian", self.alpha))
                elif i in self.numerical_features:
                    likelihoods.append(Likelihood(X[:, i], y, "numerikal", alpha=self.alpha, the_kernel=self.kernel_method))
                else:
                    likelihoods.append(Likelihood(X[:, i], y, "numerikal", alpha=self.alpha, the_kernel=self.kernel_method))
            return likelihoods
        except Exception as e:
            print("error in build_likelihoods")
            raise e

    """
    X: np.array 2 dimensi, data yang ingin dilakukan fitting
    """
    def predict(self, X) -> np.array:
        if isinstance(X, pd.DataFrame):
            X = X.values
        if not self._is_fitted:
            raise Exception("model belum di fit")
        else:
            self.posteriors = self.posterior_probability(X)
            return np.argmax(self.posteriors, axis=1)
       
        

    def posterior_probability(self, X) -> np.array:
        the_posteriors = None
        for i in range(X.shape[1]):
            if(i == 0):
                the_posteriors = self.likelihoods[i].count_likelihood(X[:, i])
            else:
                the_posteriors = the_posteriors + self.likelihoods[i].count_likelihood(X[:, i])
        the_posteriors = the_posteriors + self.prior_probs
        return the_posteriors

    """
    y: pandas series dari kolom target
    """
    def count_prior_probs(self, y):
        log_prior_probs = []
        unique_val = np.unique(y) # nilai unik dari kolom target
        for uval in unique_val:
            # menghitung jumlah data yang memiliki nilai val pada kolom target
            loc = np.where(y == uval)[0]
            log_prior_probs.append(len(loc)/len(y))
        return np.log(log_prior_probs).squeeze()
    #will be implemented later
    """
    input:
        X: data yang ingin diprediksi, contoh X =<p1, p2, p3, p4, p5> , suatu vektor yang akan memprediksi nilai target Y berdasarkan fit
        y: nilai prediksi berdasarkan fitur X, contoh: y = 1 atau y = 0 (kasus binary classification)
    """
    def debug():
        print("debugging NaiveBayes module")


class GaussianNaiveBayes:
    """
    Naive Bayes classifier untuk proses Gaussian.
    """

    def __init__(self, gaussian_features = [], numerical_features = [], alpha=1, 
                 prior_probs=None, epsilon=1e-9, kernel = False):
        
        self.alpha = alpha
        self.prior_probs = prior_probs
        self.epsilon = epsilon
        self.kernel_method = kernel

        self.var_smoothing = epsilon
        self.num_features = 0
        self._is_fitted = False
        
        self.gaussian_features = gaussian_features
        self.numerical_features = numerical_features

        self.prior_probs = None
        self.likelihoods = None
        self.posteriors = None

    def _check_input(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        return X, y
    
    def build_likelihoods(self, X, y):
        try:
            likelihoods = []
            for i in range(X.shape[1]):
                if i in self.gaussian_features:
                    likelihoods.append(Likelihood(X[:, i], y, "gaussian", self.alpha))
                elif i in self.numerical_features:
                    likelihoods.append(Likelihood(X[:, i], y, "gaussian", alpha=self.alpha, the_kernel=self.kernel_method))
                else:
                    likelihoods.append(Likelihood(X[:, i], y, "gaussian", alpha=self.alpha, the_kernel=self.kernel_method))
            return likelihoods
        except Exception as e:
            print("error in build_likelihoods")
            raise e

    def fit(self, X, y):
        # Konversi X dan y ke numpy array jika masih dalam bentuk DataFrame atau Series
        X, y = self._check_input(X, y)

        self.prior_probs = self.count_prior_probs(y)
        self.likelihoods = self.build_likelihoods(X,y)
        
        self._is_fitted = True

    def _check_X_input(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X

    def predict(self, X):
        if not self._is_fitted:
            raise Exception("Model belum di fit")

        X = self._check_X_input(X)
        posteriors = self.posterior_probability(X)
        return np.argmax(posteriors, axis=1)

    def posterior_probability(self, X):
        the_posteriors = None
        for i in range(X.shape[1]):
            if(i == 0):
                the_posteriors = self.likelihoods[i].count_likelihood(X[:, i])
            else:
                the_posteriors = the_posteriors + self.likelihoods[i].count_likelihood(X[:, i])
        the_posteriors = the_posteriors + self.prior_probs
        return the_posteriors

    def count_prior_probs(self, y):
        log_prior_probs = []
        unique_val = np.unique(y)
        for uval in unique_val:
            loc = np.where(y == uval)[0]
            log_prior_probs.append(len(loc)/len(y))
        return np.log(log_prior_probs).squeeze()

    

    @staticmethod
    def hello_world():
        print("Hello from GaussianNaiveBayes module")