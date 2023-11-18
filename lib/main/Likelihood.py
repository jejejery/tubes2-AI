
import numpy as np

"""
implementasi dari fungsi likelihood untuk melakukan perhitungan likelihood suatu data
prerequisite: data sudah dilakukan preprocessing

input:
    Xi: np.array 1 dimensi, X ke i
    y: np.array kolom target, 1 dimensi
    
    target: nama kolom target yang bersesuaian dengan kolom fitur
    tipe_fitur: tipe fitur yang ingin dihitung likelihoodnya ("categorical" atau "numerikal" atau "gaussian")

"""
class Likelihood:
    
    def __init__(self, Xi, y, tipe_fitur, alpha=1):
        try:
            self.Xi = Xi
            self.y = y
            if(len(self.Xi) != len(self.y)):
                raise Exception("jumlah data Xi dan y tidak sama")
            self.tipe_fitur = tipe_fitur
            self.alpha = alpha
        except:
            print("error in init Likelihood")
            raise

    



    """
    input:
        value: data yang ingin dihitung likelihoodnya -> array like
    
    output:
        vector likelihood dari data x sesuai dengan fitur yang ingin dihitung likelihoodnya
        misal kolom target memiliki 3 nilai unik, maka vector likelihoodnya memiliki panjang 3
        vektor likelihood ini akan digunakan untuk melakukan perhitungan posterior probability

    """
    def count_likelihood(self, value):
        if self.tipe_fitur == "categorical":
            return self.count_likelihood_categorical(value)
        elif self.tipe_fitur == "numerikal":
            return self.count_likelihood_numerikal(value)
        elif self.tipe_fitur == "gaussian":
            return self.count_likelihood_gaussian(value)
        else:
            return None
        

    """
    fungsi untuk menghitung likelihood dari data numerikal
    input:
        value: data yang ingin dihitung likelihoodnya : array like
    output:
        vektor likelihood dari data x sesuai dengan fitur yang ingin dihitung likelihoodnya
        misal kolom target memiliki 3 nilai unik, maka vector likelihoodnya memiliki panjang 3
        vektor likelihood ini akan digunakan untuk melakukan perhitungan posterior probability (dalam log probability)
    """
    def count_likelihood_categorical(self, value):
        try:
            unique_val = np.unique(self.y) # nilai unik dari kolom target
            print(value)
            likelihood = [] # vector likelihood
            for uval in unique_val:
                # menghitung jumlah data yang memiliki nilai val pada kolom target
                loc = np.where(self.y == uval)[0]
                Xi_val = self.Xi[loc]
                Xi_unique = len(np.unique(self.Xi))
                # menghitung likelihood value terhadap Xi_val (data yang memiliki nilai val pada kolom target)
                # likelihood mungkin bernilai 0 jika tidak ada data yang memiliki nilai val pada kolom target
                # oleh karena itu, gunakan laplace smoothing dengan alp
                # ha = 1
                numerator = np.array([np.sum(Xi_val == elem) for elem in value])
                denominator = len(Xi_val) + self.alpha * Xi_unique
                likelihood.append( numerator/ (denominator))
            print("likelihood")
            print(likelihood)
            return np.log(likelihood).T
        except:
            print("error in count_likelihood_categorical")
            raise