from sklearn.neighbors import KernelDensity
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
    
    def __init__(self, Xi, y, tipe_fitur, alpha=1, the_kernel = False, epsilon=1e-9, scott_rule = True):
        try:
            self.Xi = Xi
            self.y = y
            if(len(self.Xi) != len(self.y)):
                raise Exception("jumlah data Xi dan y tidak sama")
            self.tipe_fitur = tipe_fitur
            self.alpha = alpha
            self.kernel_method = the_kernel
            self.epsilon = epsilon
            self.scrot_rule = scott_rule
        except:
            print("error in init Likelihood")
            raise

    



    """
    input:
        value: data yang ingin dihitung likelihoodnya -> array like 1 dimensi
    
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
            return np.log(likelihood).T
        except:
            print("error in count_likelihood_categorical")
            raise


    """
    fungsi untuk menghitung likelihood dari data numerikal
    input:
        value: data yang ingin dihitung likelihoodnya : array like
    output:
        vektor likelihood dari data x sesuai dengan fitur yang ingin dihitung likelihoodnya
        misal kolom target memiliki 3 nilai unik, maka vector likelihoodnya memiliki panjang 3
        vektor likelihood ini akan digunakan untuk melakukan perhitungan posterior probability (dalam log probability)
    """
    def count_likelihood_numerikal(self, value):
        try:
            unique_val = np.unique(self.y) # nilai unik dari kolom target
            likelihood = [] # vector likelihood
            # using discretitation probabilities
            for uval in unique_val:
                # menghitung jumlah data yang memiliki nilai val pada kolom target
                loc = np.where(self.y == uval)[0]
                Xi_val = self.Xi[loc]
                likelihood.append(self.discretitation_probabilities(value, Xi_val, self.alpha))
            return np.log(np.array(likelihood)).T
        except:
            print("error in count_likelihood_numerikal")
            raise
    
    def count_likelihood_gaussian(self, value):
        try:
            unique_val = np.unique(self.y)  # nilai unik dari kolom target
            likelihood = []  # vector likelihood

            for uval in unique_val:
                loc = np.where(self.y == uval)[0]
                Xi_val = self.Xi[loc]

                mean = np.mean(Xi_val)
                var = np.var(Xi_val) + self.epsilon

                # Menghitung likelihood menggunakan rumus distribusi Gaussian
                likelihood_class = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((value - mean) ** 2) / (2 * var))
                likelihood.append(likelihood_class)

            return np.log(np.array(likelihood)).T
        except Exception as e:
            print("error in count_likelihood_gaussian")
            raise e

    """
    count probability of the test data
    test_data : np.array 1 dimensi
    train_data : np.array 1 dimensi
    alpha : float -> laplace smoothing constant
    """
    def discretitation_probabilities(self, test_data : np.array, train_data: np.array, alpha: float) -> np.array:
        if(self.kernel_method):
            kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(train_data.reshape([-1,1]))
            #handle 0 probability value with epsilon
            prob = np.exp(kde.score_samples(test_data.reshape([-1,1])))
            prob[prob < self.epsilon] = self.epsilon
            return prob
        else:
            min_val = train_data.min()
            max_val = train_data.max()
            if(self.scrot_rule):
                size = len(train_data)
                std_dev = np.std(train_data)
                bin_width = 3.5 * std_dev / np.power(size, 1/3)
                num_bins = int(np.ceil((np.max(train_data) - np.min(train_data)) / bin_width))
                
                x_axis = np.arange(min_val, max_val, bin_width)
                y_axis = np.array([np.sum((train_data >= x) & (train_data < x + bin_width)) for x in x_axis])
            else:
                size = len(train_data)
                num_bins = int(np.sqrt(size))
                x_axis = np.linspace(min_val, max_val, num_bins)
                #create sum array, ignore the last value
                y_axis = np.array([np.sum((train_data >= x) & (train_data < x + (max_val - min_val) / num_bins)) for x in x_axis])
            #calculate probability for each test data using histogram above
            y_index = np.digitize(test_data, x_axis, right=False)
            #the index start from 1, so we need to substract 1
            y_index = y_index - 1
            freq = y_axis[y_index]
            #handle freq value become 0 if the test data is out of range ( < min_val or > max_val)
            
            #get the index of test data that out of range
            out_of_range_index = np.where((test_data < min_val) | (test_data > max_val))

            #assign freq value to 0
            freq[out_of_range_index] = 0
            
            #perform laplace smoothing
            denominator = np.sum(y_axis) + (alpha * num_bins)
            return (freq + alpha) / denominator

    