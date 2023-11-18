
"""
implementasi dari fungsi likelihood untuk melakukan perhitungan likelihood suatu data
prerequisite: data sudah dilakukan preprocessing

input:
    dataframe: pandas dataframe dari dataset yang ingin dihitung likelihoodnya
    fitur: nama kolom fitur yang ingin dihitung likelihoodnya
    target: nama kolom target yang bersesuaian dengan kolom fitur
    tipe_fitur: tipe fitur yang ingin dihitung likelihoodnya ("categorical" atau "numerikal" atau "gaussian")

"""
class Likelihood:
    
    def __init__(self, dataframe, fitur, target, tipe_fitur):
        try:
            self.fitur = fitur
            self.target = target
            self.tipe_fitur = tipe_fitur
            self.dataframe = dataframe[[self.fitur, self.target]]
        except:
            print("error in init Likelihood")
            raise

    



    """
    input:
        x: data yang ingin dihitung likelihoodnya
    
    output:
        vector likelihood dari data x sesuai dengan fitur yang ingin dihitung likelihoodnya
        misal kolom target memiliki 3 nilai unik, maka vector likelihoodnya memiliki panjang 3
        vektor likelihood ini akan digunakan untuk melakukan perhitungan posterior probability

    """
    def count_likelihood(self, x):
        if self.tipe_fitur == "categorical":
            return self.count_likelihood_categorical(x)
        elif self.tipe_fitur == "numerikal":
            return self.count_likelihood_numerikal(x)
        elif self.tipe_fitur == "gaussian":
            return self.count_likelihood_gaussian(x)
        else:
            return None