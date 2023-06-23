import numpy as np

class FakeDataGen2D_Poisson:
    def __init__(self, pdf, Data_range, bins, scaler=1):
        """

        The fake data generator

        Arguments
        ---------
        pdf (Callable): the probability density function
        Data_range (List[List[float]]): The range of the Fake data
        bins (int): the number of the bins
        
        """

        self.pdf = pdf
        self.scaler = scaler
        self.Data_range  = Data_range
        self.bins = bins
        # Generate a grid of x and y value
        x = np.linspace(Data_range[0][0], Data_range[0][1], bins)
        y = np.linspace(Data_range[1][0], Data_range[1][1], bins)
        self.x, self.y  = np.meshgrid(x, y)
        self.FakeData = np.floor(self.pdf(np.column_stack((self.x.flatten(), self.y.flatten())))*((self.Data_range[0][1]-self.Data_range[0][0])*(self.Data_range[1][1]-self.Data_range[1][0])/(self.bins)**2)*self.scaler)
        self.FakeData = np.asanyarray([np.random.poisson(datai) for datai in self.FakeData])