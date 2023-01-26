from typing import List, Callable
import time
import numpy as np
import matplotlib.pyplot as plt

class MHMC:  
    def __init__(
    self, 
    rho: Callable, # the target distribution of the parameter
    seed=123 # setting the random seed of the distribution
    ) -> None:
        np.random.seed(seed)
        self.rho = rho

    def predict(
        self,
        theta0, # the initial value of the parameter
        qProb: Callable, # probability of the proposal distribution of the parameter
        qSamp: Callable, # draw the sample with the proposal distribution
        epoch: int # run the MCMC for n epoch
    ) -> List[int]:
        self.qProb = qProb
        self.qSamp = qSamp
        self.theta_n = theta0
        for i in range(0, epoch):
            # Updating the parameter from the proposal disribution
            self.theta_nPlus1 = self.qSamp(self.theta_n)
            self.alpha = min(1, (self.rho(self.theta_nPlus1)*self.qProb(self.theta_nPlus1,self.theta_n))/(self.rho(self.theta_n)*self.qProb(self.theta_n,self.theta_nPlus1)))
            # Deciding whether to reject the update of the parameter
            self.u = np.random.default_rng().uniform(0, 1, 1)[0]
            if(self.alpha>=self.u):
                self.theta_n = self.theta_nPlus1
            else:
                pass
        return self.theta_n
    
    def PredictAndTest(
        self,
        theta0, # the initial value of the parameter
        qProb: Callable, # probability of the proposal distribution of the parameter
        qSamp: Callable, # draw the sample with the proposal distribution
        testFreq: int, # test frequency
        epoch: int, # run the MCMC for n epoch
        thetaTrue=[], # the true value of the parameter
        runningTime=False
    ) -> List[float]:
        if runningTime:
            self.start_time = time.time()
        if testFreq>epoch:
            print("Test frequency should smaller or equal to the training epoch.")
            return theta0
        self.qProb = qProb
        self.qSamp = qSamp
        self.I = epoch//testFreq
        if thetaTrue:
            self.MSQLoss = [np.sqrt(np.sum(np.square(np.array(thetaTrue)-np.array(theta0))))]
            self.Epoch = [i*testFreq for i in range(0, self.I+1)]
        self.thetan = theta0
        for _ in range(0, self.I):
            self.thetan = self.predict(theta0=self.thetan, qProb=self.qProb, qSamp=self.qSamp, epoch=testFreq)
            if thetaTrue:
                self.MSQLoss.append(np.sqrt(np.sum(np.square(np.array(thetaTrue)-np.array(self.thetan)))))
        if epoch%testFreq!=0:
            self.thetan = self.predict(theta0=self.thetan, qProb=self.qProb, qSamp=self.qSamp, epoch=epoch%testFreq)
            if thetaTrue:
                self.MSQLoss.append(np.sqrt(np.sum(np.square(np.array(thetaTrue)-np.array(self.thetan)))))
                self.Epoch.append(self.Epoch[-1]+epoch%testFreq)
        if runningTime:
            self.runTime = (time.time()-self.start_time)
            self.runTimeS = (self.runTime)%60
            self.runTime -= self.runTimeS
            self.runTimeM = ((self.runTime)%3600)//60
            self.runTime -= self.runTimeM
            self.runTimeH = self.runTime//3600
            print(f"The total running time = ({self.runTimeH:.0f} hours, {self.runTimeM:.0f} minutes, {self.runTimeS:.2f} seconds)")
        if thetaTrue:
            plt.rcParams["figure.figsize"] = (12,8)
            plt.plot(self.Epoch, self.MSQLoss, label="Mean_Square_Loss")
            plt.legend(fontsize=20)
            plt.xlabel("epoch", fontsize=20)
            plt.ylabel("loss value", fontsize=20)
            plt.tight_layout()
            plt.show()
        return self.thetan