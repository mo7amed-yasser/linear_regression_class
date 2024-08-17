class linear_regression:
    def __init__(self,learn_rate,iter,x,y):
        self.old_error =9999999999999999999999999999999999999
        self.x=x
        self.y=y
        self.iter=iter 
        self.learn_rate=learn_rate
        if self.x.shape[1] == 1:
            self.plot_1d()
        elif self.x.shape[1] == 2:
            self.plot_2d()
        self.wight,self.bwight=self.update(x=self.x,y=self.y ,iter=self.iter,learn_rate=self.learn_rate)
        print(self.wight,self.bwight)
        
    def wieght_gen(self,x):
        #generate wieght randomly
        b=np.ones(1)
        w=np.random.rand((x.shape[-1]))
        return b,w
    def update(self,x,y,iter,learn_rate):
        x_size=x.shape
        x = np.reshape(x, (-1, x.shape[-1]))
        b,w=self.wieght_gen(x)
        for i in range(iter):
            y_pred=np.dot(x,w)+b
            error = np.mean((y_pred-y)**2)
            print("error is :",error)
            dw= np.dot(x.T, (y_pred - y)) / len(y)  
            db=np.mean((y_pred-y))
            w=w-learn_rate*dw
            b=b-learn_rate*db
            if not self.check(error):
                break
        return w,b
    def check(self,error):
        if error > self.old_error:
            print("stop here :edit hyperparameter")
            return False
        elif self.old_error/error <1.01:
            print("slowly:")
            return False
        else :
            self.old_error =error
            return True
    def plot_1d(self):
        plt.figure(figsize=(12, 6))
        plt.scatter(self.x, self.y, color='blue', label='Actual Data')
        w, b = self.weight_gen(self.x)
        y_pred = np.dot(self.x, w) + b
        plt.plot(self.x, y_pred, color='red', label='Fitted Line')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title('1D Data and Fitted Line')
        plt.legend()
        plt.show()

    def plot_2d(self):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x[:, 0], self.x[:, 1], self.y, color='blue', label='Actual Data')
        
