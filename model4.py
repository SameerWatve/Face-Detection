from DataHandling import load_train_data, load_CV_data, get_MC
import numpy as np
from numpy.linalg import inv, det
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import cv2
from numpy import array

len_train = 1000
len_CV= 100
K = 5
D = 100

class factor:
    def __init__(self,mu,Sigma,phi):
        self.mu = mu
        self.Sigma = Sigma
        self.phi = phi
        self.E_h = np.zeros((len_train,K,D))
        self.E_hi_hT = np.zeros((len_train,K,K))

    def prob(self,X,i_index):
        sigma = np.matmul(self.phi,self.phi.T) + self.Sigma
        term1 = -0.5 * (X[i_index].reshape(-1,1) - self.mu).T
        term2 = inv(sigma)
        term3 = X[i_index].reshape(-1,1) - self.mu
        expo1 = np.matmul(term1,term2)
        expo2 = np.matmul(expo1,term3)[0,0]
        # val = expo2 / np.sqrt(np.linalg.det(sigma))
        # val = val / ((2 * np.pi) ** (100 / 2))
        val = expo2 - np.log((np.sqrt(det(sigma)) + ((2 * np.pi) ** (100 / 2))))
        return val

    def E_step(self,X):
        for i in range(0,len_train):
            #print i
            t1 = np.matmul(self.phi.T, inv(self.Sigma))
            t2 = np.matmul(t1, self.phi) + np.eye(K)
            t3 = np.matmul(inv(t2), self.phi.T)
            t4 = np.matmul(t3, inv(self.Sigma))
            self.E_h[i] = np.matmul(t4, X[i] - self.mu)
            self.E_hi_hT[i] = inv(t2) + np.matmul(self.E_h[i],self.E_h[i].T)

    def M_step(self,X):
        print('Updating phi')
        t1 = np.zeros((100,K))
        t2 = np.zeros((K,K))
        for i in range(len_train):
            t1 = t1 + np.matmul( (X[i]-self.mu) , self.E_h[i].T )
            t2 = t2 + self.E_hi_hT[i]
        self.phi = np.matmul(t1,t2)    
        
        print('Updating Sigma')
        temp = np.zeros((D,D))
        for i in range(0,len_train):
            temp = temp + np.matmul(X[i].reshape(-1,1)-self.mu, (X[i].reshape(-1,1)-self.mu).T)
            t2 = np.matmul(self.phi,self.E_h[i])
            t3 = np.matmul(t2, (X[i].reshape(-1,1)-self.mu ))
            temp = temp - t3
        
        temp = temp/len_train
        self.Sigma = temp
        self.Sigma = np.diag( np.diag(temp) )

    def perform_EM_step(self,X):
        self.E_step(X)
        self.M_step(X)        
        
    def display(self):
        print("Mean = ", self.mu)
        print("Variance = ", self.sigma)
        
        
print("Loading Data")
X_train_face = load_train_data('face')
X_train_nonface = load_train_data('nonface')
X_CV_nonface = load_CV_data('nonface')
X_CV_face = load_CV_data('face')

phi_face = np.random.rand(D,K)
sigma_face = np.random.rand(D, D)
sigma_face = np.diag(np.diag(sigma_face))

phi_nonface = np.random.rand(D,K)
sigma_nonface = np.random.rand(D, D)
sigma_nonface = np.diag(np.diag(sigma_nonface))


[mean_face,covar_face] = get_MC(X_train_face)
[mean_nonface,covar_nonface] = get_MC(X_train_nonface)

mix_face = factor(mean_face.reshape(-1,1), sigma_face, phi_face)
mix_nonface = factor(mean_nonface.reshape(-1,1), sigma_nonface, phi_nonface)


n_iter= 3


for i in range(n_iter):
    print(i)
    mix_face.perform_EM_step(X_train_face)
    mix_nonface.perform_EM_step(X_train_nonface)
      

print("Testing")

prob_face_facedata = array([])
prob_nonface_facedata = array([])
prob_face_nonfacedata = array([])
prob_nonface_nonfacedata = array([])

for i in range(len_CV):
    prob_face_facedata = np.append( prob_face_facedata , mix_face.prob(X_CV_face,i) )
    prob_nonface_facedata = np.append( prob_nonface_facedata , mix_nonface.prob(X_CV_face,i) )
    prob_face_nonfacedata = np.append( prob_face_nonfacedata , mix_face.prob(X_CV_nonface,i) )
    prob_nonface_nonfacedata = np.append( prob_nonface_nonfacedata , mix_nonface.prob(X_CV_nonface,i) )

post_face_face_data = prob_face_facedata/( prob_face_facedata + prob_nonface_facedata )
post_nonface_face_data = prob_nonface_facedata/( prob_face_facedata + prob_nonface_facedata )
post_face_nonface_data = prob_face_nonfacedata/( prob_face_nonfacedata + prob_nonface_nonfacedata )
post_nonface_nonface_data = prob_nonface_nonfacedata/( prob_face_nonfacedata + prob_nonface_nonfacedata )

print("performance at threshold 0.5")
TP = 0
TN = 0
FN = 0
FP = 0
for n in range(100):
    if post_face_face_data[n] > np.log(0.5):
        TP += 1
    else:
        FN += 1

for n in range(100):
    if post_nonface_nonface_data[n] > np.log(0.5):
        TN += 1
    else:
        FP += 1

FPR = FP / 100
FNR = FN / 100
MCR = (FP + FN) / 200

print(FPR)
print(FNR)
print(MCR)

print("Plotting ROC")
predictions = np.append( post_face_nonface_data , post_face_face_data )
temp1 = [0]*len_CV
temp2 = [1]*len_CV
actual = np.append(temp1,temp2)
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.plot(false_positive_rate, true_positive_rate, 'b')
plt.title('ROC')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()

print('Visualization')
face_mean1 = mix_face.mu
min_val = np.min(face_mean1)
max_val = np.max(face_mean1)
face_mean1 = ((face_mean1-min_val)/(max_val-min_val) * 255.0).astype('uint8')
face_mean = np.reshape(face_mean1, (10, 10))

nonface_mean1 = mix_nonface.mu
min_val = np.min(nonface_mean1)
max_val = np.max(nonface_mean1)
nonface_mean1 = ((nonface_mean1-min_val)/(max_val-min_val) * 255.0).astype('uint8')
nonface_mean = np.reshape(nonface_mean1, (10, 10))

face_variance1 = np.diag(mix_face.Sigma)
min_val = np.min(face_variance1)
max_val = np.max(face_variance1)
face_variance1 = ((face_variance1-min_val)/(max_val-min_val) * 255.0).astype('uint8')
face_variance = np.reshape(face_variance1, (10, 10))

nonface_variance1 = np.diag(mix_nonface.Sigma)
min_val = np.min(nonface_variance1)
max_val = np.max(nonface_variance1)
nonface_variance1 = ((nonface_variance1-min_val)/(max_val-min_val) * 255.0).astype('uint8')
nonface_variance = np.reshape(nonface_variance1, (10, 10))

cv2.imshow("Mean Face", face_mean)
cv2.imshow("Mean NON-Face", nonface_mean)
cv2.imshow('Cov1', face_variance)
cv2.imshow('Cov2', nonface_variance)
cv2.waitKey(0)
cv2.destroyAllWindows()