from DataHandling import load_train_data, load_CV_data
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import cv2
from numpy import array

len_train = 1000
len_CV = 100
np.random.seed(3)
K = 3


class GaussianMix():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.length = self.mu.shape[1]        
        self.lambda_val = np.random.dirichlet(np.ones(K), size=1)[0]
        self.r = np.random.dirichlet(np.ones(K), size=len_train)
        
    def pdf(self, k_index, i_index, X):
        term1 = -0.5 * (X[i_index].reshape(-1, 1) - self.mu[k_index]).T
        term2 = inv(self.sigma[k_index])
        term3 = X[i_index].reshape(-1, 1) - self.mu[k_index]
        expo1 = np.matmul(term1, term2)
        expo2 = np.matmul(expo1, term3)[0, 0]
        val = np.exp(expo2)
        val = val / np.sqrt( np.linalg.det(self.sigma[k_index]) )
        val = val/((2*np.pi) ** (100/2))
        return val

    def get_prob(self, i_index, X):
        val = 0
        for k_index in range(K):
            val = val + self.lambda_val[k_index] * self.pdf(k_index, i_index, X)
        return val         
                
    def E_step(self,i_index, k_index, X):
        'This will update r matrix'    
        temp = 0
        for j in range(K):
            temp = temp + self.lambda_val[j] * self.pdf(j, i_index, X)
        self.r[i_index,k_index] = self.lambda_val[k_index] * self.pdf(k_index, i_index, X) / temp
    
    def M_step(self, i_index, k_index):
        'Updating Lambda'
        num = 0
        denom = 0
        for i in range(len_train):
            num = num + self.r[i, k_index]
            for j in range(0,K):
                denom = denom + self.r[i, j]
        self.lambda_val[k_index] = 1.0*num/denom
                  
    def update(self, i_index, k_index, X):
        'Updating Mu' 
        num = np.zeros((self.length, 1))
        denom = 0
        for i in range(0,len_train):
            num = num + self.r[i,k_index] * X[i].reshape(-1, 1)
            denom = denom + self.r[i, k_index]
        self.mu[k_index] = 1.0 * num / denom
        
        'Updating Sigma'
        num = np.zeros((self.length, self.length))
        denom = 0
        for i in range(len_train):
            num = num + self.r[i, k_index] * np.matmul( (X[i].reshape(-1, 1) - self.mu[k_index]) , (X[i].reshape(-1, 1) - self.mu[k_index]).T )
            denom = denom + self.r[i, k_index]
        self.sigma[k_index] = 1.0*num/denom    
        self.sigma[k_index] = np.diag( np.diag(self.sigma[k_index]) )
    
    def perform_EM_round(self, i_index, k_index, X):
        'E-step'
        self.E_step(i_index, k_index, X)
        
        'M-step'
        self.M_step(i_index, k_index)
        
        'update'
        self.update(i_index, k_index, X)
        
    def perform_EM(self, k_index, X):
        for i in range(0, len_train):
            self.perform_EM_round(i, k_index, X)
    
    def display(self, k_index):
        print("Mean = ", self.mu[k_index])
        print("Variance = ", self.sigma[k_index])
               

print("Loading Data")
X_train_face = load_train_data('face')
X_train_nonface = load_train_data('nonface')
X_CV_nonface = load_CV_data('nonface')
X_CV_face = load_CV_data('face')


print("Training")
mean_face = np.zeros((K, 100, 1))
mean_nonface = np.zeros((K, 100, 1))

covar_face = np.zeros((K, 100, 100))
np.fill_diagonal(covar_face[0], np.random.rand(100))
np.fill_diagonal(covar_face[1], np.random.rand(100))
np.fill_diagonal(covar_face[2], np.random.rand(100))

covar_nonface = np.zeros((K, 100, 100))
np.fill_diagonal(covar_nonface[0], np.random.rand(100))
np.fill_diagonal(covar_nonface[1], np.random.rand(100))
np.fill_diagonal(covar_nonface[2], np.random.rand(100))


mix_face = GaussianMix(mean_face, covar_face)
mix_nonface = GaussianMix(mean_nonface, covar_nonface)

print("Performing EM FACE")
for k in range(0, K):
    print("FACE component - ", k)
    mix_face.perform_EM(k, X_train_face)
    print("mix_face -> ", mix_face.lambda_val)


print("Performing EM NON-FACE")
for k in range(0, K):
    print("NON-FACE component - ", k)
    mix_nonface.perform_EM(k, X_train_nonface)
    print("mix_nonface -> ", mix_nonface.lambda_val)


print("Testing")

prob_face_facedata = array([])
prob_nonface_facedata = array([])
prob_face_nonfacedata = array([])
prob_nonface_nonfacedata = array([])

for i in range(0, len_CV):
    inp_facedata = X_CV_face[i].reshape(-1, 1)
    inp_nonfacedata = X_CV_nonface[i].reshape(-1, 1)
    
    prob_face_facedata = np.append(prob_face_facedata, mix_face.get_prob(i, X_CV_face) )
    prob_nonface_facedata = np.append(prob_nonface_facedata, mix_nonface.get_prob(i, X_CV_face) )
    prob_face_nonfacedata = np.append(prob_face_nonfacedata, mix_face.get_prob(i, X_CV_nonface) )
    prob_nonface_nonfacedata = np.append(prob_nonface_nonfacedata, mix_nonface.get_prob(i, X_CV_nonface) )

post_face_face_data = prob_face_facedata/( prob_face_facedata + prob_nonface_facedata )
post_nonface_face_data = prob_nonface_facedata/(prob_face_facedata + prob_nonface_facedata )
post_face_nonface_data = prob_face_nonfacedata/(prob_face_nonfacedata + prob_nonface_nonfacedata )
post_nonface_nonface_data = prob_nonface_nonfacedata/(prob_face_nonfacedata + prob_nonface_nonfacedata )

print("performance at threshold 0.5")
TP = 0
TN = 0
FN = 0
FP = 0
for n in range(100):
    if post_face_face_data[n] > 0.5:
        TP += 1
    else:
        FN += 1

for n in range(100):
    if post_nonface_nonface_data[n] > 0.5:
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
actual = np.append(temp1, temp2)
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.plot(false_positive_rate, true_positive_rate, 'b')
plt.title('ROC')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()
ind, threshold = min(enumerate(thresholds), key=lambda x: abs(0.5 - x[1]))
print(threshold, false_positive_rate[ind], true_positive_rate[ind])

print('Visualization')
face_mean1 = mix_face.mu[2]
min_val = np.min(face_mean1)
max_val = np.max(face_mean1)
face_mean1 = ((face_mean1-min_val)/(max_val-min_val) * 255.0).astype('uint8')
face_mean = np.reshape(face_mean1, (10, 10))

nonface_mean1 = mix_nonface.mu[2]
min_val = np.min(nonface_mean1)
max_val = np.max(nonface_mean1)
nonface_mean1 = ((nonface_mean1-min_val)/(max_val-min_val) * 255.0).astype('uint8')
nonface_mean = np.reshape(nonface_mean1, (10, 10))

face_variance1 = np.diag(mix_face.sigma[2])
min_val = np.min(face_variance1)
max_val = np.max(face_variance1)
face_variance1 = ((face_variance1-min_val)/(max_val-min_val) * 255.0).astype('uint8')
face_variance = np.reshape(face_variance1, (10, 10))

nonface_variance1 = np.diag(mix_nonface.sigma[2])
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