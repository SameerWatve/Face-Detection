from DataHandling import load_train_data, load_CV_data, get_MC
import numpy as np
from numpy.linalg import inv
from scipy.special import gamma,digamma
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import optimize
import cv2
from numpy import array


len_train = 1000
len_CV = 100

E_h = np.zeros(len_train)
E_log_h = np.zeros(len_train)
delta = np.zeros(len_train)        


def get_delta(X,i_index,mu,sigma):
    term1 = np.matmul( (X[i_index].reshape(-1,1)-mu).T,inv(sigma))
    term2 = np.matmul(term1,(X[i_index].reshape(-1,1) - mu))
    return term2    
    
def prob(i_index,v,mu,sigma,X):
    D = mu.shape[0]
    c1 = gamma( (v + D)/2.0 ) / ( pow( (v * np.pi), D/2 )*np.sqrt(np.linalg.det(sigma))*gamma(v/2))
    term2 = get_delta(X,i_index,mu,sigma)    
    c2 = (1 + term2/v)
    val = c1 * pow(c2, -(v+D)/2)
    return val[0,0] #* 100000.0

def get_E_hi(i_index,v,mu,sigma,X):
    D = mu.shape[0]
    term1 = np.matmul((X[i_index].reshape(-1,1)-mu).T , inv(sigma))
    term2 = np.matmul(term1, X[i_index].reshape(-1,1)-mu)[0,0]
    val = (v + D) / (v + term2)
    return val
    
def get_E_log_hi(i_index,v,mu,sigma,X):
    D = mu.shape[0]
    term1 = np.matmul((X[i_index].reshape(-1,1)-mu).T , inv(sigma))
    term2 = np.matmul(term1, X[i_index].reshape(-1,1)-mu)[0,0]
    val = digamma((v+D)/2) - np.log( (v + term2)/2 )
    return val

def tCost(v):
    val = 0
    for i in range(0,len_train):
       val = val + (v/2-1)*E_log_h[i] - (v/2)*E_h[i] - (v/2)*np.log(v/2) - np.log(gamma(v/2))                 
    return -val

def E_step(v,mu,sigma,X):
    for i in range(0,len_train):
        term = np.matmul( (X[i].reshape(-1,1)-mu).T , inv(sigma))
        delta[i] = np.matmul(term , (X[i].reshape(-1,1) - mu))
        E_h[i] = get_E_hi(i,v,mu,sigma,X)
        E_log_h[i] = get_E_log_hi(i,v,mu,sigma,X)
    return [delta, E_h, E_log_h]
        

def perform_EM_round(v,mu,sigma,X):
    D = mu.shape[0]
    global delta, E_h, E_log_h    
    print("E-step")
    [delta, E_h, E_log_h] = E_step(v,mu,sigma,X)
    
    'Updating Mean'            
    temp_mean = np.zeros((D,1))
    denom = 0
    for i in range(0,len_train):
        temp_mean = temp_mean + E_h[i]*X[i].reshape(-1,1)
        denom = denom + E_h[i]
    mu = temp_mean/denom    
        
    print("Updating Variance")
    num = np.zeros((D,D))
    for i in range(0,len_train):
        prod = np.matmul( (X[i].reshape(-1,1) - mu) , (X[i].reshape(-1,1) - mu).T )
        num = num + E_h[i]*prod
    sigma = num/denom
    sigma = np.diag( np.diag(sigma) )        

    print("Finding argmin v")
    v = optimize.fmin(tCost,v)          
     
    return [v[0], mu, sigma]  


print("Loading Data")
X_train_face = load_train_data('face')
X_train_nonface = load_train_data('nonface')
X_CV_nonface = load_CV_data('nonface')
X_CV_face = load_CV_data('face')

[mean_face, covar_face] = get_MC(X_train_face)
[mean_nonface, covar_nonface] = get_MC(X_train_nonface)


v_face = 50
mu_face = mean_face
sigma_face = covar_face

v_nonface = 50
mu_nonface = mean_nonface
sigma_nonface = covar_nonface

n_iter = 20
print("Training")

print("mix_face")
for i in range(0, n_iter):
    print(i)
    [v_face, mu_face, sigma_face] = perform_EM_round(v_face,mu_face.reshape(-1,1), sigma_face, X_train_face)
    print("v_face = ", v_face)
 

E_h = np.zeros(len_train)
E_log_h = np.zeros(len_train)
delta = np.zeros(len_train)  
 
print("mix_nonface")
for i in range(0, n_iter):
    print(i)
    [v_nonface, mu_nonface, sigma_nonface] = perform_EM_round(v_nonface, mu_nonface.reshape(-1, 1), sigma_nonface, X_train_nonface)
    print("v_nonface = ", v_nonface)


print("Testing")
prob_face_facedata = array([])
prob_nonface_facedata = array([])
prob_face_nonfacedata = array([])
prob_nonface_nonfacedata = array([])


for i in range(len_CV):
    prob_face_facedata = np.append( prob_face_facedata , prob(i,v_face,mu_face.reshape(-1,1),sigma_face,X_CV_face) )
    prob_face_nonfacedata = np.append( prob_face_nonfacedata , prob(i,v_face,mu_face.reshape(-1,1),sigma_face,X_CV_nonface) )
    prob_nonface_facedata = np.append( prob_nonface_facedata , prob(i,v_nonface,mu_nonface.reshape(-1,1),sigma_nonface,X_CV_face) )
    prob_nonface_nonfacedata = np.append( prob_nonface_nonfacedata , prob(i,v_nonface,mu_nonface.reshape(-1,1),sigma_nonface,X_CV_nonface) )
        

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
predictions = np.append(post_face_nonface_data, post_face_face_data )
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

face_mean1 = mean_face
min_val = np.min(face_mean1)
max_val = np.max(face_mean1)
face_mean1 = ((face_mean1-min_val)/(max_val-min_val) * 255.0).astype('uint8')
face_mean = np.reshape(face_mean1, (10, 10))

nonface_mean1 = mean_nonface
min_val = np.min(nonface_mean1)
max_val = np.max(nonface_mean1)
nonface_mean1 = ((nonface_mean1-min_val)/(max_val-min_val) * 255.0).astype('uint8')
nonface_mean = np.reshape(nonface_mean1, (10, 10))

face_variance1 = np.diag(sigma_face)
min_val = np.min(face_variance1)
max_val = np.max(face_variance1)
face_variance1 = ((face_variance1-min_val)/(max_val-min_val) * 255.0).astype('uint8')
face_variance = np.reshape(face_variance1, (10, 10))

nonface_variance1 = np.diag(sigma_nonface)
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


