from DataHandling import load_train_data, load_CV_data,  \
                         get_MC
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy import array
from sklearn.metrics import roc_curve, auc
import cv2

len_train = 1000
len_CV = 100

class Gaussian:
    def __init__(self, mu, sigma):
        """Single Gaussian Distribution"""
        self.mu = mu
        self.sigma = sigma
        
    def pdf(self, data):
        term1 = -0.5 * (data - self.mu).T
        term2 = inv(self.sigma)
        term3 = data - self.mu
        expo1 = np.matmul(term1, term2)
        expo2 = np.matmul(expo1, term3)[0, 0]
        val = np.exp(expo2)
        val = val / np.sqrt(np.linalg.det(self.sigma))
        val = val / ((2*np.pi) ** (100/2))
        return val


print("Loading Data")
X_train_face = load_train_data('face')
X_train_nonface = load_train_data('nonface')
X_CV_nonface = load_CV_data('nonface')
X_CV_face = load_CV_data('face')


print("Training")

mean_face, covar_face = get_MC(X_train_face)
mix_face = Gaussian(mean_face.reshape(-1, 1), covar_face)

mean_nonface, covar_nonface = get_MC(X_train_nonface)
mix_nonface = Gaussian(mean_nonface.reshape(-1, 1), covar_nonface)

print("Testing")

prob_face_facedata = array([])
prob_nonface_facedata = array([])
prob_face_nonfacedata = array([])
prob_nonface_nonfacedata = array([])

for i in range(len_CV):
    inp_facedata = X_CV_face[i].reshape(-1, 1)
    inp_nonfacedata = X_CV_nonface[i].reshape(-1, 1)

    prob_face_facedata = np.append(prob_face_facedata, mix_face.pdf(inp_facedata))
    prob_nonface_facedata = np.append(prob_nonface_facedata, mix_nonface.pdf(inp_facedata))
    prob_face_nonfacedata = np.append(prob_face_nonfacedata, mix_face.pdf(inp_nonfacedata))
    prob_nonface_nonfacedata = np.append(prob_nonface_nonfacedata, mix_nonface.pdf(inp_nonfacedata))

post_face_face_data = prob_face_facedata/(prob_face_facedata + prob_nonface_facedata )
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

print("Visualization")

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

face_variance1 = np.diag(covar_face)
min_val = np.min(face_variance1)
max_val = np.max(face_variance1)
face_variance1 = ((face_variance1-min_val)/(max_val-min_val) * 255.0).astype('uint8')
face_variance = np.reshape(face_variance1, (10, 10))

nonface_variance1 = np.diag(covar_nonface)
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