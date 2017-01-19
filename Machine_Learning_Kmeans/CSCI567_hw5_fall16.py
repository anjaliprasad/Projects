import kmean
import kernel_kmean
import EM
print "Kmeans running...."
kmean.main("hw5_blob.csv")
kmean.main("hw5_circle.csv")
print "Kernel Kmeans running...."
kernel_kmean.main("hw5_circle.csv")
print "EM running...."
EM.main("hw5_blob.csv")