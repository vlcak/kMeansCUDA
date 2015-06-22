setwd("D:/Git/diplomka/scripts")

TBBtimes = read.table("k-meansTBBTimes.dat", sep = ";")
SSEtimes = read.table("k-meansTBB_SSETimes.dat", sep = ";")
SerialTimes = read.table("k-means-serialTimes.dat", sep = ";")

index=5

dimension=c("2D","4D","8D","16D","32D","64D","96D","128D","160D","192D","224D","256D")

g_range <- range(0, TBBtimes[index], SSEtimes[index], SerialTimes[index])

# Graph cars using a y axis that ranges from 0 to 12
plot(TBBtimes[,index], type="o", col="blue", ylim=g_range, xaxt="n", ann=FALSE)

grid()

axis(1, at=1:5, lab=c("8K","16K","32K","65K","131K"))
box()

# Graph trucks with red dashed line and square points
lines(SSEtimes[,index], type="o", pch=22, lty=2, col="red")

# Graph trucks with red dashed line and square points
lines(SerialTimes[,index], type="o", pch=23, lty=3, col="forestgreen")

# Create a title with a red, bold/italic font

title(main=dimension[index])
title(xlab="Size")
title(ylab="Time")

legend(1, g_range[2], c("TBB","TBB with SSE", "Serial"), cex=0.8, 
   col=c("blue", "red", "forestgreen"), pch=21:23, lty=1:3);
