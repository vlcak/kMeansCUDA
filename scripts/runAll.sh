#!/usr/bin/bash

inputDataFile="$1"
binariesFile="$2"
dimensionsFile="$3"
dataType="F" 
iterations="500"
clusters="32" 
distribution="N"
dataClusters=$clusters

minDimension=$(head -n 1 $dimensionsFile)

while IFS='' read -r dataSize || [[ -n $dataSize ]]; do
    while IFS='' read -r bin || [[ -n $bin ]]; do  
        name=$(cut -d/ -f2 <<<"${bin}") 
        mkdir -p $name
		    output=$name"Times.dat"
        while IFS='' read -r dimension || [[ -n $dimension ]]; do
            dataName=$dataType$distribution$dimension"D"$dataSize"K"$dataClusters"C"
            arguments=" ../data/data"$dataName".dat "$name"/means"$dataName".dat "$name"/clusters"$dataName".dat "$clusters" "$iterations
      			tempOutput=$name"/temp.txt"
            command=$bin$arguments
      			echo "======="$name"  "$dataName"======="
            eval $command > $tempOutput
            time=$(head -n 1 $tempOutput)
            echo "Time: "$time
            if [ "$dimension" != "$minDimension" ]; then
      			   time=";"$time
      			fi
            printf "%s" "$time" >> $output
        done < "$dimensionsFile"
        echo "\n" >> $output
    done < "$binariesFile"
done < "$inputDataFile"