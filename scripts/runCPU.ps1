param(
[string]$binariesFile = "serialTBBSSE.txt",
[string]$iterations = "500",
[string]$clusters = "32",
[string]$inputDataFile = "smallData.txt",
[string]$dimensionsFile = "dimensions.txt",
[string]$dataType = "F",
[string]$distribution = "N",
[string]$dataClusters = $clusters
)

$minDimension = Get-Content $dimensionsFile -First 1

(Get-Item -Path ".\" -Verbose).FullName

foreach ($dataSize in [System.IO.File]::ReadLines($inputDataFile)) {
	foreach ($line in [System.IO.File]::ReadLines($binariesFile)) {
		$name = $line.Split("{\\}")[1]
		if((Test-Path $name) -eq 0)
		{
			mkdir $name;
		}
		$output = $name + "Times.dat"
		foreach ($dimension in [System.IO.File]::ReadLines($dimensionsFile)) {
			$dataName = $dataType + $distribution + $dimension + "D" + $dataSize + "K" + $dataClusters + "C"
			$arguments = "../data/data" + $dataName + ".dat " + $name + "/means" + $dataName + ".dat " + $name +"/clusters" + $dataName + ".dat " + $clusters + " " + $iterations
			$tempOutput = $name + "/temp.txt"
			"=======" + $name + "  " + $dataName + "======="
			Start-Process -FilePath $line -wait -ArgumentList $arguments -RedirectStandardOutput $tempOutput
			$time = Get-Content $tempOutput -First 1 #[System.IO.File]::ReadLines($tempOutput)[0]
			"Time: " + $time
			if ($dimension -ne $minDimension) {
				$time = ";" + $time
			}
			#Add-Content $output $time
			#Get-Content $tempOutput | Out-File $output -Append
			[System.IO.File]::AppendAllText($output, $time)
		}
		Add-Content $output ""
	}
}