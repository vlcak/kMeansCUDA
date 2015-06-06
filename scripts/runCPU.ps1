param(
[string]$binariesFile = "serialTBBSSE.txt",
[string]$iterations = "500",
[string]$clusters = "32",
[string]$inputDataFile = "smallData.txt",
[string]$minDimension = "2",
[string]$maxDimension = "256",
[string]$dataType = "F",
[string]$distribution = "N",
[string]$dataClusters = $clusters
)

$inputDataFile

(Get-Item -Path ".\" -Verbose).FullName

for ($dimension = [convert]::ToInt32($minDimension, 10); $dimension -le [convert]::ToInt32($maxDimension, 10); $dimension *= 2) {
	foreach ($dataSize in [System.IO.File]::ReadLines($inputDataFile)) {
		foreach ($line in [System.IO.File]::ReadLines($binariesFile)) {
			$name = $line.Split("{\\}")[1]
			if((Test-Path $name) -eq 0)
			{
				mkdir $name;
			}
			$dataName = $dataType + $distribution + $dimension + "D" + $dataSize + "K" + $dataClusters + "C"
			$arguments = "../data/data" + $dataName + ".dat " + $name + "/means" + $dataName + ".dat " + $name +"/clusters" + $dataName + ".dat " + $clusters + " " + $iterations
			$tempOutput = $name + "/temp.txt"
			$output = $name + "/" + $dataName + "times.txt"
			"=======" + $name + "  " + $dataName + "======="
			Start-Process -FilePath $line -wait -ArgumentList $arguments -RedirectStandardOutput $tempOutput
			"Time: " + [System.IO.File]::ReadLines($tempOutput)
			Get-Content $tempOutput | Out-File $output -Append
		}
	}
}