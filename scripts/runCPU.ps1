param(
[string]$binariesFile = "serialTBBSSE.txt",
[string]$iterations = "500",
[string]$clusters = "32",
[string]$inputDataFile = "smallData2D.txt"
)
foreach ($data in [System.IO.File]::ReadLines($inputDataFile)) {
	foreach ($line in [System.IO.File]::ReadLines($binariesFile)) {
		$name = $line.Split("{\\}")[1]
		if((Test-Path $name) -eq 0)
		{
			mkdir $name;
		}
		$arguments = "../data/data" + $data + ".dat " + $name + "/means" + $data + ".dat " + $name +"/clusters" + $data + ".dat " + $clusters + " " + $iterations
		$tempOutput = $name + "/temp.txt"
		$output = $name + "/" + $inputDataFile + "times.txt"
		Start-Process -FilePath $line -wait -ArgumentList $arguments -RedirectStandardOutput $tempOutput
		Get-Content $tempOutput | Out-File $output -Append
	}
}